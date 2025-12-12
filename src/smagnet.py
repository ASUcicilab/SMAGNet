# smagnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url
from torch.nn.modules import Conv2d, Module
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import modules as md

import numpy as np
import argparse
import os
from tqdm import tqdm
import copy

from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

from typing import Optional, Union, List
import pandas as pd
import glob
import re

### common modules

# Initialize weights for decoder modules
def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# Initialize weights for segmentation head modules
def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# Initialize gated fusion layers
def initialize_gate(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d)):
            nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        
# Wrapper for various activation functions.
class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)
    

# Convolution -> optional upsampling -> activation
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


# Center block used in decoder.
class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            #use_batchnorm=use_batchnorm,
            use_norm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            #use_batchnorm=use_batchnorm,
            use_norm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


# Single decoder block in decoder.
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
        scale_factor=2,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            #use_batchnorm=use_batchnorm,
            use_norm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            #use_batchnorm=use_batchnorm,
            use_norm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        self.scale_factor = scale_factor

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


# Gated feature fusion for SAR and MSI features.
# Modes:
# - sar_only: only use SAR features
# - sar_msi_gated: learnable gate to combine SAR and MSI features
# Optional spatial_mask can modulate the gate to ignore invalid regions.

class SARMSI_Gated_Feature_Fusion(nn.Module):
    def __init__(self, in_channels, 
                 sarmsiff_method='sar_msi_gated'):
        
        super().__init__()
        
        if sarmsiff_method != 'sar_only':
            # Conv2d
            layers = [nn.Conv2d(in_channels * 2, 1, kernel_size=1)]
            
            # sigmoid
            layers.append(nn.Sigmoid())
            
            self.gate = nn.Sequential(*layers)
        
            print(self.gate) 
        
        print(f'sarmsiff_method: {sarmsiff_method} - OK')
        self.sarmsiff_method = sarmsiff_method
        
    def forward(self, feat_sar, feat_msi, spatial_mask=None):
        if self.sarmsiff_method == 'sar_only':
            feat_ret = feat_sar_norm
            feat_gate = None
        else:
            feat_combined = torch.cat([feat_sar, feat_msi], dim=1)
            feat_gate = self.gate(feat_combined)

            # applying the mask to the gate
            if spatial_mask is not None:
                # mask is invalid mask(valid: 0, invalid: 1)
                spatial_mask = F.adaptive_avg_pool2d(spatial_mask, (feat_gate.shape[2], feat_gate.shape[3]))
                feat_gate = (1.0 - spatial_mask) * feat_gate
                # print(f'feat_gate.shape: {feat_gate.shape}')
                # print(f'mask.shape: {mask.shape}')
                
            if self.sarmsiff_method == 'sar_msi_gated':
                feat_ret = (1-feat_gate) * feat_sar + feat_gate * feat_msi
            else:
                raise ValueError("No matched ff_method option")
                
        return feat_ret, feat_gate     

        
# U-Net style decoder with multiple DecoderBlocks.
class Decoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=False,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):
        
        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


# SMAGNet_SegmentationModel
class SMAGNet_SegmentationModel(torch.nn.Module):
    def initialize(self):
        initialize_decoder(self.decoder_shared)
        print('Complete decoder initialization.')
        
        initialize_head(self.segmentation_head_shared)
        print('Complete segmentation header initialization.')
        
        initialize_gate(self.fuse_blocks)
        print('Complete fusion module initialization.')
        
    def check_input_shape(self, x, output_stride):
        h, w = x.shape[-2:]
        
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x_sar, x_msi, spatial_mask = None): 
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # check input shape
        self.check_input_shape(x_sar, self.encoder_sar.output_stride)
        self.check_input_shape(x_msi, self.encoder_msi.output_stride)
        
        # generate features through encoders 
        feat_sar = self.encoder_sar(x_sar)
        feat_msi = self.encoder_msi(x_msi) 

        # feature arrangement (0: first up path, 1~4: skip cons)
        feat_sar = feat_sar[1:]  # remove first skip with same spatial resolution
        feat_sar = feat_sar[::-1]  # reverse channels to start from head of encoder

        feat_msi = feat_msi[1:]  # remove first skip with same spatial resolution
        feat_msi = feat_msi[::-1]  # reverse channels to start from head of encoder
        
        # shared decoder - sar
        decoder_output_sar = self.decoder_shared(feat_sar)

        # segment head - sar
        seghead_output_sar = self.segmentation_head_shared(decoder_output_sar)

        # feature fusion
        fused_features = list()
        gate_map = list()

        if self.enable_spatial_mask == False:
            spatial_mask = None
            
        for i, fuse_blk in enumerate(self.fuse_blocks):
            feat_fused, feat_gate = fuse_blk(feat_sar[i], feat_msi[i], spatial_mask = spatial_mask)
            fused_features.append(feat_fused)
            gate_map.append(feat_gate)

        # shared decoder - sarmsiff
        decoder_output_sarmsiff = self.decoder_shared(fused_features)
        
        # segment head - sarmsiff 
        seghead_output_sarmsiff = self.segmentation_head_shared(decoder_output_sarmsiff)
        
        return seghead_output_sarmsiff, seghead_output_sar, gate_map 

    @torch.no_grad()
    def predict(self, x_sar, x_msi, spatial_mask = None):
        if self.training:
            self.eval()

        seghead_output_sarmsiff, seghead_output_sar, gate_map = self.forward(x_sar, x_msi, spatial_mask)

        return seghead_output_sarmsiff, seghead_output_sar, gate_map

# SMAGNet
class SMAGNet(SMAGNet_SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights_sar: Optional[str] = None,
        encoder_weights_msi: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = False,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        decoder_attention_type: Optional[str] = None,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        sarmsiff_method = 'sar_msi_gated', # sar_only, sar_msi_gated
        enable_spatial_mask = True,
    ):
        super().__init__()

        # enabling invalid masking
        self.enable_spatial_mask = enable_spatial_mask
        print(f'enable_spatial_mask: {enable_spatial_mask} - OK')
        
        # encoder 
        self.encoder_sar = get_encoder(
            encoder_name,
            in_channels=2, # vv, vh
            depth=encoder_depth,
            weights=encoder_weights_sar,
        )
        
        self.encoder_msi = get_encoder(
            encoder_name,
            in_channels=4, # rgb, nir
            depth=encoder_depth,
            weights=encoder_weights_msi,
        )

        if encoder_weights_sar is not None:
            print(f'pretrained {encoder_weights_sar} is applied to encoder_sar - OK')
            
        if encoder_weights_msi is not None:
            print(f'pretrained {encoder_weights_msi} is applied to encoder_msi - OK')

        # spatially masked gated feature fusion
        encoder_channels = self.encoder_sar.out_channels
        encoder_channels = encoder_channels[1:]
        fusion_channels = encoder_channels[::-1]
        
        fuse_blocks = [
            SARMSI_Gated_Feature_Fusion(fusion_ch, sarmsiff_method)
            for fusion_ch in fusion_channels
        ]
            
        self.fuse_blocks = nn.ModuleList(fuse_blocks)

        print(f'fusion_channels: {fusion_channels}')
        print(f'sarmsiff_method: {sarmsiff_method}')
        
        # shared decoder
        self.decoder_shared = Decoder(
            encoder_channels=self.encoder_sar.out_channels, 
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        
        print(f"shared_decoder class: {self.decoder_shared.__class__.__name__}")
        print(f'shared_decoder channels: {decoder_channels}')
        print(f'shared_decoder use_batchnorm: {decoder_use_batchnorm}')
        
        # segment head
        self.segmentation_head_shared = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        
        self.name = "u-{}".format(encoder_name)
        self.initialize()
