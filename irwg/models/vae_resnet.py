from math import prod
from typing import List, Optional, Tuple, Union

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

"""
Adapted from https://github1s.com/Lightning-AI/lightning-bolts/blob/d1083298b057ea3775ab8fc275c7a7cc2edb6850/pl_bolts/models/autoencoders/components.p
"""

class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def resize_conv3x3(in_planes, out_planes, scale=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes))


def resize_conv1x1(in_planes, out_planes, scale=1):
    """upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes))

class EncoderBlock(nn.Module):
    """ResNet block, copied from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_prob=0.):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetEncoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], layers: List[int], layer_widths: List[int], latent_dim: int, first_conv: bool=False, maxpool1: bool=False, dropout_prob:float=0.):
        super().__init__()

        self.input_shape = input_shape
        self.inplanes = layer_widths[0]
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.latent_dim = latent_dim

        if self.first_conv:
            self.conv1 = nn.Conv2d(input_shape[0], self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(input_shape[0], self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        assert len(layers) == len(layer_widths) and len(layers) > 0

        self.layers = nn.ModuleList()
        for i, (n_layers, width) in enumerate(zip(layers, layer_widths)):
            self.layers.append(self._make_layer(EncoderBlock, width, n_layers, stride=(1 if i == 0 else 2), dropout_prob=dropout_prob))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.final = nn.Linear(in_features=layer_widths[-1], out_features=latent_dim)

    def _make_layer(self, block, planes, blocks, stride=1, dropout_prob=0.):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_prob=dropout_prob))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_prob=dropout_prob))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, *self.input_shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for l in self.layers:
            x = l(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Find the index of the dim where the datapoint dimensions start as given by the input_shape
        # This allows processing of 1d, 2d or higher order datapoints
        dim = np.nonzero(np.cumprod(x_shape[::-1])[::-1] == np.prod(self.input_shape))[0][-1].item()

        x = x.view(*x_shape[:dim], -1)
        x = self.final(x)
        return x

class DecoderBlock(nn.Module):
    """ResNet block, but convs replaced with resize convs, and channel increase is in second conv, not first."""

    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None, dropout_prob=0.):
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.dropout(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetDecoder(nn.Module):
    """Resnet in reverse order."""

    def __init__(self, output_shape: Tuple[int, int, int], layers: List[int], layer_upscale: List[Optional[int]], layer_widths: List[int], downscaled_resolution: Tuple[int, int], latent_dim: int, upscale_factor: int, first_conv: bool=False, maxpool1: bool=False, dropout_prob: float=0.):
        super().__init__()

        self.output_shape = output_shape
        self.expansion = DecoderBlock.expansion
        self.layer_widths = layer_widths
        self.inplanes = layer_widths[0]*2 * DecoderBlock.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = output_shape[-2]
        self.downscaled_resolution = downscaled_resolution
        self.latent_dim = latent_dim

        self.upscale_factor = upscale_factor

        self.linear = nn.Linear(latent_dim, self.inplanes * prod(downscaled_resolution))

        assert len(layers) == len(layer_widths) and len(layers) > 0

        self.layers = nn.ModuleList()
        for i, (n_layers, width) in enumerate(zip(layers, layer_widths)):
            if i < len(layers)-1:
                self.layers.append(self._make_layer(DecoderBlock, width, n_layers, scale=2, dropout_prob=dropout_prob))
            else:
                if self.maxpool1:
                    self.layers.append(self._make_layer(DecoderBlock, width, n_layers, scale=2, dropout_prob=dropout_prob))
                    self.upscale_factor *= 2
                else:
                    self.layers.append(self._make_layer(DecoderBlock, width, n_layers, dropout_prob=dropout_prob))

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        self.upscales = nn.ModuleList()
        for i, up in enumerate(layer_upscale):
            if up is not None:
                self.upscales.append(
                    Interpolate(size=up)
                )
            else:
                self.upscales.append(None)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=self.input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(layer_widths[-1] * DecoderBlock.expansion, self.output_shape[0], kernel_size=3, stride=1, padding=1, bias=False)

    @property
    def input_dim(self):
        return self.linear.in_features

    def _make_layer(self, block, planes, blocks, scale=1, dropout_prob=0.):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample, dropout_prob=dropout_prob))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_prob=dropout_prob))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])

        x = self.linear(x)

        x = x.view(x.size(0), self.layer_widths[0]*2*self.expansion, *self.downscaled_resolution)
        x = self.upscale1(x)

        for i, l in enumerate(self.layers):
            x = l(x)
            if self.upscales[i] is not None:
                x = self.upscales[i](x)
        x = self.upscale(x)

        x = self.conv1(x)

        x = x.view(*x_shape[:-1], -1)
        return x

