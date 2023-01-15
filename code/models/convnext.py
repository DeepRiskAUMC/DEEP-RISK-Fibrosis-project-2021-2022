# source : https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pl_classification_model_2d import ModelBase
from timm.models.layers import DropPath, trunc_normal_


class LightningConvNeXt(ModelBase):
    r"""
    ConvNext, but  1. Pytorch lightning. 2. With CAM 3. Extra options wrt stem layer and normalization layers
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1 (num_classes=1 makes use of binary classification)
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        stem_type (str): Type of initial downsampling to use. "patch" for regular ConvNext stem, "resnet_like" for conv stride 2 followed by maxpool stride 2.
        norm_type (str): Type of normalization. "layer" for layer normalization, "batch" for batch normalization.
        no_cam (Boolean): Whether to use class activation maps or not. Default False (i.e. do use CAM)
    """
    def __init__(self, in_chans=3, num_classes=1,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 strides=[4, 2, 2, 2], stem_type="patch", norm_type="layer",
                 no_cam=False, **kwargs
            ):
        super(LightningConvNeXt, self).__init__(num_classes=num_classes, final_feature_size=dims[-1], **kwargs)
        print(f"Model {kwargs=}")
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.head_init_scale = head_init_scale
        self.strides = strides
        self.stem_type = stem_type
        self.norm_type = norm_type
        self.no_cam = no_cam
        self.save_hyperparameters()


        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        # select normalization type
        if stem_type not in ["patch", "resnet_like"]:
            raise NotImplementedError
        if norm_type not in ["layer", "batch"]:
            raise NotImplementedError
        init_norms = nn.ModuleList()
        if norm_type == "batch":
            init_norms.append(nn.BatchNorm2d(dims[0]))
            for i in range(3):
                init_norms.append(nn.BatchNorm2d(dims[i]))
        elif norm_type == "layer":
            init_norms.append(LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
            for i in range(3):
                init_norms.append(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"))


        if stem_type == "patch":
                stem = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=strides[0], stride=strides[0]),
                    init_norms[0]
                )
        elif stem_type == "resnet_like":
                stem = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                    init_norms[0],
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
                )
        self.downsample_layers.append(stem)


        for i in range(3):
            downsample_layer = nn.Sequential(
                init_norms[i+1],
                nn.Conv2d(dims[i], dims[i+1], kernel_size=strides[i+1], stride=strides[i+1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], norm_type=self.norm_type, drop_path=dp_rates[cur + j],
                layer_scale_init_value=self.layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # changes for Class Activation Maps
        if no_cam == True:
            raise NotImplementedError
            if norm_type == "batch":
                self.norm = nn.BatchNorm1d(dims[-1])
            elif norm_type == "layer":
                self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
            self.head = nn.Linear(dims[-1], num_classes)
            self.apply(self._init_weights)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        elif no_cam == False:
            if norm_type == "batch":
                self.norm = nn.BatchNorm2d(dims[-1])
            elif norm_type == "layer":
                self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x




class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, norm_type="layer"):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm_type = norm_type
        if self.norm_type not in ["batch", "layer"]:
            raise NotImplementedError
        if self.norm_type == "batch":
            self.norm = nn.BatchNorm2d(dim)
        elif self.norm_type == "layer":
            self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        if self.norm_type == "batch":
            x = self.norm(x)
            x = x.permute(0, 2, 3, 1)
        elif self.norm_type == "layer":
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
