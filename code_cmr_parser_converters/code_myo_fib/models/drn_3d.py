import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from models.pl_base_model import ModelBase

BatchNorm = nn.BatchNorm3d

def conv1x3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                     padding=(0, padding, padding), bias=False, dilation=(1, dilation, dilation))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                               padding=(0, dilation[1], dilation[1]), bias=False,
                               dilation=(1, dilation[1], dilation[1]))
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DRN3D(nn.Module):

    def __init__(self, depths, block="basic", num_classes=1000,
                 dims=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, arch='D',
                 in_chans=1, downsampling=8):

        super().__init__()
        self.inplanes = dims[0]
        self.out_map = out_map
        self.out_dim = dims[-1]
        self.out_middle = out_middle
        self.arch = arch
        self.num_classes = num_classes
        self.in_chans = in_chans

        assert downsampling in [4, 8]
        stride4 = 2 if downsampling == 8 else 1

        if block == "basic":
            block = BasicBlock
        else:
            block = Bottleneck

        if arch == 'C':
            self.conv1 = nn.Conv3d(self.in_chans, dims[0], kernel_size=(1, 7, 7), stride=1,
                                   padding=(0, 3, 3), bias=False)
            self.bn1 = BatchNorm(dims[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(
                BasicBlock, dims[0], depths[0], stride=1)
            self.layer2 = self._make_layer(
                BasicBlock, dims[1], depths[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv3d(self.in_chans, dims[0], kernel_size=(1, 7, 7), stride=(1, 1, 1), padding=(0, 3, 3),
                          bias=False),
                BatchNorm(dims[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(
                dims[0], depths[0], stride=1)
            self.layer2 = self._make_conv_layers(
                dims[1], depths[1], stride=2)

        self.layer3 = self._make_layer(block, dims[2], depths[2], stride=2)
        self.layer4 = self._make_layer(block, dims[3], depths[3], stride=stride4)
        self.layer5 = self._make_layer(block, dims[4], depths[4],
                                       dilation=2, new_level=False)
        self.layer6 = None if depths[5] == 0 else \
            self._make_layer(block, dims[5], depths[5], dilation=4,
                             new_level=False)

        if arch == 'C':
            self.layer7 = None if depths[6] == 0 else \
                self._make_layer(BasicBlock, dims[6], depths[6], dilation=2,
                                 new_level=False, residual=False)
            self.layer8 = None if depths[7] == 0 else \
                self._make_layer(BasicBlock, dims[7], depths[7], dilation=1,
                                 new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if depths[6] == 0 else \
                self._make_conv_layers(dims[6], depths[6], dilation=2)
            self.layer8 = None if depths[7] == 0 else \
                self._make_conv_layers(dims[7], depths[7], dilation=1)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, dims, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv3d(self.inplanes, dims, kernel_size=(1, 3, 3),
                          stride=(1, stride, stride) if i == 0 else 1,
                          padding=(0, dilation, dilation), bias=False, dilation=(1, dilation, dilation)),
                BatchNorm(dims),
                nn.ReLU(inplace=True)])
            self.inplanes = dims
        return nn.Sequential(*modules)

    def forward_features(self, x):
        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        if self.layer6 is not None:
            x = self.layer6(x)
        if self.layer7 is not None:
            x = self.layer7(x)
        if self.layer8 is not None:
            x = self.layer8(x)

        return x

    def fan_out(self):
        return self.out_dim






def drn_c_26(pretrained=False, **kwargs):
    model = DRN3D(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-26']))
    return model


def drn_c_42(pretrained=False, **kwargs):
    model = DRN3D(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-42']))
    return model


def drn_c_58(pretrained=False, **kwargs):
    model = DRN3D(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-58']))
    return model


def drn_d_22(pretrained=False, **kwargs):
    model = DRN3D(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-22']))
    return model


def drn_d_24(pretrained=False, **kwargs):
    model = DRN3D(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-24']))
    return model


def drn_d_38(pretrained=False, **kwargs):
    model = DRN3D(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-38']))
    return model


def drn_d_40(pretrained=False, **kwargs):
    model = DRN3D(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-40']))
    return model


def drn_d_54(pretrained=False, **kwargs):
    model = DRN3D(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-54']))
    return model


def drn_d_56(pretrained=False, **kwargs):
    model = DRN3D(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-56']))
    return model


def drn_d_105(pretrained=False, **kwargs):
    model = DRN3D(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-105']))
    return model


def drn_d_107(pretrained=False, **kwargs):
    model = DRN3D(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-107']))
    return model



if __name__ == "__main__":
    model = drn_d_22()
    print(model)
