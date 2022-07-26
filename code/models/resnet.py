import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
import pytorch_lightning as pl

from models.pl_base_model import ModelBase


class LightningResNet(ModelBase):
    def __init__(self, no_cam=False, highres=False, pretrain=False, model='resnet18', num_classes=1, in_chans=1, **kwargs):
        # arguments that are only necessary for ModelBase are in kwargs
        super().__init__(num_classes=num_classes, final_feature_size=512, **kwargs)
        print(f"Model {kwargs=}")
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.highres = highres
        self.model = model
        self.pretrain = pretrain
        self.no_cam = no_cam
        self.save_hyperparameters()

        # load model
        if model == 'resnet18':
            normal_resnet = models.resnet18(pretrained=pretrain)
        elif model == 'resnet50':
            normal_resnet = models.resnet50(pretrained=pretrain)
        elif model == 'resnet101':
            normal_resnet = models.resnet101(pretrained=pretrain)

        # build stem
        self.conv1 =  nn.Conv2d(self.in_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = normal_resnet.bn1
        self.relu = normal_resnet.relu
        self.maxpool = normal_resnet.maxpool
        self.stem = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)

        # build stages
        if highres == False:
            self.stages = nn.Sequential(normal_resnet.layer1,
                                        normal_resnet.layer2,
                                        normal_resnet.layer3,
                                        normal_resnet.layer4)
            last_conv_fmaps = normal_resnet.fc.weight.shape[1]
        else:
            self.stages = nn.Sequential(normal_resnet.layer1,
                                        normal_resnet.layer2,
                                        normal_resnet.layer3,
                                        nn.Identity())
            last_conv_fmaps = int(normal_resnet.fc.weight.shape[1] / 2)

        # build head
        if no_cam == True:
            raise NotImplementedError
            self.fc = nn.Linear(last_conv_fmaps, self.num_classes, bias=True)
            self.head = nn.Sequential(self.avgpool, nn.Flatten(), self.fc)
        #elif no_cam == False:
        #    self.classifier = nn.Conv2d(last_conv_fmaps, self.num_classes, 1, bias=True)


    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        #x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = models.resnet18()
    print(model)
