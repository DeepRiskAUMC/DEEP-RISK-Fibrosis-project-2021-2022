import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
import pytorch_lightning as pl

from models.pl_base_model import ModelBase



class LightningSimpleNet(ModelBase):
    def __init__(self, highres=False, num_classes=1, in_chans=1, **kwargs):
        # arguments that are only necessary for ModelBase are in kwargs
        super().__init__(num_classes=num_classes, final_feature_size=128, **kwargs)
        print(f"Model {kwargs=}")
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.highres = highres
        self.save_hyperparameters()

        if highres == False:
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(self.in_chans, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(kernel_size=2, stride=2)

            )
        else:
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(self.in_chans, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(kernel_size=2, stride=1)
            )



        #self.classifier = nn.Conv2d(128, self.num_classes, 1, bias=True)


    def forward_features(self, x):
        x = self.cnn_layers(x)
        return x


if __name__ == "__main__":
    model = LightningSimpleNet(lr=1e-1, highres=True, schedule='step')
    print(model)
    print(model.hparams)
