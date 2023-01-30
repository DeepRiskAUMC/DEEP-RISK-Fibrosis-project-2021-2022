# from msilib.schema import Feature
import torch
import torch.nn as nn
from models.unet_components import *
from models.canet_components import *

class Simple_2d_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, adapted=False):
        super(Simple_2d_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)   
        self.down1 = Down(64, 128)              #maxpool 2x2, doubleConv(64-128-128)
        self.down2 = Down(128, 256)             #maxpool 2x2, doubleConv(128-256-256)
        self.down3 = Down(256, 512)             #maxpool 2x2, doubleConv(256-512-512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  #bileair: maxpool 2x2, doubleConv(512-512-512)    #non-bilineair: maxpool 2x2, doubleConv(512-1024-1024)
        if adapted:
            self.up1 = UpAdapted(1024, 512 // factor, bilinear)
            self.up2 = UpAdapted(512, 256 // factor, bilinear)
            self.up3 = UpAdapted(256, 128 // factor, bilinear)
            self.up4 = UpAdapted(128, 64, bilinear)
        else:
            self.up1 = Up(1024, 512 // factor, bilinear)    #bilineair: upsample(scale=2), doubleConv(1024-256-512)     #non-bilineair: convtrans(2x2:1024-512), doubleConv(1024-512-512)
            self.up2 = Up(512, 256 // factor, bilinear)     #bilineair: upsample(scale=2), doubleConv(512-128-256)      #non-bilineair: convtrans(2x2:512-256), doubleConv(512-256-256)
            self.up3 = Up(256, 128 // factor, bilinear)     #bilineair: upsample(scale=2), doubleConv(256-64-128)       #non-bilineair: convtrans(2x2:256-128), doubleConv(256-128-128)
            self.up4 = Up(128, 64, bilinear)                #bilineair: upsample(scale=2), doubleConv(128-64-64)        #non-bilineair: convtrans(2x2:128-64), doubleConv(128-64-64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = nn.Sigmoid()(logits)
        return logits

class Floor_2d_Unet_without_final_conv(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, feature_multiplication=4):
        super(Floor_2d_Unet_without_final_conv, self).__init__()
        self.bilinear = bilinear
        print(f'bilinear: {bilinear}')
        self.feature_multiplication = feature_multiplication
        #Encoder
        self.double_conv_down1 = Double_conv_floor(n_channels, int(16*self.feature_multiplication))
        self.double_conv_down2 = Double_conv_floor(int(16*self.feature_multiplication), int(32*self.feature_multiplication))
        self.double_conv_down3 = Double_conv_floor(int(32*self.feature_multiplication), int(64*self.feature_multiplication))
        self.double_conv_down4 = Double_conv_floor(int(64*self.feature_multiplication), int(128*self.feature_multiplication))
        factor = 2 if bilinear else 1
        self.double_conv_down5 = Double_conv_floor(int(128*self.feature_multiplication), int(256//factor*self.feature_multiplication))
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        #Decoder
        self.pad_concat = Concat_and_pad()
        if bilinear:
            self.deconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.deconv1 = nn.ConvTranspose2d(int(256*self.feature_multiplication), int(128*self.feature_multiplication), kernel_size=2, stride=2)
            self.deconv2 =  nn.ConvTranspose2d(int(128*self.feature_multiplication), int(64*self.feature_multiplication), kernel_size=2, stride=2)
            self.deconv3 =  nn.ConvTranspose2d(int(64*self.feature_multiplication), int(32*self.feature_multiplication), kernel_size=2, stride=2)
            self.deconv4 =  nn.ConvTranspose2d(int(32*self.feature_multiplication), int(16*self.feature_multiplication), kernel_size=2, stride=2)
        self.double_conv_up1 = Double_conv_floor(int(256*self.feature_multiplication), int(128//factor*self.feature_multiplication))
        self.double_conv_up2 = Double_conv_floor(int(128*self.feature_multiplication), int(64//factor*self.feature_multiplication))
        self.double_conv_up3 = Double_conv_floor(int(64*self.feature_multiplication), int(32//factor*self.feature_multiplication))
        self.double_conv_up4 = Double_conv_floor(int(32*self.feature_multiplication), int(16*self.feature_multiplication))

    def forward(self, x):
        x1 = self.double_conv_down1(x)
        x2 = self.maxpool(x1)
        x2 = self.double_conv_down2(x2)
        x3 = self.maxpool(x2)
        x3 = self.double_conv_down3(x3)
        x4 = self.maxpool(x3)
        x4 = self.double_conv_down4(x4)
        x = self.maxpool(x4)
        x = self.double_conv_down5(x)

        x = self.deconv1(x)
        x = self.pad_concat(x, x4)
        x = self.double_conv_up1(x)
        x = self.deconv2(x)
        x = self.pad_concat(x, x3)
        x = self.double_conv_up2(x)
        x = self.deconv3(x)
        x = self.pad_concat(x, x2)
        x = self.double_conv_up3(x)
        x = self.deconv4(x)
        x = self.pad_concat(x, x1)
        x = self.double_conv_up4(x)

        return x

class Floor_2d_Unet_without_final_conv_multiple_outputs(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, feature_multiplication=4):
        super(Floor_2d_Unet_without_final_conv, self).__init__()
        self.bilinear = bilinear
        print(f'bilinear: {bilinear}')
        self.feature_multiplication = feature_multiplication
        #Encoder
        self.double_conv_down1 = Double_conv_floor(n_channels, int(16*self.feature_multiplication))
        self.double_conv_down2 = Double_conv_floor(int(16*self.feature_multiplication), int(32*self.feature_multiplication))
        self.double_conv_down3 = Double_conv_floor(int(32*self.feature_multiplication), int(64*self.feature_multiplication))
        self.double_conv_down4 = Double_conv_floor(int(64*self.feature_multiplication), int(128*self.feature_multiplication))
        factor = 2 if bilinear else 1
        self.double_conv_down5 = Double_conv_floor(int(128*self.feature_multiplication), int(256//factor*self.feature_multiplication))
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        #Decoder
        self.pad_concat = Concat_and_pad()
        if bilinear:
            self.deconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.deconv1 = nn.ConvTranspose2d(int(256*self.feature_multiplication), int(128*self.feature_multiplication), kernel_size=2, stride=2)
            self.deconv2 =  nn.ConvTranspose2d(int(128*self.feature_multiplication), int(64*self.feature_multiplication), kernel_size=2, stride=2)
            self.deconv3 =  nn.ConvTranspose2d(int(64*self.feature_multiplication), int(32*self.feature_multiplication), kernel_size=2, stride=2)
            self.deconv4 =  nn.ConvTranspose2d(int(32*self.feature_multiplication), int(16*self.feature_multiplication), kernel_size=2, stride=2)
        self.double_conv_up1 = Double_conv_floor(int(256*self.feature_multiplication), int(128//factor*self.feature_multiplication))
        self.double_conv_up2 = Double_conv_floor(int(128*self.feature_multiplication), int(64//factor*self.feature_multiplication))
        self.double_conv_up3 = Double_conv_floor(int(64*self.feature_multiplication), int(32//factor*self.feature_multiplication))
        self.double_conv_up4 = Double_conv_floor(int(32*self.feature_multiplication), int(16*self.feature_multiplication))

    def forward(self, x):
        x1 = self.double_conv_down1(x)
        x2 = self.maxpool(x1)
        x2 = self.double_conv_down2(x2)
        x3 = self.maxpool(x2)
        x3 = self.double_conv_down3(x3)
        x4 = self.maxpool(x3)
        x4 = self.double_conv_down4(x4)
        x = self.maxpool(x4)
        x_out1 = self.double_conv_down5(x)

        x = self.deconv1(x_out1)
        x = self.pad_concat(x, x4)
        x_out2 = self.double_conv_up1(x)
        x = self.deconv2(x_out2)
        x = self.pad_concat(x, x3)
        x_out3 = self.double_conv_up2(x)
        x = self.deconv3(x_out3)
        x = self.pad_concat(x, x2)
        x_out4 = self.double_conv_up3(x)
        x = self.deconv4(x_out4)
        x = self.pad_concat(x, x1)
        x = self.double_conv_up4(x)

        return x, x_out1, x_out2, x_out3, x_out4

class Floor_2d_Unet_encoder_only(nn.Module):
    def __init__(self, n_channels, bilinear=True, feature_multiplication=4):
        super(Floor_2d_Unet_encoder_only, self).__init__()
        self.bilinear = bilinear
        print(f'bilinear: {bilinear}')
        self.feature_multiplication = feature_multiplication
        #Encoder
        self.double_conv_down1 = Double_conv_floor(n_channels, int(16*self.feature_multiplication))
        self.double_conv_down2 = Double_conv_floor(int(16*self.feature_multiplication), int(32*self.feature_multiplication))
        self.double_conv_down3 = Double_conv_floor(int(32*self.feature_multiplication), int(64*self.feature_multiplication))
        self.double_conv_down4 = Double_conv_floor(int(64*self.feature_multiplication), int(128*self.feature_multiplication))
        factor = 2 if bilinear else 1
        self.double_conv_down5 = Double_conv_floor(int(128*self.feature_multiplication), int(256//factor*self.feature_multiplication))
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.double_conv_down1(x)
        x2 = self.maxpool(x1)
        x2 = self.double_conv_down2(x2)
        x3 = self.maxpool(x2)
        x3 = self.double_conv_down3(x3)
        x4 = self.maxpool(x3)
        x4 = self.double_conv_down4(x4)
        x = self.maxpool(x4)
        x = self.double_conv_down5(x)
        return x

class Floor_2d_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, sigmoid_finish=True, feature_multiplication=4):
        super(Floor_2d_Unet, self).__init__()
        # self.segmentation_net = Floor_2d_Unet_without_final_conv(n_channels, n_classes, bilinear=bilinear, feature_multiplication=feature_multiplication)
        self.bilinear = bilinear
        print(f'bilinear: {bilinear}')
        self.sigmoid_finish = sigmoid_finish
        self.feature_multiplication = feature_multiplication
        #Encoder
        self.double_conv_down1 = Double_conv_floor(n_channels, int(16*self.feature_multiplication))
        self.double_conv_down2 = Double_conv_floor(int(16*self.feature_multiplication), int(32*self.feature_multiplication))
        self.double_conv_down3 = Double_conv_floor(int(32*self.feature_multiplication), int(64*self.feature_multiplication))
        self.double_conv_down4 = Double_conv_floor(int(64*self.feature_multiplication), int(128*self.feature_multiplication))
        factor = 2 if bilinear else 1
        self.double_conv_down5 = Double_conv_floor(int(128*self.feature_multiplication), int(256//factor*self.feature_multiplication))
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        #Decoder
        self.pad_concat = Concat_and_pad()
        if bilinear:
            self.deconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.deconv1 = nn.ConvTranspose2d(int(256*self.feature_multiplication), int(128*self.feature_multiplication), kernel_size=2, stride=2)
            self.deconv2 =  nn.ConvTranspose2d(int(128*self.feature_multiplication), int(64*self.feature_multiplication), kernel_size=2, stride=2)
            self.deconv3 =  nn.ConvTranspose2d(int(64*self.feature_multiplication), int(32*self.feature_multiplication), kernel_size=2, stride=2)
            self.deconv4 =  nn.ConvTranspose2d(int(32*self.feature_multiplication), int(16*self.feature_multiplication), kernel_size=2, stride=2)
        self.double_conv_up1 = Double_conv_floor(int(256*self.feature_multiplication), int(128//factor*self.feature_multiplication))
        self.double_conv_up2 = Double_conv_floor(int(128*self.feature_multiplication), int(64//factor*self.feature_multiplication))
        self.double_conv_up3 = Double_conv_floor(int(64*self.feature_multiplication), int(32//factor*self.feature_multiplication))
        self.double_conv_up4 = Double_conv_floor(int(32*self.feature_multiplication), int(16*self.feature_multiplication))

        self.final_conv = nn.Conv2d(int(16*self.feature_multiplication), n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.segmentation_net(x)
        x1 = self.double_conv_down1(x)
        x2 = self.maxpool(x1)
        x2 = self.double_conv_down2(x2)
        x3 = self.maxpool(x2)
        x3 = self.double_conv_down3(x3)
        x4 = self.maxpool(x3)
        x4 = self.double_conv_down4(x4)
        x = self.maxpool(x4)
        x = self.double_conv_down5(x)

        x = self.deconv1(x)
        x = self.pad_concat(x, x4)
        x = self.double_conv_up1(x)
        x = self.deconv2(x)
        x = self.pad_concat(x, x3)
        x = self.double_conv_up2(x)
        x = self.deconv3(x)
        x = self.pad_concat(x, x2)
        x = self.double_conv_up3(x)
        x = self.deconv4(x)
        x = self.pad_concat(x, x1)
        x = self.double_conv_up4(x)

        logits = self.final_conv(x)
        if self.sigmoid_finish:
            output = self.sigmoid(logits)
        else:
            output = logits
        return output

class Floor_2d_UnetAllOutputs(Floor_2d_Unet):
    def __init__(self, n_channels, n_classes, bilinear=True, sigmoid_finish=True, feature_multiplication=4):
        super().__init__(n_channels, n_classes, bilinear, sigmoid_finish, feature_multiplication)

    def forward(self, x):
        x1 = self.double_conv_down1(x)
        x2 = self.maxpool(x1)
        x2 = self.double_conv_down2(x2)
        x3 = self.maxpool(x2)
        x3 = self.double_conv_down3(x3)
        x4 = self.maxpool(x3)
        x4 = self.double_conv_down4(x4)
        x = self.maxpool(x4)
        out_x1 = self.double_conv_down5(x)

        x = self.deconv1(out_x1)
        x = self.pad_concat(x, x4)
        out_x2 = self.double_conv_up1(x)
        x = self.deconv2(out_x2)
        x = self.pad_concat(x, x3)
        out_x3 = self.double_conv_up2(x)
        x = self.deconv3(out_x3)
        x = self.pad_concat(x, x2)
        out_x4 = self.double_conv_up3(x)
        x = self.deconv4(out_x4)
        x = self.pad_concat(x, x1)
        out_x5 = self.double_conv_up4(x)

        return out_x1, out_x2, out_x3, out_x4, out_x5

class Floor_3D_half_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, sigmoid_finish=True, feature_multiplication=4):
        super(Floor_3D_half_Unet, self).__init__()
        self.bilinear = bilinear
        self.sigmoid_finish = sigmoid_finish
        print(f'bilinear: {bilinear}')
        self.feature_multiplication = feature_multiplication
        #Encoder
        self.double_conv_down1 = Double_conv_floor_3D(n_channels, 16*self.feature_multiplication, conv_type='2D')
        self.double_conv_down2 = Double_conv_floor_3D(16*self.feature_multiplication, 32*self.feature_multiplication, conv_type='2D')
        self.double_conv_down3 = Double_conv_floor_3D(32*self.feature_multiplication, 64*self.feature_multiplication, conv_type='2D')
        self.double_conv_down4 = Double_conv_floor_3D(64*self.feature_multiplication, 128*self.feature_multiplication, conv_type='2D')
        factor = 2 if bilinear else 1
        self.double_conv_down5 = Double_conv_floor_3D(128*self.feature_multiplication, 256*self.feature_multiplication//factor, conv_type='3D')
        self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2))

        #Decoder
        self.pad_concat = Concat_and_pad(mode='3D')
        if bilinear:
            self.deconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.deconv1 = nn.ConvTranspose3d(256*self.feature_multiplication, 128*self.feature_multiplication, kernel_size=(1,2,2), stride=(1,2,2))
            self.deconv2 =  nn.ConvTranspose3d(128*self.feature_multiplication, 64*self.feature_multiplication, kernel_size=(1,2,2), stride=(1,2,2))
            self.deconv3 =  nn.ConvTranspose3d(64*self.feature_multiplication, 32*self.feature_multiplication, kernel_size=(1,2,2), stride=(1,2,2))
            self.deconv4 =  nn.ConvTranspose3d(32*self.feature_multiplication, 16*self.feature_multiplication, kernel_size=(1,2,2), stride=(1,2,2))
        self.double_conv_up1 = Double_conv_floor_3D(256*self.feature_multiplication, 128*self.feature_multiplication//factor, conv_type='3D')
        self.double_conv_up2 = Double_conv_floor_3D(128*self.feature_multiplication, 64*self.feature_multiplication//factor, conv_type='2D')
        self.double_conv_up3 = Double_conv_floor_3D(64*self.feature_multiplication, 32*self.feature_multiplication//factor, conv_type='2D')
        self.double_conv_up4 = Double_conv_floor_3D(32*self.feature_multiplication, 16*self.feature_multiplication, conv_type='2D')

        self.final_conv = nn.Conv3d(16*self.feature_multiplication, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.double_conv_down1(x)
        x2 = self.maxpool(x1)
        x2 = self.double_conv_down2(x2)
        x3 = self.maxpool(x2)
        x3 = self.double_conv_down3(x3)
        x4 = self.maxpool(x3)
        x4 = self.double_conv_down4(x4)
        x = self.maxpool(x4)
        x = self.double_conv_down5(x)

        x = self.deconv1(x)
        x = self.pad_concat(x, x4)
        x = self.double_conv_up1(x)
        x = self.deconv2(x)
        x = self.pad_concat(x, x3)
        x = self.double_conv_up2(x)
        x = self.deconv3(x)
        x = self.pad_concat(x, x2)
        x = self.double_conv_up3(x)
        x = self.deconv4(x)
        x = self.pad_concat(x, x1)
        x = self.double_conv_up4(x)

        logits = self.final_conv(x)
        if self.sigmoid_finish:
            output = self.sigmoid(logits)
        else:
            output = logits
        return output
    
class Floor_3D_full_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, sigmoid_finish=True, feature_multiplication=4):
        super(Floor_3D_full_Unet, self).__init__()
        self.bilinear = bilinear
        self.sigmoid_finish = sigmoid_finish
        print(f'bilinear: {bilinear}')
        self.feature_multiplication = feature_multiplication
        #Encoder
        self.double_conv_down1 = Double_conv_floor_3D(n_channels, 16*self.feature_multiplication, conv_type='3D')
        self.double_conv_down2 = Double_conv_floor_3D(16*self.feature_multiplication, 32*self.feature_multiplication, conv_type='3D')
        self.double_conv_down3 = Double_conv_floor_3D(32*self.feature_multiplication, 64*self.feature_multiplication, conv_type='3D')
        self.double_conv_down4 = Double_conv_floor_3D(64*self.feature_multiplication, 128*self.feature_multiplication, conv_type='3D')
        factor = 2 if bilinear else 1
        self.double_conv_down5 = Double_conv_floor_3D(128*self.feature_multiplication, 256*self.feature_multiplication//factor, conv_type='3D')
        self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2))

        #Decoder
        self.pad_concat = Concat_and_pad(mode='3D')
        if bilinear:
            self.deconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.deconv1 = nn.ConvTranspose3d(256*self.feature_multiplication, 128*self.feature_multiplication, kernel_size=(1,2,2), stride=(1,2,2))
            self.deconv2 =  nn.ConvTranspose3d(128*self.feature_multiplication, 64*self.feature_multiplication, kernel_size=(1,2,2), stride=(1,2,2))
            self.deconv3 =  nn.ConvTranspose3d(64*self.feature_multiplication, 32*self.feature_multiplication, kernel_size=(1,2,2), stride=(1,2,2))
            self.deconv4 =  nn.ConvTranspose3d(32*self.feature_multiplication, 16*self.feature_multiplication, kernel_size=(1,2,2), stride=(1,2,2))
        self.double_conv_up1 = Double_conv_floor_3D(256*self.feature_multiplication, 128*self.feature_multiplication//factor, conv_type='3D')
        self.double_conv_up2 = Double_conv_floor_3D(128*self.feature_multiplication, 64*self.feature_multiplication//factor, conv_type='3D')
        self.double_conv_up3 = Double_conv_floor_3D(64*self.feature_multiplication, 32*self.feature_multiplication//factor, conv_type='3D')
        self.double_conv_up4 = Double_conv_floor_3D(32*self.feature_multiplication, 16*self.feature_multiplication, conv_type='3D')

        self.final_conv = nn.Conv3d(16*self.feature_multiplication, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.double_conv_down1(x)
        x2 = self.maxpool(x1)
        x2 = self.double_conv_down2(x2)
        x3 = self.maxpool(x2)
        x3 = self.double_conv_down3(x3)
        x4 = self.maxpool(x3)
        x4 = self.double_conv_down4(x4)
        x = self.maxpool(x4)
        x = self.double_conv_down5(x)

        x = self.deconv1(x)
        x = self.pad_concat(x, x4)
        x = self.double_conv_up1(x)
        x = self.deconv2(x)
        x = self.pad_concat(x, x3)
        x = self.double_conv_up2(x)
        x = self.deconv3(x)
        x = self.pad_concat(x, x2)
        x = self.double_conv_up3(x)
        x = self.deconv4(x)
        x = self.pad_concat(x, x1)
        x = self.double_conv_up4(x)

        logits = self.final_conv(x)
        if self.sigmoid_finish:
            output = self.sigmoid(logits)
        else:
            output = logits
        return output

class CA_2d_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(CA_2d_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)      #maxpool 2x2, doubleConv(64-128-128)
        self.down2 = Down(128, 256)     #maxpool 2x2, doubleConv(128-256-256)
        self.down3 = Down(256, 512)     #maxpool 2x2, doubleConv(256-512-512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  #bileair: maxpool 2x2, doubleConv(512-512-512)    #non-bilineair: maxpool 2x2, doubleConv(512-1024-1024)
        self.concat1 = DeconvConcat(1024, 512, bilinear)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        self.sa1 = SA1(1024, 512, bilinear)
        self.ca1 = CA_block(512, 512, bilinear)
        self.sa2 = SA234(512, 256, bilinear)
        self.ca2 = CA_block(256, 256, bilinear)
        self.sa3 = SA234(256, 128, bilinear)
        self.ca3 = CA_block(128, 128, bilinear)
        self.sa4 = SA234(128, 64, bilinear)
        self.ca4 = CA_block(64, 64, bilinear)

        self.la = LA_block(512, 4, 1)
        self.outc = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.concat1(x5, x4)
        x, sa1_map = self.sa1(x)
        x6, ca1_map = self.ca1(x)
        x, sa2_map = self.sa2(x6, x3)
        x7, ca2_map = self.ca2(x)
        x, sa3_map = self.sa3(x7, x2)
        x8, ca3_map = self.ca3(x)
        x, sa4_map = self.sa4(x8, x1)
        x9, ca4_map = self.ca4(x)
        
        x, la1_map, la2_map = self.la(x6, x7, x8, x9)
        logits = self.outc(x)
        logits = nn.Sigmoid()(logits)
        return logits, [[sa1_map, sa2_map, sa3_map, sa4_map], [ca1_map, ca2_map, ca3_map, ca4_map], [la1_map, la2_map]]
