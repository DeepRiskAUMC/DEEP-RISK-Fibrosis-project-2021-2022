import torch
import torch.nn as nn
import torch.nn.functional as F

class DeconvConcat(nn.Module):
    """Upscaling and concatenating"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x

class SA1(nn.Module):
    """Spatial Attention Block 1 + Conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv1_3 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.row_softmax = nn.Softmax(dim=1)
        self.conv2 = nn.Conv2d(64, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        #SA module
        batch_size, _, h, w = x.shape
        x1 = self.conv1_1(x).view(batch_size, 64,-1)
        x2 = self.conv1_2(x).view(batch_size, 64,-1)
        x3 = self.conv1_3(x).view(batch_size, 64,-1)
        x1 = torch.transpose(x1,1,2)
        x_dot = torch.matmul(x1, x2)
        alpha = self.row_softmax(x_dot)
        x3 = torch.transpose(x3,1,2)
        x_hat = torch.matmul(alpha, x3)
        x_hat = x_hat.view(batch_size, 64, h, w)
        x_hat = self.conv2(x_hat)
        x_hat = self.bn(x_hat)
        y_SA1 = x_hat + x

        #convolution
        output = self.conv3(y_SA1)
        return output, alpha

class SA234(nn.Module):
    """Deconvolution + Spatial Attention Block 2, 3 and 4 + Concatanation + Conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.deconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels // 2, 64, kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels//2, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1)
        self.conv3_1 = nn.Conv2d(in_channels // 2, 64, kernel_size=1)
        self.conv3_2 = nn.Conv2d(in_channels//2, 64, kernel_size=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, xh, xl):
        #deconvolution
        xh = self.deconv(xh)

        diffY = xl.size()[2] - xh.size()[2]
        diffX = xl.size()[3] - xh.size()[3]

        xh = F.pad(xh, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        #SA-block one-way
        xh_2 = self.conv1_1(xh)
        xl_2 = self.conv1_2(xl)
        xl_2 = xh_2 + xl_2
        xl_2 = self.relu(xl_2)
        xl_2 = self.conv2(xl_2)
        alpha_1 = self.sigmoid(xl_2)

        #SA-block other way
        xh_2 = self.conv3_1(xh)
        xl_2 = self.conv3_2(xl)
        xl_2 = xh_2 + xl_2
        xl_2 = self.relu(xl_2)
        xl_2 = self.conv4(xl_2)
        alpha_2 = self.sigmoid(xl_2)

        xl_1 = torch.mul(xl, alpha_1)
        xl_2 = torch.mul(xl, alpha_2)
        y_SA = torch.cat([xl_1, xl_2], dim=1)
        y_SA = self.conv5(y_SA)
        y_SA = self.bn(y_SA)
        y_SA = self.relu(y_SA)

        #concatenation + Convolution
        output = torch.cat([y_SA, xh], dim=1)
        output = self.conv6(output)
        return output, [alpha_1, alpha_2]

class CA_block(nn.Module):
    """Channel Attention Block + Conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.mlp = nn.Sequential(
                        nn.Linear(in_channels, int(in_channels/2)),
                        nn.ReLU(),
                        nn.Linear(int(in_channels/2), in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        #CA attention block
        batch_size, channels, h, w = x.shape
        x1 = x.view(batch_size, channels, -1).mean(2)
        x2 = x.view(batch_size, channels, -1).max(2)[0]
        x1 = self.mlp(x1)
        x2 = self.mlp(x2)
        beta = self.sigmoid(x1+x2).view(batch_size,-1,1,1)
        y_CA = torch.mul(x, beta)
        y_CA = y_CA + x

        #Conv layer
        output = self.conv(y_CA)
        return output, beta

class LA_block(nn.Module):
    """Bilinear interpolation + concatenation + Layer Attention Block"""

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.bilin_1 = F.interpolate
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels//2, 1, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels//4, 1, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels//8, 1, kernel_size=1)
        self.mlp = nn.Sequential(
                        nn.Linear(mid_channels, 2*mid_channels),
                        nn.ReLU(),
                        nn.Linear(2*mid_channels, mid_channels)
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.conv5 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1)


    def forward(self, x1, x2, x3, x4):
        #Bilinear interpolation + concatanation
        x1 = F.interpolate(x1, scale_factor=8, mode='bilinear')
        x1 = self.conv1(x1)
        x2 = F.interpolate(x2, scale_factor=4, mode='bilinear')
        x2 = self.conv2(x2)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        #Layer attention
        batch_size, channels, h, w = x.shape
        x1 = x.view(batch_size, channels, -1).mean(2)
        x2 = x.view(batch_size, channels, -1).max(2)[0]
        x1 = self.mlp(x1)
        x2 = self.mlp(x2)
        gamma = self.sigmoid(x1+x2).view(batch_size,-1,1,1)
        F_hat_1 = torch.mul(x, gamma)

        #Layer Attention *
        gamma_star = self.conv5(F_hat_1)
        gamma_star = self.relu(gamma_star)
        gamma_star = self.conv6(gamma_star)
        gamma_star = self.sigmoid(gamma_star)
        F_hat_2 = torch.mul(F_hat_1, gamma_star)

        output = F_hat_1 + F_hat_2 + x

        #convolution
        return output, gamma, gamma_star