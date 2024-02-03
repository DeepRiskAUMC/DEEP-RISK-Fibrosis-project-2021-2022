import torch
import torch.nn as nn
import torch.utils.data
from sklearn import svm, metrics
"""
model based on: https://github.com/LukeDitria/CNN-VAE/blob/master/RES_VAE_Dynamic.py
"""

def initialize_weights(m):
  if isinstance(m, nn.Conv3d):
     nn.init.xavier_uniform_(m.weight)
     m.bias.data.fill_(0.01)
  elif isinstance(m, nn.BatchNorm3d):
     nn.init.constant_(m.weight.data, 1)
     nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm1d):
     nn.init.constant_(m.weight.data, 1)
     nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
     nn.init.xavier_uniform_(m.weight.data)
     nn.init.constant_(m.bias.data, 0)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, C, H, W, D):
        super(UnFlatten, self).__init__()
        self.C, self.H, self.W, self.D = C, H, W, D

    def forward(self, input):
        return input.view(input.size(0), self.C, self.H, self.W, self.D)
  
class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, downsample = False):
        super(ResDown, self).__init__()
        
        self.down_net = nn.Sequential(
            nn.BatchNorm3d(channel_in, eps=1e-4),
            nn.GELU(),
            nn.Conv3d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2),
            nn.BatchNorm3d(channel_out//2, eps=1e-4),
            nn.GELU(),
            nn.Conv3d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        )

        self.downsample = nn.Sequential(
            nn.BatchNorm3d(channel_in, eps=1e-4),
            nn.GELU(),
            nn.Conv3d(channel_in, channel_out, kernel_size, 2, kernel_size // 2),
        ) 

    def forward(self, x):
        z = self.down_net(x)
        residual = self.downsample(x)
        out = z + residual
        return out


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, upsample = False):
        super(ResUp, self).__init__()

        self.up_net = nn.Sequential(
            nn.Conv3d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2),

            nn.BatchNorm3d(channel_in // 2, eps=1e-4),
            nn.GELU(),
            nn.Conv3d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        )

        self.upsample = nn.Sequential(
            nn.BatchNorm3d(channel_in, eps=1e-4),
            nn.GELU(),
            nn.Upsample(scale_factor=scale_factor, mode="nearest")
        )  
        self.residual = nn.Conv3d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

    def forward(self, x):
        x = self.upsample(x)
        z = self.up_net(x)
        residual = self.residual(x)
        out = z + residual
        return out


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResBlock, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm3d(channel_in, eps=1e-4),
            nn.GELU(), 
            nn.Conv3d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2),

            nn.BatchNorm3d(channel_in // 2, eps=1e-4),
            nn.GELU(),
            nn.Conv3d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        )

        self.residual = nn.Sequential(
            nn.BatchNorm3d(channel_in, eps=1e-4),
            nn.GELU()
        )
        
    def forward(self, x):
        z = self.net(x)
        residual = self.residual(x)
        out = z + residual
        return out
    

class Latent_Classifier(nn.Module):
    """Latent space classifier seperate"""
    def __init__(self, args, n_out=2, sigmoid=False, conv=True):
        super(Latent_Classifier, self).__init__()
        mlp_do = args.dropout
        self.sigmoid = sigmoid
        self.mlp_c = None
        latent_size = args.latent_size
        if conv:
            self.flatten = Flatten()
            blocks = []            
            flat_mu_size = int((args.win_size[0]/(args.blocks[-1]*2))**3 * args.latent_size)

            n_filters = args.latent_size
            size = args.blocks[-1]*2

            while args.latent_size != flat_mu_size:
                blocks.append(n_filters)
                flat_mu_size = int((args.win_size[0]/(size*2))**3 * args.latent_size)
                size = size*2
            
            if len(blocks) != 0:
                widths_in = blocks
                widths_out = blocks[1:] + [blocks[-1]]

                layers_conv = []
                n = len(blocks)
                for i, (w_in, w_out) in enumerate(zip(widths_in, widths_out)):
                    if i == n-2:
                        layers_conv.append(nn.Conv3d(w_in, w_out, 2))
                    else:
                        layers_conv.append(nn.Conv3d(w_in, w_out, 3))
                    layers_conv.append(nn.GELU())

                layers_conv.append(nn.Conv3d(blocks[-1], args.latent_size, 1, 1))
                layers_conv.append(nn.GELU())
                self.mlp_c = nn.Sequential(*layers_conv)
            latent_size = flat_mu_size

        self.mlp_small = nn.Sequential(
                nn.Linear(latent_size, n_out),
        )

    def forward(self, x):
        if self.mlp_c:
            x = self.mlp_c(x)  
            x = self.flatten(x)
        y =self.mlp_small(x)
        if self.sigmoid:
            sigmoid = torch.nn.Sigmoid()
            y = sigmoid(y)
        return y
    
class Classifier(nn.Module):
    """Latent space classifier of VAE"""
    def __init__(self, args, n_out=2, sigmoid=False, conv=True):
        super(Classifier, self).__init__()
        mlp_do = args.dropout
        self.sigmoid = sigmoid
        self.mlp_c = None
    
        if args.dataset == 'EMIDEC':
            self.attri = True
        else:
            self.attri = False
        latent_size = args.latent_size
        if conv:
            self.flatten = Flatten()

            blocks = []
            
            flat_mu_size = int((args.win_size[0]/(args.blocks[-1]*2))**3 * args.latent_size)

            n_filters = args.latent_size
            size = args.blocks[-1]*2

            while args.latent_size != flat_mu_size:
                blocks.append(n_filters)
                flat_mu_size = int((args.win_size[0]/(size*2))**3 * args.latent_size)
                size = size*2
            
            if len(blocks) != 0:
                widths_in = blocks
                widths_out = blocks[1:] + [blocks[-1]]

                layers_conv = []
                n = len(blocks)
                for i, (w_in, w_out) in enumerate(zip(widths_in, widths_out)):
                    if i == n-2:
                        layers_conv.append(nn.Conv3d(w_in, w_out, 2))
                    else:
                        layers_conv.append(nn.Conv3d(w_in, w_out, 3))
                    layers_conv.append(nn.GELU())

                layers_conv.append(nn.Conv3d(blocks[-1], args.latent_size, 1, 1))
                layers_conv.append(nn.GELU())
                self.mlp_c = nn.Sequential(*layers_conv)

                
            latent_size = flat_mu_size

        self.mlp_l = nn.Sequential(
            nn.Linear(latent_size, latent_size//2),
            nn.GELU(),
            nn.Dropout(mlp_do),

            nn.Linear(latent_size//2, latent_size//2),
            nn.GELU(),
            nn.Dropout(mlp_do),

            nn.Linear(latent_size//2, latent_size//4),
            nn.GELU(),
            nn.Dropout(mlp_do),

            nn.Linear(latent_size//4, latent_size//8),
            nn.GELU(),
            nn.Dropout(mlp_do),

            nn.Linear(latent_size//8, n_out),
            )
        
        self.mlp_attri = nn.Sequential(
                nn.Linear(latent_size, int(latent_size/2)),
                nn.BatchNorm1d(int(latent_size/2)),
                nn.GELU(),
                nn.Linear(int(latent_size/2), int(latent_size/4)), 
                nn.BatchNorm1d(int(latent_size/4)),
                nn.GELU(),
                nn.Linear(int(latent_size/4), n_out), 
        )

    def forward(self, x):
        if self.mlp_c:
            x = self.mlp_c(x)  
            x = self.flatten(x)
        if self.attri:
            y =self.mlp_attri(x)
        else:
            y =self.mlp_l(x)
        if self.sigmoid:
            sigmoid = torch.nn.Sigmoid()
            y = sigmoid(y)
        return y


class Encoder(nn.Module):
    """
    Encoder block
    """

    def __init__(self, channels, ch=16, blocks=(1, 2, 4, 8), latent_size=64, sample=False):
        super(Encoder, self).__init__()
        self.sampling = sample
        self.in_enc = nn.Conv3d(channels, blocks[0] * ch, 3, 1, 1)
        bottleneck_channel = blocks[-1] * ch#16 * ch
        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [blocks[-1]]

        layer_blocks = []
        for w_in, w_out in zip(widths_in, widths_out):
            layer_blocks.append(ResDown(w_in * ch, w_out * ch))
        layer_blocks.append(ResBlock(bottleneck_channel, bottleneck_channel))
        layer_blocks.append(ResBlock(bottleneck_channel, bottleneck_channel))
        layer_blocks.append(nn.BatchNorm3d(bottleneck_channel, eps=1e-4))
        layer_blocks.append(nn.GELU())

        self.down_res = nn.Sequential(*layer_blocks)

        self.mu = nn.Conv3d(bottleneck_channel, latent_size, 1, 1)
        self.logvar = nn.Conv3d(bottleneck_channel, latent_size, 1, 1)

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample=False):
        x = self.in_enc(x)
        x = self.down_res(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        if self.training or self.sampling:
            x = self.sample(mu, logvar)
        else:
            x = mu
        return x, mu, logvar
        

class Decoder(nn.Module):
    """
    Decoder block
    """

    def __init__(self, channels_out, ch=16, blocks=(1, 2, 4, 8), bottleneck_channel=8, latent_size=64):
        super(Decoder, self).__init__()
        bottleneck_channel = blocks[-1] * ch 

        self.in_dec = nn.Conv3d(latent_size, bottleneck_channel, 1, 1)

        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [blocks[-1]])[::-1] #[16])[::-1]#
 
        layer_blocks = []
        layer_blocks.append(ResBlock(bottleneck_channel, bottleneck_channel))
        layer_blocks.append(ResBlock(bottleneck_channel, bottleneck_channel))
        for w_in, w_out in zip(widths_in, widths_out):
            layer_blocks.append(ResUp(w_in * ch, w_out * ch))
        layer_blocks.append(nn.BatchNorm3d(w_out * ch, eps=1e-4))
        layer_blocks.append(nn.GELU())

        self.up_res = nn.Sequential(*layer_blocks)

        self.out_dec = nn.Conv3d(blocks[0] * ch, channels_out, 3, 1, 1)
        self.out_af = nn.Sigmoid()


    def forward(self, x):
        x = self.in_dec(x)
        x = self.up_res(x)
        mu = self.out_af(self.out_dec(x))
        return mu


class VAE_Multi(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, args):
        super(VAE_Multi, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        
        self.encoder = Encoder(args.IMG, ch=args.n_filters, blocks=args.blocks, latent_size=args.latent_size, sample=False)
        self.decoder = Decoder(args.IMG, ch=args.n_filters, blocks=args.blocks, latent_size=args.latent_size)
        
        self.classifier = Classifier(args) if args.model_type == 'vae_mlp' else None
            
    def forward(self, x):

        output_dict = {}
        encoding, mu, log_var = self.encoder(x)
        recon_img = self.decoder(encoding)

        if self.classifier:
            pred = self.classifier(encoding)
            output_dict['pred'] = pred

        output_dict['img'] = recon_img
        output_dict['mu'] = mu
        output_dict['log_var'] = log_var
        return output_dict
    
