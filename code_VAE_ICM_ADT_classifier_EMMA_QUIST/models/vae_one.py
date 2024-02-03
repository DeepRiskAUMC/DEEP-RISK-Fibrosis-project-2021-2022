import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.distributions import Normal

class FC_Classifier(nn.Module):
    """Latent space discriminator"""
    def __init__(self, latent_size, n_out=1):
        super(FC_Classifier, self).__init__()
        self.latent_size = latent_size
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(latent_size, latent_size//2, bias=True),
            # nn.BatchNorm1d(latent_size//2),
            nn.ReLU(),
            nn.Linear(latent_size//2, latent_size//2, bias=True),
            # nn.BatchNorm1d(latent_size//2),
            nn.ReLU(),
            nn.Linear(latent_size//2, latent_size//4, bias=True),
            # nn.BatchNorm1d(latent_size//4),
            nn.ReLU(),
            nn.Linear(latent_size//4, latent_size//4, bias=True),
            # nn.BatchNorm1d(latent_size//4),
            nn.ReLU(),
            nn.Linear(latent_size//4, latent_size//4, bias=True),
            # nn.BatchNorm1d(latent_size//4),
            nn.ReLU(),
            nn.Linear(latent_size//4, n_out, bias=True),
        )

    def forward(self, x):
        return self.net(x)
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, C, H, W, D):
        super(UnFlatten, self).__init__()
        self.C, self.H, self.W, self.D = C, H, W, D

    def forward(self, input):
        return input.view(input.size(0), self.C, self.H, self.W, self.D)
    
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout_rate=0.1, bias=False):
        super(Conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),#changed to relu from leaky
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        )

    def forward(self, x):
        return self.conv(x)
    
class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dropout_rate=0.1):
        super(ConvTranspose, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.ReLU(),#changed to relu from leaky
            nn.Dropout(dropout_rate)#added dropout here
        )

    def forward(self, x):
        return self.conv(x)

class ConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvUpsample, self).__init__()
        
        self.scale_factor = kernel_size
        self.upconv = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        )

    def forward(self, x):
        return self.upconv(x)
    
class ResBlock_Dec_Transpose(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.hidden_channel = out_channel

        self.conv1 = nn.Conv3d(in_channel, self.hidden_channel, kernel_size=3,  padding=1)
        self.conv2 = nn.Conv3d(self.hidden_channel, out_channel, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm3d(in_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        
        self.act = nn.ReLU()
        if upsample or (in_channel != out_channel):
            self.conv_up = nn.ConvTranspose3d(in_channel, out_channel,
                                    kernel_size=3, stride=2, padding=1, output_padding=1)
        else: 
            self.conv_up = None

    def res_up(self, x):
        if self.conv_up:
            x = self.conv_up(x)
        return x

    def forward(self, x):
        
        h = self.act(self.bn1(x))
        if self.conv_up:
            h = self.conv_up(h)
        else:
            h = self.conv1(h)
        h = self.act(self.bn2(h))
        h = self.conv2(h)
        return h + self.res_up(x)
    
class ResBlock_Dec_Up(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.hidden_channel = out_channel
        self.conv1 = nn.Conv3d(in_channel, self.hidden_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(self.hidden_channel, out_channel, kernel_size=3, padding=1, bias=False)

        self.bn1 = nn.BatchNorm3d(in_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        
        self.act = nn.ReLU()
        if upsample or (in_channel != out_channel):
            self.upsample = nn.Upsample(scale_factor=2)
            self.conv_up = nn.Conv3d(in_channel, out_channel,
                                    kernel_size=1, padding=0)
        else: 
            self.upsample = None


    def res_up(self, x):
        if self.upsample:
            x = self.upsample(x)
            x = self.conv_up(x)
        return x

    def forward(self, x):
        h = self.act(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
        h = self.conv1(h)
        h = self.act(self.bn2(h))
        h = self.conv2(h)
        # print(h.shape)
        return h + self.res_up(x)

class Resblock_Enc(nn.Module):
    def __init__(self, in_channel, out_channel, downsample, preactivation):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.hidden_channel = out_channel
        self.conv1 = nn.Conv3d(in_channel, self.hidden_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(self.hidden_channel, out_channel, kernel_size=3, padding=1, bias=False)

        self.preactivation = preactivation
        self.act = nn.ReLU()

        self.bn1 = nn.BatchNorm3d(in_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)

        if downsample or (in_channel != out_channel):
            self.downsample = nn.AvgPool3d(2)
            self.conv_down = nn.Conv3d(in_channel, out_channel,
                                    kernel_size=1, padding=0)
        else: 
            self.downsample = None


    def res_down(self, x):
        if self.preactivation:
            if self.downsample:
                x = self.conv_down(x)
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
                x = self.conv_down(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(self.bn1(x))
        else:
            h = x
        h = self.conv1(h)
        h  = self.conv2(self.act(self.bn2(h)))
        if self.downsample:
            h = self.downsample(h)
        # print(h.shape)
        return h + self.res_down(x)    
    
    
class VAE_One(nn.Module):
    def __init__(self, args, img_size):
        super(VAE_One, self).__init__()
        self.image_channels  = args.IMG
        self.img_size = img_size
        self.latent_size = args.latent_size
        try:
            self.upsample = args.upsample
        except:
            self.upsample = args.up_sample
        # if args.dropout:
        #     self.dropout = 0.1
        # else:
        #     self.dropout = 0.0
        try:
            self.final_dropout = args.final_dropout
        except:
            self.final_dropout = 0.25
        self.bottleneck_channel = args.bottleneck_channel

        self.n_filters = args.n_filters
        self.mlp = False
        if args.model_type == 'vae_mlp':
            self.mlp = True
        self.log_sigma = 0
        if args.version == 'sigma':
            ## Sigma VAE
           self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=True)

        self._create_network()
        self._init_params()

    def _create_network(self):

        self.input_encoder = nn.Sequential(
            nn.Conv3d(self.image_channels, self.n_filters, kernel_size=3, padding=1, bias=True),
        )

        self.encoder = nn.Sequential(
            Resblock_Enc(self.n_filters, 2*self.n_filters, downsample=True, preactivation=True),
            Resblock_Enc(2 * self.n_filters, 4 * self.n_filters, downsample=True, preactivation=True),
            Resblock_Enc(4 * self.n_filters, 8 * self.n_filters, downsample=True, preactivation=True),
            Resblock_Enc(8 * self.n_filters, self.bottleneck_channel, downsample=True, preactivation=True),
            nn.BatchNorm3d(self.bottleneck_channel),
            nn.ReLU(),
            Flatten(), 
            nn.Dropout(self.final_dropout)
        )

        h_dim = int(self.bottleneck_channel * self.img_size/16 * self.img_size/16 * self.img_size/16)
        self.h_dim = h_dim

        self.extra_fc_layers = nn.Sequential(
            nn.Linear(h_dim, h_dim//2),
            # nn.BatchNorm1d(h_dim//2),
            nn.ReLU(),
            nn.Linear(h_dim//2, (self.latent_size + self.latent_size//2)),
            # nn.BatchNorm1d((self.latent_size + self.latent_size//2)),
            nn.ReLU()
        )
        self.encoder_mu = nn.Linear((self.latent_size + self.latent_size//2), self.latent_size)  # Corrected input size
        self.encoder_logvar = nn.Linear((self.latent_size + self.latent_size//2), self.latent_size)  # Corrected input size
        # self.encoder_mu = nn.Linear(h_dim, self.latent_size)  # Corrected input size
        # self.encoder_logvar = nn.Linear(h_dim, self.latent_size)  # Corrected input size
        if self.mlp:    
            self.prediction_net = FC_Classifier(self.latent_size)



        # else:
        #     self.encoder_mu = nn.Linear(h_dim, self.latent_size)  # Corrected input size
        #     self.encoder_logvar = nn.Linear(h_dim, self.latent_size)  # Corrected input size

        self.input_decoder = nn.Sequential(
            nn.Linear(self.latent_size, h_dim),  # Adjust the output size
            # nn.ReLU()
        )
        if self.upsample:
            self.decoder = nn.Sequential(
                    UnFlatten(C=self.bottleneck_channel, H=int(self.img_size/16), W=int(self.img_size/16), D=int(self.img_size/16)),
                    ResBlock_Dec_Up(self.bottleneck_channel, 8 * self.n_filters, upsample=True),
                    ResBlock_Dec_Up(8 * self.n_filters, 4 * self.n_filters, upsample=True),
                    ResBlock_Dec_Up(4 * self.n_filters, 2 * self.n_filters,  upsample=True),
                    ResBlock_Dec_Up(2 * self.n_filters, self.n_filters, upsample=True),
                    nn.BatchNorm3d(self.n_filters),
                    nn.ReLU()
                )
        else:
            self.decoder = nn.Sequential(
                    UnFlatten(C=self.bottleneck_channel, H=int(self.img_size/16), W=int(self.img_size/16), D=int(self.img_size/16)),
                    ResBlock_Dec_Transpose(self.bottleneck_channel, 8 * self.n_filters, upsample=True),
                    ResBlock_Dec_Transpose(8 * self.n_filters, 4 * self.n_filters, upsample=True),
                    ResBlock_Dec_Transpose(4 * self.n_filters, 2 * self.n_filters,  upsample=True),
                    ResBlock_Dec_Transpose(2 * self.n_filters, self.n_filters, upsample=True),
                    nn.BatchNorm3d(self.n_filters),
                    nn.ReLU()
            )

        self.out_layer = nn.Sequential(
            nn.Conv3d(self.n_filters, self.image_channels, kernel_size=3, stride=1, padding=1, bias=True), 
            nn.Sigmoid()  # Change activation function to Sigmoid
        )
        
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
                nn.init.xavier_uniform_(m.weight)

                

    def classifier(self, z):
        z = self.prediction_net(z)
        return z
    
    def encode(self, x):
        x = self.input_encoder(x)
        x = self.encoder(x)
        x = self.extra_fc_layers(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)

        z_distribution = Normal(loc=mu, scale=torch.exp(logvar))
        return mu, logvar, z_distribution

    def reparameterize(self, mu, logvar, z_dist=False):
        if not z_dist:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            return z, 0, 0
        else:
            # reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_sampled_eq = eps.mul(std).add_(mu) # sample

            # compute prior : normal distribution
            prior_dist = Normal(loc=torch.zeros_like(z_dist.loc),scale=torch.ones_like(z_dist.scale))

            ### sample from the defined (in encoder) distribution
            z_tilde = z_dist.rsample() # implemented reparameterization trick
            return z_tilde, z_sampled_eq, prior_dist

    def decode(self, z):
        z = self.input_decoder(z)
        z = self.decoder(z)
        return self.out_layer(z)

    def forward(self, x):
        mu, logvar, z_dist = self.encode(x)
        
        z_tilde, z_sampled_eq, prior_dist = self.reparameterize(mu, logvar, z_dist)
        if self.train:
            z_tilde = mu

        if self.mlp:
            pred = self.classifier(z_tilde)
            return self.decode(z_tilde), mu, logvar, pred, z_sampled_eq, prior_dist, z_tilde, z_dist
        else:
            return self.decode(z_tilde), mu, logvar, z_sampled_eq, prior_dist, z_tilde, z_dist
        
