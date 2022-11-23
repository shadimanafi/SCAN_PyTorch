"""implementing models"""

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import collections


#import model zoo in torchvision
import torchvision.transforms as transforms


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class base_model(nn.Module):
    def __init__(self, z_dim, nc):
        super(base_model, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
    def _encode(self, x):
        return self.encoder(x)

class AutoEncoder(base_model):
    def __init__(self, z_dim, nc):
        super(AutoEncoder, self).__init__(z_dim, nc)
    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _decode(self, z):
        if z.shape[1] == self.z_dim:
            return self.decoder(z)
        else:
            mu = z[:, :self.z_dim]
            logvar = z[:, self.z_dim:]
            z = reparametrize(mu, logvar)
            return self.decoder(z)


class BetaVAE_H_net(AutoEncoder):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=32, nc=3):
        super(BetaVAE_H_net, self).__init__(z_dim, nc)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),             # B, 32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),             # B, 32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),             # B, 64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),             # B, 64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),               # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                    # B, 256
            nn.Linear(256, z_dim*2),                # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),                  # B, 256
            View((-1, 256, 1, 1)),                  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),         # B, 64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),    # B, 64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # B, 32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),    # B, 32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),    # B, nc, 64, 64
            nn.Sigmoid()
        )

        self.weight_init()

class BetaVAE_B_net(AutoEncoder):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=32, nc=1):
        super(BetaVAE_B_net, self).__init__(z_dim, nc)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),             # B, 32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),             # B, 32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),             # B, 32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),             # B, 32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                     # B, 512
            nn.Linear(32*4*4, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                    # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),                # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),                  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                    # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),                 # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                   # B, 32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1),    # B, 32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),    # B, 32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),    # B, 32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),    # B, nc, 64, 64
            nn.Sigmoid()
        )
        self.weight_init()

class DAE_net(base_model):
    def __init__(self, z_dim=100, nc=3):
        super(DAE_net, self).__init__(z_dim, nc)
        # self.encoder = nn.Sequential(
        #     collections.OrderedDict(
        #         [
        #             ("conv1",nn.Conv2d(nc, 32, 4, 2, 1)),             # B, 32, 32, 32
        #             ("relu1",nn.ReLU(True)),
        #             ("conv2",nn.Conv2d(32, 32, 4, 2, 1)),             # B, 32, 16, 16
        #             ("relu2",nn.ReLU(True)),
        #             ("conv3",nn.Conv2d(32, 64, 4, 2, 1)),             # B, 64,  8,  8
        #             ("relu3",nn.ReLU(True)),
        #             ("conv4",nn.Conv2d(64, 64, 4, 2, 1)),             # B, 64,  4,  4
        #             ("relu4",nn.ReLU(True)),
        #             ("relu4-reverse",View((-1, 1024))),                       # B, 1024
        #             ("lin1",nn.Linear(1024, z_dim)),                 # B, z_dim
        #         ]
        #     )
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),             # B, 32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),             # B, 32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),             # B, 64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),             # B, 64,  4,  4
            nn.ReLU(True),
            View((-1, 1024)),                       # B, 1024
            nn.Linear(1024, z_dim),                 # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024),                 # B, 1024
            View((-1, 64, 4, 4)),                   # B, 64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),    # B, 64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # B, 32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),    # B, 32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),    # B, nc, 64, 64
            nn.Sigmoid()
        )

        self.weight_init()

    def forward(self, x):
        x_encoded = self._encode(x)
        x_recon = self._decode(x_encoded)
        return x_recon
    def _decode(self, z):
        return self.decoder(z)

class SCAN_net(AutoEncoder):
    """Model proposed in SCAN: Learning Hierarchical Compositional Visual Concepts, Higgins et al., ICLR 2018."""

    def __init__(self, z_dim=32, nc=40):
        super(SCAN_net, self).__init__(z_dim, nc)
        self.encoder = nn.Sequential(
            nn.Linear(nc, 500),                     # B, 500
            nn.ReLU(True),
            nn.Linear(500, 500),                    # B, 500
            nn.ReLU(True),
            nn.Linear(500, self.z_dim * 2),         # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 500),                  # B, 500
            nn.ReLU(True),
            nn.Linear(500, 500),                    # B, 500
            nn.ReLU(True),
            nn.Linear(500, nc),                     # B, nc
            nn.Sigmoid(),
        )
        self.weight_init()
