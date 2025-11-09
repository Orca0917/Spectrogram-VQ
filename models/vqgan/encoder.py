from torch import nn
from models.vqgan.modules import Swish, GroupNorm
from models.vqgan.residual import ResidualBlock


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.latent_dim = args.latent_dim
        channels = [128, 128, 128, 256, 256]
        num_res_blocks = 2
        layers = []

        layers.append(nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1))
        layers.append(nn.Conv2d(channels[0], channels[1], kernel_size=(4, 4), stride=(4, 2), padding=(0, 1)))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[1], channels[2], kernel_size=(5, 1), stride=(5, 1)))
        layers.append(Swish())

        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(channels[2], channels[2]))

        layers.append(ResidualBlock(channels[2], channels[3]))
        layers.append(ResidualBlock(channels[3], channels[3]))

        layers.append(GroupNorm(channels[3]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[3], self.latent_dim, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)