import torch.nn as nn
from models.vqgan.modules import Swish, GroupNorm
from models.vqgan.residual import ResidualBlock

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        latent_dim = args.latent_dim
        channels = [128, 128, 64, 32]
        num_res_blocks = 2

        layers = []

        layers.append(nn.Conv2d(latent_dim, channels[0], kernel_size=3, stride=1, padding=1))
        layers.append(Swish())

        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(channels[0], channels[0]))

        layers.append(
            nn.ConvTranspose2d(
                channels[0], channels[0],
                kernel_size=(5, 1), stride=(5, 1)
            )
        )
        layers.append(Swish())

        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(channels[1], channels[1]))

        layers.append(
            nn.ConvTranspose2d(
                channels[1], channels[2],
                kernel_size=(4, 4), stride=(4, 2), padding=(0, 1)
            )
        )
        layers.append(Swish())

        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(channels[2], channels[2]))

        layers.append(GroupNorm(channels[2]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[2], 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)