import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args, num_filters_last=64, n_layers=3):
        super().__init__()

        in_channel = 1
        layers = [
            nn.Conv2d(in_channel, num_filters_last, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2)
        
        ]
        num_filters_mult = 1
        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            stride = (1, 2) if i < n_layers else (1, 1)
            
            layers += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    kernel_size=(3, 4),
                    stride=stride,
                    padding=(1, 1),
                    bias=False
                ),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, kernel_size=(3, 3), stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)