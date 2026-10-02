import torch
import torch.nn as nn


def conv_block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


def up_block(cin, cout):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), conv_block(cin, cout))


class UniversalAE(nn.Module):
    """Encoder: 128->64->32->16->8 spatial, channels c,2c,4c,8c -> 1x1 conv -> flatten -> latent vector.
    Decoder mirrors it. No skip connections."""

    def __init__(self, base_channels=32, latent_dim=256, dropout=0.1, bottleneck_ch=32):
        super().__init__()
        c = base_channels
        self.bottleneck_ch = bottleneck_ch
        self.encoder = nn.Sequential(
            conv_block(3, c, 2),          # 64
            conv_block(c, 2 * c, 2),      # 32
            conv_block(2 * c, 4 * c, 2),  # 16
            conv_block(4 * c, 8 * c, 2),  # 8
            nn.Conv2d(8 * c, bottleneck_ch, 1),
        )
        flat = bottleneck_ch * 8 * 8
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(flat, latent_dim), nn.Dropout(dropout))
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, flat), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_ch, 8 * c, 1), nn.BatchNorm2d(8 * c), nn.ReLU(inplace=True),
            up_block(8 * c, 4 * c),  # 16
            up_block(4 * c, 2 * c),  # 32
            up_block(2 * c, c),      # 64
            up_block(c, c),          # 128
            nn.Conv2d(c, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.to_latent(self.encoder(x))

    def decode(self, z):
        h = self.from_latent(z).view(-1, self.bottleneck_ch, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        return self.decode(self.encode(x))
