import torch.nn as nn


def conv_block(i, o, stride=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class ConvAE(nn.Module):
    """Plain convolutional autoencoder with NO skip connections.
    128x128x3 -> encoder (4 stride-2 stages) -> latent (latent_ch x 8 x 8) -> decoder -> 128x128x3 in [0,1].
    Dropout2d is applied on the latent and on the two lowest-resolution decoder stages."""

    def __init__(self, base_ch=64, latent_ch=32, dropout=0.1):
        super().__init__()
        c = base_ch
        chs = [c, 2 * c, 4 * c, 8 * c]
        enc, i = [], 3
        for o in chs:
            enc += [conv_block(i, o, 2), conv_block(o, o, 1)]
            i = o
        enc.append(nn.Conv2d(i, latent_ch, 1))
        self.encoder = nn.Sequential(*enc)
        self.latent_drop = nn.Dropout2d(dropout)

        self.dec_in = conv_block(latent_ch, 8 * c, 1) if False else nn.Sequential(
            nn.Conv2d(latent_ch, 8 * c, 1, bias=False), nn.BatchNorm2d(8 * c), nn.ReLU(True))
        stages, prev = [], 8 * c
        for k, o in enumerate([4 * c, 2 * c, c, c]):
            layers = [nn.Upsample(scale_factor=2, mode="nearest"), conv_block(prev, o, 1)]
            if k < 2:
                layers.append(nn.Dropout2d(dropout))
            stages.append(nn.Sequential(*layers))
            prev = o
        self.decoder = nn.Sequential(*stages)
        self.out = nn.Sequential(nn.Conv2d(c, 3, 3, 1, 1), nn.Sigmoid())
        self.latent_ch = latent_ch

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.latent_drop(self.encoder(x))
        return self.out(self.decoder(self.dec_in(z)))
