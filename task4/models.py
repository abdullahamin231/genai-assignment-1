import torch
import torch.nn as nn


def down(i, o, norm=True):
    layers = [nn.Conv2d(i, o, 4, 2, 1, bias=False)]
    if norm:
        layers.append(nn.InstanceNorm2d(o, affine=True))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)


def up(i, o, dropout=0.0):
    layers = [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.InstanceNorm2d(o, affine=True)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


def _bcast(e, h, w):
    return e[:, :, None, None].expand(-1, -1, h, w)


class UNetGenerator(nn.Module):
    """U-Net, 128x128 RGB photo + style id -> 128x128 grayscale sketch in [-1, 1]."""

    def __init__(self, base=64, emb_dim=16, dropout=0.3, num_styles=3):
        super().__init__()
        c = base
        self.emb = nn.Embedding(num_styles, emb_dim)
        self.d1 = down(3 + emb_dim, c, norm=False)  # 64
        self.d2 = down(c, 2 * c)                    # 32
        self.d3 = down(2 * c, 4 * c)                # 16
        self.d4 = down(4 * c, 8 * c)                # 8
        self.d5 = down(8 * c, 8 * c)                # 4
        self.d6 = down(8 * c, 8 * c, norm=False)    # 2 (bottleneck)
        self.u1 = up(8 * c + emb_dim, 8 * c, dropout)  # 4
        self.u2 = up(16 * c, 8 * c, dropout)           # 8
        self.u3 = up(16 * c, 4 * c, dropout)           # 16
        self.u4 = up(8 * c, 2 * c)                     # 32
        self.u5 = up(4 * c, c)                         # 64
        self.out = nn.Sequential(nn.ConvTranspose2d(2 * c, 1, 4, 2, 1), nn.Tanh())  # 128

    def forward(self, x, style):
        e = self.emb(style)
        d1 = self.d1(torch.cat([x, _bcast(e, x.shape[2], x.shape[3])], 1))
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        b = torch.cat([d6, _bcast(e, d6.shape[2], d6.shape[3])], 1)
        u1 = torch.cat([self.u1(b), d5], 1)
        u2 = torch.cat([self.u2(u1), d4], 1)
        u3 = torch.cat([self.u3(u2), d3], 1)
        u4 = torch.cat([self.u4(u3), d2], 1)
        u5 = torch.cat([self.u5(u4), d1], 1)
        return self.out(u5)


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN on (photo, sketch, style embedding) -> patch logits."""

    def __init__(self, base=64, emb_dim=16, num_styles=3):
        super().__init__()
        c = base
        self.emb = nn.Embedding(num_styles, emb_dim)
        self.net = nn.Sequential(
            nn.Conv2d(3 + 1 + emb_dim, c, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, 2 * c, 4, 2, 1, bias=False), nn.InstanceNorm2d(2 * c, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1, bias=False), nn.InstanceNorm2d(4 * c, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * c, 8 * c, 4, 1, 1, bias=False), nn.InstanceNorm2d(8 * c, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(8 * c, 1, 4, 1, 1),
        )

    def forward(self, photo, sketch, style):
        e = _bcast(self.emb(style), photo.shape[2], photo.shape[3])
        return self.net(torch.cat([photo, sketch, e], 1))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 0.02)
