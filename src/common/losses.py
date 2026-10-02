import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(ws, sigma, channels, device, dtype):
    coords = torch.arange(ws, device=device, dtype=dtype) - ws // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win2d = g[:, None] @ g[None, :]
    return win2d.expand(channels, 1, ws, ws).contiguous()


def ssim_per_image(x, y, ws=11, sigma=1.5, data_range=1.0):
    """x, y: (B,C,H,W) in [0, data_range]. Returns (B,) SSIM."""
    C = x.shape[1]
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    win = _gaussian_window(ws, sigma, C, x.device, x.dtype)
    pad = ws // 2
    mu_x = F.conv2d(x, win, padding=pad, groups=C)
    mu_y = F.conv2d(y, win, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    s_x = F.conv2d(x * x, win, padding=pad, groups=C) - mu_x2
    s_y = F.conv2d(y * y, win, padding=pad, groups=C) - mu_y2
    s_xy = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy
    smap = ((2 * mu_xy + c1) * (2 * s_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (s_x + s_y + c2))
    return smap.flatten(1).mean(1)


def psnr_per_image(x, y, data_range=1.0):
    mse = ((x - y) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(data_range ** 2 / mse)


class CombinedLoss(nn.Module):
    """L = alpha * L1 + (1 - alpha) * (1 - SSIM)   [the Task-1 specification formula]

    `mse_w` and `edge_w` are optional extra terms (both default to 0, i.e. the exact
    specification formula). They exist so that alternative objectives can be ablated and
    reported as "losses investigated" without touching the code; they are deliberately
    NOT part of the Optuna search space or the default training objective.
    """

    def __init__(self, alpha=0.8, mse_w=0.0, edge_w=0.0):
        super().__init__()
        self.alpha = alpha
        self.mse_w = mse_w
        self.edge_w = edge_w

    @staticmethod
    def _grad_mag(x):
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                          dtype=x.dtype, device=x.device)
        ky = kx.t()
        n = x.shape[1]
        gx = F.conv2d(x, kx.expand(n, 1, 3, 3), padding=1, groups=n)
        gy = F.conv2d(x, ky.expand(n, 1, 3, 3), padding=1, groups=n)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred, target):
        l1 = (pred - target).abs().mean()
        ssim = ssim_per_image(pred, target).mean()
        loss = self.alpha * l1 + (1 - self.alpha) * (1 - ssim)
        if self.mse_w > 0:
            loss = loss + self.mse_w * F.mse_loss(pred, target)
        if self.edge_w > 0:
            loss = loss + self.edge_w * (self._grad_mag(pred) - self._grad_mag(target)).abs().mean()
        return loss, l1.detach(), ssim.detach()
