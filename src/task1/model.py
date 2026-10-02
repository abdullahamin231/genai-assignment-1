import torch
import torch.nn as nn

# Skip levels are named after the encoder stage resolution they come from.
SKIP_LEVELS = ("8", "16", "32", "64")


def parse_skips(v):
    """Canonicalise a skip spec: None / "" / "none" / "16,32" / ("16", 32) -> ("16", "32")."""
    if v is None:
        return ()
    parts = [p.strip() for p in v.split(",")] if isinstance(v, str) else [str(p).strip() for p in v]
    out = tuple(p for p in parts if p and p.lower() != "none")
    bad = set(out) - set(SKIP_LEVELS)
    if bad:
        raise ValueError(f"unknown skip levels {sorted(bad)}; choose from {SKIP_LEVELS} or 'none'")
    return out


def conv_block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


def up_block(cin, cout):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), conv_block(cin, cout))


class SelfAttention2d(nn.Module):
    """Single-head non-local block for low-resolution feature maps.

    Global receptive field at 8x8 (64 tokens) is what lets the decoder inpaint occluded
    regions using context from outside the mask instead of only local evidence.

    Written with bmm/softmax/reshape_as (no fused SDPA op) so it exports cleanly to ONNX
    opset 17 with a dynamic batch axis. The residual gain `gamma` is zero-initialised, so
    the block starts as an exact identity and cannot hurt early training.
    """

    def __init__(self, ch, qk_ch=None):
        super().__init__()
        qk = qk_ch or max(ch // 2, 1)
        self.theta = nn.Conv2d(ch, qk, 1, bias=False)
        self.phi = nn.Conv2d(ch, qk, 1, bias=False)
        self.g = nn.Conv2d(ch, ch, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = qk ** -0.5

    def forward(self, x):
        th = self.theta(x).flatten(2)                       # B, qk, N
        ph = self.phi(x).flatten(2)                         # B, qk, N
        att = torch.softmax((th.transpose(1, 2) @ ph) * self.scale, dim=-1)  # B, N, N
        gv = self.g(x).flatten(2)                           # B, C, N
        out = torch.bmm(gv, att.transpose(1, 2)).reshape_as(x)  # B, C, H, W
        return x + self.gamma * self.proj(out)


class UniversalAE(nn.Module):
    """Universal multi-corruption denoising autoencoder (Task 1).

    Encoder: 128->64->32->16->8 spatial, channels c,2c,4c,8c -> 1x1 conv -> flatten ->
    dense latent vector (the genuine compressed bottleneck) -> dense -> 8x8x bottleneck_ch.
    Decoder mirrors it. No input->output residual path exists: everything the decoder
    knows must pass through the latent (plus, optionally, the *limited* skips below).

    Optional, independently switchable additions (defaults keep the v1 behaviour when a
    checkpoint was written before they existed):

    * ``skips`` -- limited encoder->decoder skip connections. Encoder features are first
      narrowed by a 1x1 conv to ``skip_ch`` (32 vs. 48..384 decoder channels) and then
      concatenated, so they carry local edge/texture hints but cannot bypass the latent
      wholesale. The assignment allows limited skips provided their purpose and effect
      are investigated and justified; Optuna searches the skip pattern so the report can
      present that ablation.
    * ``attn`` -- one zero-init self-attention block on the 8x8 bottleneck features
      (global context, mainly for occlusion inpainting).
    * ``refine`` -- an extra full-resolution conv block before the output conv; the
      64->128 stage was the thinnest part of v1.
    """

    def __init__(self, base_channels=32, latent_dim=256, dropout=0.1, bottleneck_ch=32,
                 skips=(), skip_ch=32, attn=False, refine=False):
        super().__init__()
        c = base_channels
        skips = parse_skips(skips)
        self.bottleneck_ch = bottleneck_ch
        self.latent_dim = latent_dim
        self.skips = skips
        self.skip_ch = skip_ch
        self.attn = bool(attn)
        self.refine = bool(refine)

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

        # Decoder keys/weights are byte-identical to v1 when skips=() (the extra input
        # channels only exist when the matching skip level is enabled).
        def up(cin, cout, level):
            return up_block(cin + (skip_ch if level in skips else 0), cout)

        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_ch, 8 * c, 1), nn.BatchNorm2d(8 * c), nn.ReLU(inplace=True),
            up(8 * c, 4 * c, "8"),    # 16
            up(4 * c, 2 * c, "16"),   # 32
            up(2 * c, c, "32"),       # 64
            up(c, c, "64"),           # 128
            nn.Conv2d(c, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
        enc_ch = {"8": 8 * c, "16": 4 * c, "32": 2 * c, "64": c}
        self.skip_projs = nn.ModuleDict({f"s{l}": nn.Conv2d(enc_ch[l], skip_ch, 1) for l in skips})
        self.attn_block = SelfAttention2d(8 * c) if self.attn else nn.Identity()
        self.refine_block = conv_block(c, c) if self.refine else nn.Identity()

    def _encode(self, x):
        """Stage-by-stage encoder pass; returns (bottleneck, {level: feature})."""
        feats, h = {}, x
        for i, level in enumerate(("64", "32", "16", "8")):
            h = self.encoder[i](h)
            feats[level] = h
        return self.encoder[4](h), feats

    def encode(self, x):
        return self.to_latent(self.encoder(x))

    def decode(self, z, feats=None):
        if feats is None and self.skips:
            raise ValueError("decode() needs encoder features when skips are enabled; use forward()")
        h = self.from_latent(z).view(-1, self.bottleneck_ch, 8, 8)
        h = self.decoder[0](h)
        h = self.decoder[1](h)
        h = self.decoder[2](h)
        if self.attn:
            h = self.attn_block(h)
        for i, level in enumerate(("8", "16", "32", "64")):
            if level in self.skips:
                h = torch.cat([h, self.skip_projs[f"s{level}"](feats[level])], dim=1)
            h = self.decoder[3 + i](h)
        if self.refine:
            h = self.refine_block(h)
        h = self.decoder[7](h)
        return self.decoder[8](h)

    def forward(self, x):
        bottleneck, feats = self._encode(x)
        z = self.to_latent(bottleneck)
        return self.decode(z, feats)
