"""Exponential moving average of model weights.

The EMA weights are what gets evaluated and checkpointed (they are almost always a bit
better than the raw weights on restoration metrics); the raw weights are kept in the
checkpoint as `model_raw` for inspection.
"""
import torch


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:  # BN num_batches_tracked etc.
                self.shadow[k].copy_(v)

    @torch.no_grad()
    def copy_to(self, model):
        """Overwrite model weights with the EMA weights (for evaluation / saving)."""
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def store(self, model):
        """Remember the raw training weights so they can be put back afterwards."""
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def restore(self, model):
        """Put the raw training weights back after an EMA evaluation."""
        if self.backup is None:
            raise RuntimeError("ModelEMA.restore() called without a preceding store()")
        model.load_state_dict(self.backup, strict=True)
        self.backup = None
