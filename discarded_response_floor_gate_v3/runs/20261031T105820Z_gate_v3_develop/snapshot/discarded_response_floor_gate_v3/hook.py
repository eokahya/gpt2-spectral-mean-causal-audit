"""Replace the block-L MLP branch input (ln_2 output) at the final real token by a given vector.

The residual stream (block input + attention output) that the MLP output is added to is untouched, other positions
are untouched, and every later layer is recomputed by the original network."""
import contextlib

import torch
from torch import nn

from scp.model import SpectralMLP


class BranchMLP(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.orig = orig
        self.last_idx = None
        self.branch = None
        self.x_in_last = None

    def forward(self, x):
        ar = torch.arange(x.shape[0], device=x.device)
        self.x_in_last = x[ar, self.last_idx].detach()
        if self.branch is not None:
            x = x.clone()
            x[ar, self.last_idx] = self.branch.to(device=x.device, dtype=x.dtype)
        return self.orig(x)


@contextlib.contextmanager
def branch_mlp(model, layer):
    block = model.transformer.h[layer]
    orig = block.mlp
    if isinstance(orig, (SpectralMLP, BranchMLP)):
        raise RuntimeError("block MLP already wrapped; nesting is not allowed")
    w = BranchMLP(orig)
    block.mlp = w
    try:
        yield w
    finally:
        block.mlp = orig


@torch.inference_mode()
def forward(model, w, batch, branch=None):
    """Final-position logits [B, V] and the unpatched branch input at the final token [B, d]."""
    if model.training:
        raise RuntimeError("model must be in eval mode")
    w.last_idx, w.branch = batch.last_idx, branch
    try:
        hs = model.transformer(input_ids=batch.ids, attention_mask=batch.mask).last_hidden_state
        ar = torch.arange(hs.shape[0], device=hs.device)
        return {"logits": model.lm_head(hs[ar, batch.last_idx]), "x_in_last": w.x_in_last, "hidden_all": hs}
    finally:
        w.branch = None
