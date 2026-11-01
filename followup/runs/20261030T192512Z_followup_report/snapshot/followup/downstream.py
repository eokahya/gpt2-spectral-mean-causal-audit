"""Downstream sensitivities at the ORIGINAL UNPERTURBED recipient: fixed Rademacher VJPs of the centred
final-position logits with respect to the block-L MLP output at the last token.

Only the leaf (the local MLP output at the last token) requires grad. The model is frozen and in eval mode;
nothing upstream of the leaf, no donor activation and no feature tensor is part of the graph. The residual
bypass is untouched: the leaf replaces only the MLP output, which the block adds to its residual as usual."""
import contextlib

import torch
from torch import nn

from scp.model import SpectralMLP


def rademacher_probes(n, vocab, seed):
    """[n, vocab] float64 +-1 probes from a fixed CPU generator (identical for every recipient and stage)."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randint(0, 2, (n, vocab), generator=g, dtype=torch.int64).to(torch.float64) * 2 - 1


def centred_probes(U):
    """<centred(l), u> = <l, u - mean(u)>: centring moves onto the probe."""
    return U - U.mean(-1, keepdim=True)


class LeafMLP(nn.Module):
    """Wraps the original MLP; optionally replaces its output at the final real position with `leaf`."""

    def __init__(self, orig):
        super().__init__()
        self.orig = orig
        self.last_idx = None
        self.leaf = None
        self.out_last = None

    def forward(self, x):
        with torch.no_grad():
            out = self.orig(x)
        ar = torch.arange(x.shape[0], device=x.device)
        self.out_last = out[ar, self.last_idx].detach()
        if self.leaf is None:
            return out
        mask = torch.zeros(out.shape[:2], dtype=torch.bool, device=out.device)
        mask[ar, self.last_idx] = True
        return torch.where(mask.unsqueeze(-1), self.leaf.unsqueeze(1).expand_as(out), out)


@contextlib.contextmanager
def leaf_mlp(model, layer):
    block = model.transformer.h[layer]
    orig = block.mlp
    if isinstance(orig, (SpectralMLP, LeafMLP)):
        raise RuntimeError("block MLP already wrapped; nesting is not allowed")
    w = LeafMLP(orig)
    block.mlp = w
    try:
        yield w
    finally:
        block.mlp = orig


def _logits_last(model, batch):
    hs = model.transformer(input_ids=batch.ids, attention_mask=batch.mask).last_hidden_state
    ar = torch.arange(hs.shape[0], device=hs.device)
    return model.lm_head(hs[ar, batch.last_idx])


def leaf_logits(model, layer, batch, leaf_value):
    """No-grad final-position logits with the last-token MLP output set to leaf_value (FD / smoke helper)."""
    with leaf_mlp(model, layer) as lm, torch.no_grad():
        lm.last_idx = batch.last_idx
        lm.leaf = leaf_value
        return _logits_last(model, batch)


def probe_vjps(model, layer, batch, Uc_dev):
    """Returns (G [B, n, d] on the model device/dtype, m0 [B, d], logits0 [B, V], max |leaf logits - logits0|).

    g_j = J^T u_j with J = d centred(logits_last) / d m at m = m0 (the original unperturbed MLP output)."""
    if model.training:
        raise RuntimeError("model must be in eval mode")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("model parameters must be frozen")
    with leaf_mlp(model, layer) as lm:
        lm.last_idx = batch.last_idx
        with torch.no_grad():
            logits0 = _logits_last(model, batch)
        m0 = lm.out_last.clone()
        with torch.enable_grad():
            leaf = m0.clone().requires_grad_(True)
            lm.leaf = leaf
            logits = _logits_last(model, batch)
            leaf_diff = float((logits.detach() - logits0).abs().max())
            G = []
            n = Uc_dev.shape[0]
            for j in range(n):
                (g,) = torch.autograd.grad((logits * Uc_dev[j]).sum(), leaf, retain_graph=j < n - 1)
                G.append(g.detach())
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("a parameter received a gradient")
    return torch.stack(G, 1), m0, logits0, leaf_diff


def exact_jvp(model, layer, batch, m0, v):
    """Exact J v (centred logits) by the double-backward trick (technical smoke only)."""
    with leaf_mlp(model, layer) as lm, torch.enable_grad():
        lm.last_idx = batch.last_idx
        leaf = m0.clone().requires_grad_(True)
        lm.leaf = leaf
        y = _logits_last(model, batch)
        y = y - y.mean(-1, keepdim=True)
        w = torch.zeros_like(y, requires_grad=True)
        (gw,) = torch.autograd.grad((y * w).sum(), leaf, create_graph=True)
        (jv,) = torch.autograd.grad((gw * v).sum(), w)
    return jv.detach()


def directional(G, v):
    """sqrt(mean_j (g_j . v)^2) per row; G [B, n, d], v [B, d] (float64 CPU)."""
    return torch.einsum("bjd,bd->bj", G, v).pow(2).mean(-1).sqrt()
