"""GPT-2 loading, the block-L MLP wrapper (patch / reduce / capture) and forward helpers.

Intervention scope, exactly:
* patch: x' = x + delta at the final real position of each prompt, applied to the MLP input (ln_2 output) of
  block L only. The residual bypass (block input + attention output) is untouched; other positions are untouched;
  later layers recompute normally.
* reduction scope 'last_token': reduced/compensated W_in used only at the final real position (local diagnosis).
* reduction scope 'full_layer': reduced/compensated W_in used at every position (actual module replacement).
"""
import contextlib
from dataclasses import dataclass

import torch
from torch import nn

from .common import sha256_file
from .spectral import gauss_expect_gelu, gelu_new

METHODS = ("zero", "bias", "gauss", "gauss_wo")
SCOPES = ("last_token", "full_layer")


def load_model(cfg, device):
    from transformers import AutoTokenizer, GPT2LMHeadModel
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"])
    model = GPT2LMHeadModel.from_pretrained(
        m["name"], revision=m["revision"], dtype=getattr(torch, m["dtype"]),
        attn_implementation=m.get("attn_implementation", "eager"))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def model_file_info(cfg):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(cfg["model"]["name"], "model.safetensors", revision=cfg["model"]["revision"])
    return {"safetensors_path": path, "safetensors_sha256": sha256_file(path)}


@dataclass
class Reduction:
    """Reduced W_in = W_in P_R with a compensation of the discarded pre-activation q.

    Tensors are float32 on the model device; mu/sigma are fixed (never updated from the evaluated x)."""
    P_R: torch.Tensor
    Wc: torch.Tensor
    b: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    method: str
    scope: str
    n_nodes: int
    sigma_bound: float

    def __post_init__(self):
        if self.method not in METHODS or self.scope not in SCOPES:
            raise ValueError(f"bad method/scope {self.method}/{self.scope}")
        # quadrature accuracy was only verified for sigma <= sigma_bound (quadcheck stage / tests)
        if float(self.sigma.max()) > self.sigma_bound:
            raise ValueError(f"sigma max {float(self.sigma.max()):.3f} exceeds checked bound {self.sigma_bound}")

    def hidden(self, x):
        z = (x @ self.P_R) @ self.Wc + self.b
        if self.method == "zero":
            return gelu_new(z)
        if self.method == "bias":
            return gelu_new(z + self.mu)
        return gauss_expect_gelu(z + self.mu, self.sigma, self.n_nodes)


def make_reduction(P_R64, mlp, mu64, sigma64, method, scope, n_nodes, sigma_bound, device):
    def f32(t):  # cast on CPU first (MPS has no float64), then move
        return t.detach().cpu().float().to(device)
    zeros = torch.zeros(mlp.c_fc.weight.shape[1], dtype=torch.float32, device=device)
    if method == "zero":
        mu, sigma = zeros, zeros
    elif method == "bias":
        mu, sigma = f32(mu64), zeros
    else:
        mu, sigma = f32(mu64), f32(sigma64)
    return Reduction(P_R=f32(P_R64), Wc=mlp.c_fc.weight, b=mlp.c_fc.bias, mu=mu, sigma=sigma,
                     method=method, scope=scope, n_nodes=n_nodes,
                     sigma_bound=sigma_bound)


class SpectralMLP(nn.Module):
    """Drop-in replacement for a GPT2MLP that can patch its input, reduce W_in and capture activations."""

    def __init__(self, orig):
        super().__init__()
        self.orig = orig
        self.last_idx = None
        self.delta = None
        self.reduction = None
        self.capture_all = False
        self.cache = {}

    def forward(self, x):
        B = x.shape[0]
        ar = torch.arange(B, device=x.device)
        li = self.last_idx
        self.cache = {"x_pre_last": x[ar, li].detach()}
        if self.capture_all:
            self.cache["x_all"] = x.detach()
        if self.delta is not None:
            x = x.clone()
            x[ar, li] = x[ar, li] + self.delta
        h = self.orig.act(self.orig.c_fc(x))
        red = self.reduction
        if red is not None:
            if red.scope == "full_layer":
                h = red.hidden(x)
            else:
                h = h.clone()
                h[ar, li] = red.hidden(x[ar, li])
        out = self.orig.dropout(self.orig.c_proj(h))
        self.cache["x_last"] = x[ar, li].detach()
        self.cache["out_last"] = out[ar, li].detach()
        self.cache["out_all"] = out.detach()
        return out


@contextlib.contextmanager
def wrapped_mlp(model, layer):
    """Install SpectralMLP at block `layer`; the original module is always restored on exit."""
    block = model.transformer.h[layer]
    orig = block.mlp
    if isinstance(orig, SpectralMLP):
        raise RuntimeError("SpectralMLP already installed; nested wrapping is not allowed")
    wrapper = SpectralMLP(orig)
    block.mlp = wrapper
    try:
        yield wrapper
    finally:
        block.mlp = orig


@torch.no_grad()
def forward_last(model, wrapper, batch, delta=None, reduction=None, capture_all=False):
    """Run the model; return final-position logits and the wrapper's captures. Wrapper state is reset after."""
    wrapper.last_idx = batch.last_idx
    wrapper.delta = delta
    wrapper.reduction = reduction
    wrapper.capture_all = capture_all
    try:
        hs = model.transformer(input_ids=batch.ids, attention_mask=batch.mask).last_hidden_state
        ar = torch.arange(hs.shape[0], device=hs.device)
        logits = model.lm_head(hs[ar, batch.last_idx])
        out = dict(wrapper.cache)
        out["logits"] = logits
        out["hidden_all"] = hs
        return out
    finally:
        wrapper.delta = None
        wrapper.reduction = None
        wrapper.capture_all = False
        wrapper.cache = {}
