"""SVD of W_in, subspace projectors, discarded pre-activation moments, Gaussian expectation of gelu_new.

Orientation: HF Conv1D c_fc.weight `Wc` is [d_model, d_mlp] and computes x @ Wc + b (row vectors).
The column-vector matrix is W_in = Wc.T ([d_mlp, d_model]); its right singular vectors V live in the
d_model input space, so P_R = V_k V_k^T acts on x. Row-vector form of the reduced layer: (x @ P_R) @ Wc + b.
"""
import math
from dataclasses import dataclass

import torch
from scipy.special import roots_hermitenorm

_GELU_C = math.sqrt(2.0 / math.pi)


def gelu_new(x):
    """GPT-2's tanh-approximated GELU (identical formula to transformers' NewGELUActivation)."""
    return 0.5 * x * (1.0 + torch.tanh(_GELU_C * (x + 0.044715 * torch.pow(x, 3.0))))


def hermite_rule(n_nodes):
    """Probabilists' Gauss-Hermite rule for E_{eps~N(0,1)}; weights renormalised to sum exactly to 1.

    scipy's Golub-Welsch roots are used (numpy's hermegauss overflows for several hundred nodes)."""
    t, w = roots_hermitenorm(n_nodes)
    return t.tolist(), (w / w.sum()).tolist()


def gauss_expect_gelu(m, sigma, n_nodes=32):
    """Elementwise E[gelu_new(m + sigma * eps)], eps ~ N(0,1), by Gauss-Hermite quadrature.

    sigma broadcasts against m. Where sigma == 0 the result is exactly gelu_new(m)."""
    nodes, weights = hermite_rule(n_nodes)
    acc = torch.zeros_like(m)
    for t, w in zip(nodes, weights):
        acc = acc + w * gelu_new(m + sigma * t)
    return torch.where(sigma == 0, gelu_new(m), acc)


@dataclass
class Spectrum:
    W_in: torch.Tensor  # [d_mlp, d_model] float64, CPU
    U: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor  # [d_model, d_model], columns = right singular vectors

    @property
    def d_model(self):
        return self.V.shape[0]

    def projectors(self, k):
        """(P_R, P_B) in float64 on CPU for rank cut k."""
        Vk = self.V[:, :k]
        P_R = Vk @ Vk.T
        P_B = torch.eye(self.d_model, dtype=P_R.dtype) - P_R
        return P_R, P_B

    def retained_energy(self, k):
        e = self.S.pow(2)
        return float(e[:k].sum() / e.sum())


def svd_w_in(c_fc_weight):
    """SVD of the column-vector W_in = c_fc.weight.T, on CPU in float64 (recorded in run manifests)."""
    Wc = c_fc_weight.detach().cpu().to(torch.float64)
    if Wc.shape[0] >= Wc.shape[1]:
        raise ValueError(f"expected Conv1D weight [d_model, d_mlp], got {tuple(Wc.shape)}")
    W_in = Wc.T.contiguous()
    U, S, Vh = torch.linalg.svd(W_in, full_matrices=False)
    return Spectrum(W_in=W_in, U=U, S=S, V=Vh.T.contiguous())


def discarded_preact(x, P_B, Wc):
    """q = W_in P_B x in row form: (x @ P_B) @ Wc. No bias (the bias stays in z)."""
    return (x @ P_B) @ Wc


def calibrated_moments(x, P_B, Wc):
    """Per-neuron mean and std of q over the rows of x (float64)."""
    q = discarded_preact(x.to(torch.float64), P_B.to(torch.float64), Wc.to(torch.float64))
    return q.mean(0), q.std(0, unbiased=True)


def weights_only_moments(P_B, Wc, ln_weight, ln_bias):
    """Isotropy assumption only: LN-normalised coordinates n ~ (0, I), so x = gamma*n + beta has
    mean beta and is treated as isotropic with per-coordinate variance tau^2 = ||gamma||^2 / d.
    mu_i = (W_in P_B beta)_i, sigma_i = tau * ||(W_in P_B)_{i,:}||."""
    P_B, Wc = P_B.to(torch.float64), Wc.to(torch.float64)
    gamma, beta = ln_weight.detach().cpu().to(torch.float64), ln_bias.detach().cpu().to(torch.float64)
    tau = gamma.norm() / math.sqrt(gamma.numel())
    M = P_B @ Wc  # [d_model, d_mlp]; column i is (W_in P_B)_{i,:}
    return beta @ M, tau * M.norm(dim=0)
