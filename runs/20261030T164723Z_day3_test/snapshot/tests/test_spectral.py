"""SVD orientation, projectors, gelu_new, Gauss-Hermite expectation (CPU)."""
import numpy as np
import pytest
import torch
from transformers.activations import ACT2FN

from scp.spectral import (calibrated_moments, discarded_preact, gauss_expect_gelu, gelu_new, svd_w_in,
                          weights_only_moments)


def test_gelu_new_matches_hf():
    x = torch.linspace(-12, 12, 20001, dtype=torch.float64)
    assert torch.equal(gelu_new(x), ACT2FN["gelu_new"](x))
    x32 = x.float()
    assert torch.equal(gelu_new(x32), ACT2FN["gelu_new"](x32))


def test_svd_orientation_and_reconstruction(cpu_model, cfg, spectrum):
    model, _ = cpu_model
    Wc = model.transformer.h[cfg["layer"]].mlp.c_fc.weight.detach().double()
    assert tuple(Wc.shape) == (768, 3072)  # HF Conv1D: [d_model, d_mlp], computes x @ Wc
    W_in = spectrum.W_in
    assert tuple(W_in.shape) == (3072, 768) and torch.equal(W_in, Wc.T)
    x = torch.randn(5, 768, dtype=torch.float64)
    assert torch.allclose(x @ Wc, (W_in @ x.T).T)  # column-vector equation uses the transpose
    assert tuple(spectrum.V.shape) == (768, 768)
    for k in cfg["ranks"]:
        P_R, P_B = spectrum.projectors(k)
        # plain rank-k truncation U_k S_k V_k^T equals W_in P_R; residual energy = sum of discarded s^2
        trunc = spectrum.U[:, :k] @ torch.diag(spectrum.S[:k]) @ spectrum.V[:, :k].T
        assert torch.allclose(W_in @ P_R, trunc, atol=1e-10)
        err2 = (W_in - W_in @ P_R).pow(2).sum()
        assert torch.isclose(err2, spectrum.S[k:].pow(2).sum(), rtol=1e-9)
        # row-vector form used in the model: (x @ P_R) @ Wc == (trunc @ x^T)^T
        assert torch.allclose((x @ P_R) @ Wc, (trunc @ x.T).T, atol=1e-9)
    # Same input subspace from the SVD of the Conv1D weight itself (left singular vectors of Wc).
    U2, _, _ = torch.linalg.svd(Wc, full_matrices=False)
    k = cfg["ranks"][1]
    assert torch.allclose(U2[:, :k] @ U2[:, :k].T, spectrum.projectors(k)[0], atol=1e-8)


def test_projectors_complementary(spectrum, cfg):
    I = torch.eye(768, dtype=torch.float64)
    for k in cfg["ranks"] + [768]:
        P_R, P_B = spectrum.projectors(k)
        assert torch.allclose(P_R + P_B, I, atol=1e-12)
        assert torch.allclose(P_R @ P_R, P_R, atol=1e-10) and torch.allclose(P_B @ P_B, P_B, atol=1e-10)
        assert torch.allclose(P_R, P_R.T, atol=1e-12)
        assert torch.allclose(P_R @ P_B, torch.zeros_like(I), atol=1e-10)
        assert abs(float(torch.trace(P_R)) - k) < 1e-8


def test_gauss_sigma_zero_is_exact_gelu_new():
    m = torch.linspace(-10, 10, 4001)
    zero = torch.zeros_like(m)
    assert torch.equal(gauss_expect_gelu(m, zero), gelu_new(m))
    m64 = m.double()
    assert torch.equal(gauss_expect_gelu(m64, zero.double()), gelu_new(m64))


def test_quadrature_node_convergence(cfg):
    """Configured node count converges (< 1e-7 abs vs 512 nodes) over sigma in [0, checked bound].

    Accuracy is NOT claimed beyond the bound (32 nodes at sigma=4 err ~5e-3); Reduction refuses larger sigma."""
    n, bound = cfg["quadrature_nodes"], cfg["quadrature_sigma_max_checked"]
    m = torch.linspace(-20, 20, 801, dtype=torch.float64).reshape(-1, 1)
    s = torch.linspace(0, bound, 41, dtype=torch.float64).reshape(1, -1)
    ref = gauss_expect_gelu(m, s, 512)
    errs = {k: float((gauss_expect_gelu(m, s, k) - ref).abs().max()) for k in (16, 32, 64, 128)}
    assert errs[n] < 1e-7, errs
    assert errs[16] >= errs[32] >= errs[64]


def test_reduction_refuses_unchecked_sigma(cfg, spectrum, cpu_model):
    from scp.model import make_reduction
    model, _ = cpu_model
    mlp = model.transformer.h[cfg["layer"]].mlp
    P_R, _ = spectrum.projectors(192)
    bound = cfg["quadrature_sigma_max_checked"]
    sigma = torch.full((3072,), bound * 1.01, dtype=torch.float64)
    with pytest.raises(ValueError):
        make_reduction(P_R, mlp, torch.zeros(3072), sigma, "gauss", "last_token", cfg["quadrature_nodes"], bound,
                       torch.device("cpu"))


def test_quadrature_vs_independent_monte_carlo():
    g = torch.Generator().manual_seed(1234)
    eps = torch.randn(4_000_000, generator=g, dtype=torch.float64)
    for m, s in [(-3.0, 0.5), (-1.0, 1.0), (0.0, 0.3), (0.5, 2.0), (2.0, 1.5), (-0.2, 3.0)]:
        samples = gelu_new(m + s * eps)
        mc, se = float(samples.mean()), float(samples.std() / np.sqrt(len(eps)))
        quad = float(gauss_expect_gelu(torch.tensor(m, dtype=torch.float64), torch.tensor(s, dtype=torch.float64)))
        assert abs(quad - mc) < 5 * se + 1e-12, (m, s, quad, mc, se)


def test_independent_gaussian_q_local_mean_synthetic():
    """If q ~ N(mu, sigma^2) independent of z, E[g(z+q) | z] = gauss_expect_gelu(z+mu, sigma) (local mean only)."""
    g = torch.Generator().manual_seed(7)
    z = torch.randn(6, dtype=torch.float64, generator=g)
    mu, sigma = 0.3, 0.8
    q = mu + sigma * torch.randn(2_000_000, 1, dtype=torch.float64, generator=g)
    mc = gelu_new(z + q)
    se = mc.std(0) / np.sqrt(q.shape[0])
    quad = gauss_expect_gelu(z + mu, torch.full_like(z, sigma))
    assert ((quad - mc.mean(0)).abs() < 5 * se).all()
    # and it is not the same as the bias-only g(z + mu) (the compensations differ)
    assert (quad - gelu_new(z + mu)).abs().max() > 1e-2


def test_moments_definitions(spectrum, cpu_model, cfg):
    model, _ = cpu_model
    mlp = model.transformer.h[cfg["layer"]].mlp
    Wc = mlp.c_fc.weight.detach().double()
    P_R, P_B = spectrum.projectors(192)
    x = torch.randn(300, 768, dtype=torch.float64)
    # a = z + q exactly
    a = x @ Wc + mlp.c_fc.bias.double()
    z = (x @ P_R) @ Wc + mlp.c_fc.bias.double()
    assert torch.allclose(a, z + discarded_preact(x, P_B, Wc), atol=1e-9)
    mu, sd = calibrated_moments(x, P_B, Wc)
    assert mu.shape == (3072,) and sd.shape == (3072,)
    # full rank: nothing discarded -> zero moments
    _, P_B_full = spectrum.projectors(768)
    mu0, sd0 = calibrated_moments(x, P_B_full, Wc)
    assert mu0.abs().max() < 1e-9 and sd0.abs().max() < 1e-9
    ln = model.transformer.h[cfg["layer"]].ln_2
    mw, sw = weights_only_moments(P_B_full, Wc, ln.weight, ln.bias)
    assert mw.abs().max() < 1e-9 and sw.abs().max() < 1e-9
    mw, sw = weights_only_moments(P_B, Wc, ln.weight, ln.bias)
    assert (sw > 0).all()
