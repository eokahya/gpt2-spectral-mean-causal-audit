"""Day-2 bases, whitening algebra, matching, contrasts, guards (PROTOCOL_DAY2)."""
import subprocess
import sys

import numpy as np
import pytest
import torch

from scp import ROOT
from scp.bases import (Basis, build_bases, check_projector, coord_family, coord_order, haar_orthogonal, pca_family,
                       random_family, svd_family, whitening, wsvd_family)
from scp.common import load_config
from scp.data import make_batch
from scp.day2 import indicator_alignment, paired_contrast
from scp.matching import apply_match, match_scale, preact_norm
from scp.model import forward_last, load_model, make_reduction, wrapped_mlp
from scp.spectral import calibrated_moments, gauss_expect_gelu, weights_only_moments

F64 = torch.float64


@pytest.fixture(scope="session")
def cfg2():
    c = load_config(ROOT / "config" / "day2_v1.json")
    assert c["quadrature_nodes"] is not None, "run run_day2.py quadcheck and fix nodes in config first"
    return c


@pytest.fixture(scope="session")
def pool_x(cpu_model, pilot):
    model, tok = cpu_model
    _, _, moment = pilot
    b = make_batch([m["ids"] for m in moment], tok.eos_token_id, torch.device("cpu"))
    with wrapped_mlp(model, 9) as w:
        o = forward_last(model, w, b, capture_all=True)
    return o["x_pre_last"].to(F64), o["x_all"][b.mask.bool()].to(F64)


@pytest.fixture(scope="session")
def mlp(cpu_model):
    return cpu_model[0].transformer.h[9].mlp


def _pair_batches(cpu_model, small):
    model, _ = cpu_model
    with wrapped_mlp(model, 9) as w:
        rec = forward_last(model, w, small["rec"])
        don = forward_last(model, w, small["don"])
    return rec, don


# ------------------------------------------------------------------ bases

def test_pca_deterministic_and_matches_numpy(pool_x):
    X = pool_x[0]
    a, b = pca_family(X)(192), pca_family(X)(192)
    assert torch.equal(a.P, b.P)
    Xn = X.numpy()
    ev, U = np.linalg.eigh(np.cov(Xn, rowvar=False))
    Uk = U[:, np.argsort(-ev, kind="stable")[:192]]
    assert np.allclose(a.P.numpy(), Uk @ Uk.T, atol=1e-8)
    assert torch.allclose(a.P, a.P.T, atol=1e-12)


def test_coord_selection_and_tie_break():
    X = torch.tensor([[1.0, 2.0, 2.0, 0.5], [1.0, 2.0, 2.0, 0.5]], dtype=F64)  # cols 1 and 2 tie
    order, _ = coord_order(X)
    assert list(order[:3]) == [1, 2, 0]
    P = coord_family(X)(1).P
    assert P[1, 1] == 1 and P.sum() == 1  # tie -> lower index


def test_coord_real_topk(pool_x):
    X = pool_x[0]
    P = coord_family(X)(96).P
    m2 = X.pow(2).mean(0)
    chosen = torch.diagonal(P).bool()
    assert chosen.sum() == 96 and m2[chosen].min() >= m2[~chosen].max()


def test_random_seed_repeat_and_orthogonality():
    O1, O2, O3 = haar_orthogonal(768, 101), haar_orthogonal(768, 101), haar_orthogonal(768, 202)
    assert torch.equal(O1, O2) and not torch.allclose(O1, O3)
    assert torch.allclose(O1.T @ O1, torch.eye(768, dtype=F64), atol=1e-10)
    fam = random_family(768, 303)
    P96, P192 = fam(96).P, fam(192).P
    assert torch.allclose(P192 @ P96, P96, atol=1e-10)  # nested in k
    assert fam(96).id == "random_s303"


def test_whitening_regularization(pool_x):
    X = pool_x[0]
    C, L, Linv, m, diag = whitening(X, 0.01)
    S = torch.from_numpy(np.cov(X.numpy(), rowvar=False))
    lam = 0.01 * float(torch.trace(S)) / 768
    assert abs(diag["lambda"] - lam) < 1e-12 * lam
    assert torch.allclose(C - S, lam * torch.eye(768, dtype=F64), atol=1e-10)
    assert torch.allclose(L @ L.T, C, atol=1e-9) and torch.allclose(L @ Linv, torch.eye(768, dtype=F64), atol=1e-8)
    assert diag["eig_cov_min"] < 1e-6 * diag["eig_cov_max"]  # covariance is (near) singular: n=576 < d=768 + LN
    assert np.isfinite(diag["cond_C"]) and diag["cond_C"] > 1


def test_whitened_oblique_algebra(pool_x, spectrum):
    X = pool_x[0]
    fam = wsvd_family(spectrum.W_in, X, 0.01)
    parts = fam.parts
    I = torch.eye(768, dtype=F64)
    for k in (96, 192, 384):
        b = fam(k)
        P, Q = b.P, b.Q
        assert torch.allclose(P @ P, P, atol=1e-7 * float(P.abs().max()) ** 2)
        assert torch.allclose(P + Q, I, atol=1e-12)
        assert b.meta["asym_rel"] > 1e-3  # oblique: P != P^T
        # independent recomputation: W_in P == (W_in L)_k L^-1
        U, S, Vh = torch.linalg.svd(spectrum.W_in @ parts["L"], full_matrices=False)
        Wk = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k] @ torch.linalg.inv(parts["L"])
        assert torch.allclose(spectrum.W_in @ P, Wk, atol=1e-7 * float(Wk.abs().max()))
        resid = ((spectrum.W_in - spectrum.W_in @ P) @ parts["L"]).pow(2).sum()
        assert torch.isclose(resid, S[k:].pow(2).sum(), rtol=1e-7)
        # orientation: column map P x  <->  row map x_row @ P^T
        x = X[:5]
        R_row, B_row = b.row_maps()
        assert torch.allclose((P @ x.T).T, x @ R_row, atol=1e-9)
        assert torch.allclose(x @ R_row + x @ B_row, x, atol=1e-9)
    assert torch.allclose(fam(768).P, I, atol=1e-8)


def test_check_projector_rejects_non_projector():
    with pytest.raises(ValueError):
        check_projector(Basis("svd", 2, torch.eye(4, dtype=F64) * 0.5))


def test_build_bases_deterministic(cfg2, spectrum, pool_x):
    a = build_bases(cfg2, spectrum, pool_x[0])
    b = build_bases(cfg2, spectrum, pool_x[0])
    ids = {bid for bid, _ in a}
    assert ids == {"svd", "pca", "coord", "wsvd"} | {f"random_s{s}" for s in cfg2["bases"]["random_seeds"]}
    assert len(cfg2["bases"]["random_seeds"]) >= 3
    for key in a:
        assert torch.equal(a[key].P, b[key].P)
    assert torch.allclose(a[("svd", 192)].P, spectrum.projectors(192)[0])


# ------------------------------------------------------------------ moments / reductions

def _families(spectrum, X):
    return [svd_family(spectrum), pca_family(X), coord_family(X), random_family(768, 101),
            wsvd_family(spectrum.W_in, X, 0.01)]


def _red(cfg2, basis, mom, method, scope, mlp_, device=torch.device("cpu")):
    mu, sd = mom
    return make_reduction(basis.row_maps()[0], mlp_, mu, sd, method, scope, cfg2["quadrature_nodes"],
                          cfg2["quadrature_sigma_max_checked"], device)


def test_weights_only_oblique_row_map(pool_x, spectrum, mlp):
    X = pool_x[0]
    b = wsvd_family(spectrum.W_in, X, 0.01)(192)
    Wc = mlp.c_fc.weight.detach().to(F64)
    _, B_row = b.row_maps()
    # row form q_row = x_row @ Q^T @ Wc equals column form (W_in Q x)^T
    x = X[:7]
    assert torch.allclose(x @ B_row @ Wc, (spectrum.W_in @ b.Q @ x.T).T, atol=1e-8)
    mu, sd = calibrated_moments(X, B_row, Wc)
    qcol = (spectrum.W_in @ b.Q @ X.T).T
    assert torch.allclose(mu, qcol.mean(0), atol=1e-8) and torch.allclose(sd, qcol.std(0), atol=1e-8)
    ln = load_config()  # noqa: F841  (config import sanity)
    g = torch.ones(768, dtype=F64) * 0.7
    be = torch.linspace(-0.1, 0.1, 768, dtype=F64)
    mw, sw = weights_only_moments(B_row, Wc, g, be)
    M = (spectrum.W_in @ b.Q).T  # column i = (W_in Q)_{i,:}
    tau = g.norm() / np.sqrt(768)
    assert torch.allclose(mw, be @ M, atol=1e-9) and torch.allclose(sw, tau * M.norm(dim=0), atol=1e-9)


def test_full_rank_every_basis_is_original(cfg2, cpu_model, spectrum, pool_x, mlp, small):
    model, _ = cpu_model
    X, Xa = pool_x
    rec, _ = _pair_batches(cpu_model, small)
    with wrapped_mlp(model, 9) as w:
        for fam in _families(spectrum, X):
            b = fam(768)
            _, B_row = b.row_maps()
            mom = calibrated_moments(X, B_row, mlp.c_fc.weight.detach())
            for method, scope in (("zero", "full_layer"), ("gauss", "full_layer"), ("bias", "last_token")):
                out = forward_last(model, w, small["rec"], reduction=_red(cfg2, b, mom, method, scope, mlp))
                assert torch.allclose(out["logits"], rec["logits"], atol=5e-4), (b.id, method, scope)


def test_branch_patch_and_reduced_B_null_all_bases(cfg2, cpu_model, spectrum, pool_x, mlp, small):
    model, _ = cpu_model
    X, Xa = pool_x
    rec, don = _pair_batches(cpu_model, small)
    d = (don["x_pre_last"] - rec["x_pre_last"]).to(F64)
    with wrapped_mlp(model, 9) as w:
        for fam in _families(spectrum, X):
            b = fam(192)
            R_row, B_row = b.row_maps()
            dR, dB = d @ R_row, d @ B_row
            assert torch.allclose(dR + dB, d, atol=1e-9)
            full = forward_last(model, w, small["rec"], delta=(dR + dB).float())
            assert torch.allclose(full["out_last"], don["out_last"], atol=2e-4), b.id
            # semantic check (not a size requirement: a small B response is a valid research outcome):
            # the projected delta and its pre-activation change are nonzero, and the patched branch equals an
            # independent computation of the MLP on x_rec + dB.
            assert float(dB.norm()) > 0 and float((dB @ mlp.c_fc.weight.detach().to(F64)).norm()) > 0, b.id
            o_b = forward_last(model, w, small["rec"], delta=dB.float())
            x_ind = rec["x_pre_last"] + dB.float()
            assert torch.allclose(o_b["x_last"], x_ind, atol=1e-5), b.id
            with torch.no_grad():
                out_ind = mlp.c_proj(mlp.act(mlp.c_fc(x_ind)))
            assert torch.allclose(o_b["out_last"], out_ind, atol=2e-4), b.id
            for pool, method, scope in ((X, "gauss", "full_layer"), (X, "bias", "last_token"),
                                        (Xa, "gauss", "full_layer")):
                mom = calibrated_moments(pool, B_row, mlp.c_fc.weight.detach())
                red = _red(cfg2, b, mom, method, scope, mlp)
                base = forward_last(model, w, small["rec"], reduction=red)
                pat = forward_last(model, w, small["rec"], delta=dB.float(), reduction=red)
                assert (pat["out_last"] - base["out_last"]).abs().max() < 2e-4, (b.id, method, scope)
                assert (pat["logits"] - base["logits"]).abs().max() < 2e-3, (b.id, method, scope)


def test_no_current_x_moment_leak(cfg2, cpu_model, spectrum, pool_x, mlp, small):
    """Moments are fixed tensors from the pool: unchanged by forwards; outputs row-independent of the batch."""
    model, tok = cpu_model
    X, _ = pool_x
    b = wsvd_family(spectrum.W_in, X, 0.01)(96)
    mom = calibrated_moments(X, b.row_maps()[1], mlp.c_fc.weight.detach())
    red = _red(cfg2, b, mom, "gauss", "full_layer", mlp)
    mu0, sd0 = red.mu.clone(), red.sigma.clone()
    rows = [p["rec_ids"] for p in small["rows"]]
    with wrapped_mlp(model, 9) as w:
        full = forward_last(model, w, make_batch(rows, tok.eos_token_id, torch.device("cpu")), reduction=red)
        one = forward_last(model, w, make_batch(rows[2:3], tok.eos_token_id, torch.device("cpu")), reduction=red)
    assert torch.equal(red.mu, mu0) and torch.equal(red.sigma, sd0)
    assert torch.allclose(full["logits"][2], one["logits"][0], atol=1e-3)


def test_day2_quadrature_converges_over_bound(cfg2):
    n, bound = cfg2["quadrature_nodes"], cfg2["quadrature_sigma_max_checked"]
    m = torch.linspace(-20, 20, 401, dtype=F64).reshape(-1, 1)
    s = torch.linspace(0, bound, 21, dtype=F64).reshape(1, -1)
    ref = gauss_expect_gelu(m, s, cfg2["quadrature_rule"]["ref_nodes"])
    assert float((gauss_expect_gelu(m, s, n) - ref).abs().max()) < cfg2["quadrature_rule"]["tol_max_abs"]


# ------------------------------------------------------------------ matching / contrasts

MC = {"abs_floor": 1e-8, "rel_floor": 1e-6, "oos_range": [0.25, 4.0], "assert_rel_tol_f64": 1e-9,
      "assert_rel_tol_f32": 1e-5}


def test_match_scale_rules():
    native = np.array([1.0, 0.0, 1e-9, 2.0, 1.0, 1.0, 1e-7])
    target = np.array([2.0, 1.0, 1.0, 0.4, 10.0, 0.0, 1.0])
    scale, deg, oos = match_scale(native, target, MC)
    assert deg.tolist() == [False, True, True, False, False, True, True]
    assert scale[0] == 2.0 and np.isnan(scale[1]) and scale[3] == 0.2
    assert oos.tolist() == [False, False, False, True, True, False, False]
    with pytest.raises(ValueError):
        match_scale(np.array([np.nan]), np.array([1.0]), MC)


def test_apply_match_l2_and_preact(mlp):
    g = torch.Generator().manual_seed(0)
    d = torch.randn(10, 768, generator=g, dtype=F64)
    d[3] = 0.0
    tgt = np.linspace(0.5, 3.0, 10)
    out, info = apply_match(d, "l2", tgt, MC)
    ok = ~info["degenerate"]
    assert info["degenerate"][3] and (out[3] == 0).all()
    assert np.allclose(out.to(F64).norm(dim=-1).numpy()[ok], tgt[ok], rtol=1e-5)
    Wc = mlp.c_fc.weight.detach()
    out, info = apply_match(d, "preact", tgt * 10, MC, Wc)
    assert np.allclose(preact_norm(out, Wc).numpy()[ok], (tgt * 10)[ok], rtol=1e-5)
    # l2 and preact matching give different deltas (one scalar cannot match both)
    out_l2, _ = apply_match(d, "l2", tgt, MC)
    assert not np.allclose(preact_norm(out_l2, Wc).numpy()[ok], tgt[ok], rtol=1e-3)


def test_paired_contrast_common_mask():
    pids = ["a", "b", "c", "d"]
    tm = ["t1", "t1", "t2", "t2"]
    lhs = {"a": dict(T=1.0, degenerate=0, oos=0), "b": dict(T=2.0, degenerate=1, oos=0),
           "c": dict(T=3.0, degenerate=0, oos=0), "d": dict(T=4.0, degenerate=0, oos=0)}
    rhs = {"a": dict(T=0.5, degenerate=0, oos=0), "b": dict(T=0.5, degenerate=0, oos=0),
           "c": dict(T=1.0, degenerate=0, oos=1), "d": dict(T=np.nan, degenerate=1, oos=0)}
    out = paired_contrast(lhs, rhs, pids, tm, 50, 0)
    assert out["all"]["n"] == 2 and np.isclose(out["all"]["diff_mean_ci"][0], (0.5 + 2.0) / 2)
    assert out["excl_oos"]["n"] == 1 and np.isclose(out["excl_oos"]["diff_mean_ci"][0], 0.5)


def test_indicator_alignment():
    r, d = np.array([20, 50]), np.array([40, 30])
    assert np.allclose(indicator_alignment(r, d, r), 0.0)
    assert np.allclose(indicator_alignment(r, d, d), 1.0)


# ------------------------------------------------------------------ parity / CLI

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_cpu_parity_whitened(cfg2, cpu_model, spectrum, pool_x, small):
    X, _ = pool_x
    b = wsvd_family(spectrum.W_in, X, 0.01)(192)
    model_m, tok = load_model(cfg2, torch.device("mps"))
    rows = [p["rec_ids"] for p in small["rows"]]
    res = {}
    for name, model, dev in (("cpu", cpu_model[0], torch.device("cpu")), ("mps", model_m, torch.device("mps"))):
        m = model.transformer.h[9].mlp
        mom = calibrated_moments(X, b.row_maps()[1], m.c_fc.weight.detach().cpu())
        red = _red(cfg2, b, mom, "gauss", "full_layer", m, dev)
        with wrapped_mlp(model, 9) as w:
            res[name] = forward_last(model, w, make_batch(rows, tok.eos_token_id, dev), reduction=red)["logits"].cpu()
    assert (res["cpu"] - res["mps"]).abs().max() < 2e-3


def _cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "run_day2.py"), *args], capture_output=True, text=True, cwd=ROOT)


def test_day2_cli_guards():
    for split in ("validation", "test"):
        p = _cli("run", "--split", split)
        assert p.returncode != 0 and "calibration split only" in p.stderr
    p = _cli("run", "--layer", "10")
    assert p.returncode != 0 and "not allowed" in p.stderr
