"""Follow-up propagation audit: numerics, hook scope, probes, splits/leakage, ridge, cluster bootstrap, locks.

No test runs a follow-up fit/validation/test model forward; model tests use a tiny random GPT-2 (CPU float64)."""
import json
import math
import types

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from followup import ROOT, data, pipeline
from followup.common import bound_hashes, load_config, write_once
from followup.downstream import (_logits_last, centred_probes, directional, exact_jvp, leaf_logits, leaf_mlp,
                                 probe_vjps, rademacher_probes)
from followup.features import feature_rows, label_norm, response_norm, scale_from_fit, shuffle_perm
from followup.local import LocalMLP, exact_delta_b, gelu_new_prime, taylor_d1
from followup.ridge import ridge_fit, ridge_predict, select_lambda
from followup.stats import cluster_tables, decision, gain, multiplicity_draws, weighted_mse
from scp.data import make_batch
from scp.model import Reduction, forward_last, wrapped_mlp
from scp.spectral import gelu_new

F64 = torch.float64
LAYER = 1


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def tiny():
    c = GPT2Config(vocab_size=61, n_positions=32, n_embd=24, n_layer=3, n_head=4, attn_implementation="eager",
                   bos_token_id=0, eos_token_id=0)
    torch.manual_seed(0)
    m = GPT2LMHeadModel(c).to(F64).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def tiny_batch(lengths=(7, 5, 9), seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = [torch.randint(1, 61, (n,), generator=g).tolist() for n in lengths]
    return make_batch(ids, 0, torch.device("cpu"))


def rand_projector(d, k, seed):
    g = torch.Generator().manual_seed(seed)
    Qm, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=F64))
    return Qm[:, :k] @ Qm[:, :k].T


def tiny_local(tiny, k=10, seed=1):
    mlp = tiny.transformer.h[LAYER].mlp
    P = rand_projector(24, k, seed)
    g = torch.Generator().manual_seed(seed + 1)
    m = torch.randn(24, generator=g, dtype=F64)
    Wc = mlp.c_fc.weight.detach().to(F64)
    mu = (m @ (torch.eye(24, dtype=F64) - P).T) @ Wc  # = m (I-P) Wc, as the calibrated mean would be
    return LocalMLP.from_mlp(mlp, P, mu), m


# ------------------------------------------------------------------ local numerics

def test_gelu_new_prime_matches_finite_difference():
    x = torch.linspace(-8, 8, 4001, dtype=F64)
    h = 1e-5
    fd = (gelu_new(x + h) - gelu_new(x - h)) / (2 * h)
    assert float((gelu_new_prime(x) - fd).abs().max()) < 1e-9
    assert float(gelu_new_prime(torch.tensor([0.0], dtype=F64))) == pytest.approx(0.5)


def test_d1_has_orig_minus_red_sign_and_second_order_error():
    g = torch.Generator().manual_seed(3)
    m, d = 64, 16
    u = torch.randn(4, m, generator=g, dtype=F64)
    aR = 0.7 * torch.randn(4, m, generator=g, dtype=F64)
    q0 = torch.randn(4, m, generator=g, dtype=F64)
    Wout = torch.randn(m, d, generator=g, dtype=F64) / 8
    zero = torch.zeros_like(u)
    errs = []
    for t in (4e-2, 2e-2, 1e-2):
        db = exact_delta_b(u, aR, t * q0, zero, Wout)
        D1 = taylor_d1(u, aR, t * q0, zero, Wout)
        cos = float((D1 * -db).sum() / (D1.norm() * db.norm()))
        assert cos > 0.99  # D1 ~ -delta_b  (orig - red)
        errs.append(float((db + D1).norm()))
    r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
    assert 3.6 < r1 < 4.4 and 3.6 < r2 < 4.4  # O(q^2)


def test_local_b_matches_reduction_module_and_retained_only_geometry(tiny):
    local, m = tiny_local(tiny)
    mlp = tiny.transformer.h[LAYER].mlp
    red = Reduction(P_R=local.R_row, Wc=mlp.c_fc.weight, b=mlp.c_fc.bias, mu=local.mu, sigma=torch.zeros_like(local.mu),
                    method="bias", scope="last_token", n_nodes=8, sigma_bound=1.0)
    g = torch.Generator().manual_seed(5)
    x_r, x_a = torch.randn(6, 24, generator=g, dtype=F64), torch.randn(6, 24, generator=g, dtype=F64)
    b_mod = mlp.c_proj(red.hidden(x_r)) - mlp.c_proj(gelu_new(mlp.c_fc(x_r)))
    assert float((b_mod - local.b(x_r)).abs().max()) < 1e-12
    m_B, res = local.discarded_mean_input()
    assert res < 1e-12 and float((m_B - m @ local.Q_row).abs().max()) < 1e-10
    for alpha in (0.5, 1.0, 1.5):
        lq = local.quantities(x_r, x_a, alpha, m_B)
        xp = lq["x_prime"]
        # retained part moves by alpha * P(x_a - x_r); discarded part, q and the input reconstruction error are fixed
        assert float((xp @ local.R_row - (x_r @ local.R_row + alpha * ((x_a - x_r) @ local.R_row))).abs().max()) < 1e-12
        assert float((xp @ local.Q_row - x_r @ local.Q_row).abs().max()) < 1e-12
        assert float(lq["a_B"].abs().max()) < 1e-12
        assert float((local.preacts(xp)[2] - lq["q"]).abs().max()) < 1e-12
        assert float((lq["disc_input"] - (x_r - m) @ local.Q_row).abs().max()) < 1e-12
        assert float((lq["delta_b"] - (local.b(xp) - local.b(x_r))).abs().max()) < 1e-13


def test_zero_and_self_intervention_give_exact_zero():
    local = LocalMLP(*(torch.randn(*s, dtype=F64) for s in ((8, 20), (20,), (20, 8))), rand_projector(8, 3, 0),
                     torch.randn(20, dtype=F64))
    x = torch.randn(5, 8, dtype=F64)
    m_B, _ = local.discarded_mean_input()
    for lq in (local.quantities(x, x + 1.0, 0.0, m_B), local.quantities(x, x, 1.0, m_B)):
        assert float(lq["delta_b"].abs().max()) == 0.0 and float(lq["D1"].abs().max()) == 0.0
        G = torch.randn(5, 8, 8, dtype=F64)
        assert float(directional(G, lq["delta_b"]).max()) == 0.0


# ------------------------------------------------------------------ downstream estimator

def test_synthetic_linear_downstream_predicted_norm_vs_true():
    g = torch.Generator().manual_seed(11)
    V, d = 300, 12
    J = torch.randn(V, d, generator=g, dtype=F64)
    C = torch.eye(V, dtype=F64) - 1.0 / V
    delta = torch.randn(3, d, generator=g, dtype=F64)
    true = (delta @ (C @ J).T).norm(dim=-1)  # linear downstream: the label equals ||C J delta_b|| exactly
    U = rademacher_probes(20000, V, 7)
    G = (centred_probes(U) @ J).unsqueeze(0).expand(3, -1, -1)
    assert float((directional(G, delta) / true - 1).abs().max()) < 0.03
    # 8 probes: unbiased for the squared norm (average over many fixed probe sets)
    est2 = torch.stack([directional((centred_probes(rademacher_probes(8, V, s)) @ J).expand(3, -1, -1), delta) ** 2
                        for s in range(400)]).mean(0)
    assert float((est2 / true ** 2 - 1).abs().max()) < 0.1


def graph_logits(model, b1, m):
    """Differentiable single-row leaf logits (reference Jacobian only)."""
    with leaf_mlp(model, LAYER) as lm, torch.enable_grad():
        lm.last_idx, lm.leaf = b1.last_idx, m.unsqueeze(0)
        return _logits_last(model, b1)[0]


def test_probe_vjps_equal_full_jacobian_and_exact_jvp(tiny):
    b = tiny_batch()
    U = rademacher_probes(8, 61, 2)
    Uc = centred_probes(U)
    G, m0, logits0, ld = probe_vjps(tiny, LAYER, b, Uc)
    assert ld == 0.0
    with wrapped_mlp(tiny, LAYER) as w:
        o = forward_last(tiny, w, b)
    assert float((m0 - o["out_last"]).abs().max()) == 0.0 and float((logits0 - o["logits"]).abs().max()) == 0.0
    C = torch.eye(61, dtype=F64) - 1.0 / 61
    v = torch.randn(3, 24, dtype=F64)
    jv = exact_jvp(tiny, LAYER, b, m0, v)
    for i in range(3):
        bi = b.slice(slice(i, i + 1))
        f = lambda m: leaf_logits(tiny, LAYER, bi, m.unsqueeze(0))[0]  # noqa: E731
        J = torch.autograd.functional.jacobian(lambda m: graph_logits(tiny, bi, m), m0[i])  # [V, d]
        assert float((G[i] - U @ C @ J).abs().max()) < 1e-10
        assert float((jv[i] - C @ J @ v[i]).abs().max()) < 1e-10
        # central FD at two step sizes: small relative error, and second-order (error ratio ~4 between 2h and h)
        an = G[i] @ v[i]
        err = []
        for eps in (2e-5, 1e-5):
            fd = (f(m0[i] + eps * v[i]) - f(m0[i] - eps * v[i])) @ Uc.T / (2 * eps)
            err.append(float((fd - an).norm() / an.norm()))
        assert err[1] < 1e-6 and 3.5 < err[0] / err[1] < 4.5
    assert all(p.grad is None for p in tiny.parameters())


def test_probes_fixed_and_reused_across_recipients(tiny):
    U1, U2 = rademacher_probes(8, 61, 5), rademacher_probes(8, 61, 5)
    assert torch.equal(U1, U2) and set(U1.unique().tolist()) == {-1.0, 1.0}
    assert not torch.equal(U1, rademacher_probes(8, 61, 6))
    b = tiny_batch()
    Uc = centred_probes(U1)
    G, *_ = probe_vjps(tiny, LAYER, b, Uc)
    for i in range(3):  # per-recipient computation with the same probes reproduces the batched rows
        Gi, *_ = probe_vjps(tiny, LAYER, b.slice(slice(i, i + 1)), Uc)
        assert float((Gi[0] - G[i]).abs().max()) < 1e-12


def test_vjp_guards(tiny):
    b = tiny_batch()
    Uc = centred_probes(rademacher_probes(2, 61, 0))
    tiny.train()
    try:
        with pytest.raises(RuntimeError, match="eval"):
            probe_vjps(tiny, LAYER, b, Uc)
    finally:
        tiny.eval()
    with wrapped_mlp(tiny, LAYER):
        with pytest.raises(RuntimeError, match="nesting"):
            with leaf_mlp(tiny, LAYER):
                pass
    tiny.transformer.h[0].mlp.c_fc.weight.requires_grad_(True)
    try:
        with pytest.raises(RuntimeError, match="frozen"):
            probe_vjps(tiny, LAYER, b, Uc)
    finally:
        tiny.transformer.h[0].mlp.c_fc.weight.requires_grad_(False)


def test_hook_patches_last_token_mlp_input_only_and_keeps_bypass(tiny):
    b = tiny_batch()
    block = tiny.transformer.h[LAYER]
    cap = {}
    h1 = block.ln_2.register_forward_hook(lambda mod, i, o: cap.__setitem__("resid_mid", i[0].clone()))
    h2 = block.register_forward_hook(lambda mod, i, o: cap.__setitem__("out", o[0].clone() if isinstance(o, tuple) else o.clone()))
    try:
        with wrapped_mlp(tiny, LAYER) as w:
            base = forward_last(tiny, w, b)
            rm0, out0 = cap["resid_mid"], cap["out"]
            delta = torch.randn(3, 24, dtype=F64)
            pat = forward_last(tiny, w, b, delta=delta)
            rm1, out1 = cap["resid_mid"], cap["out"]
            zero = forward_last(tiny, w, b, delta=torch.zeros(3, 24, dtype=F64))
    finally:
        h1.remove()
        h2.remove()
    ar, li = torch.arange(3), b.last_idx
    assert torch.equal(rm0, rm1)  # residual bypass untouched
    mlp = block.mlp
    x_last = base["x_pre_last"] + delta
    assert float((out1[ar, li] - (rm1[ar, li] + mlp(x_last.unsqueeze(1))[:, 0])).abs().max()) < 1e-12
    other = torch.arange(out0.shape[1])[None, :] != li[:, None]
    assert torch.equal(out0[other], out1[other])
    pos = torch.arange(out0.shape[1])[None, :] < li[:, None]
    assert torch.equal(base["hidden_all"][pos], pat["hidden_all"][pos])
    assert torch.equal(zero["logits"], base["logits"])


# ------------------------------------------------------------------ features / scale / labels

def test_feature_rows_definitions(cfg, tiny):
    local, _ = tiny_local(tiny)
    m_B, _ = local.discarded_mean_input()
    g = torch.Generator().manual_seed(9)
    N = 4
    rows = [{"rid": f"x-{i}", "split": "fit", "frame": "N0", "noun": f"n{i}", "template": f"N0|n{i}",
             "direction": "up", "diff_bin": "10-19", "r": 10, "d": 25} for i in range(N)]
    unp = {"x_r": torch.randn(N, 24, generator=g, dtype=F64), "x_a": torch.randn(N, 24, generator=g, dtype=F64),
           "G": torch.randn(N, 8, 24, generator=g, dtype=F64), "kl": torch.rand(N, generator=g, dtype=F64)}
    s = 2.5
    perm = shuffle_perm(24, 1)
    out = feature_rows(cfg, local, m_B, rows, unp, s, perm)
    assert len(out) == N * len(cfg["intervention"]["alphas"])
    for f in out:
        assert set(cfg["features"]["baseline"]) <= set(f) and {"d_exact", "d_taylor"} <= set(f)
        assert f["preact_norm_product"] == pytest.approx(f["disc_preact_norm"] * f["preact_delta_norm"], rel=1e-12)
        assert f["b_iso"] == pytest.approx(f["jnorm"] * f["delta_b_norm"] / (s * math.sqrt(24)), rel=1e-12)
    i = 1
    lq = local.quantities(unp["x_r"], unp["x_a"], 1.0, m_B)
    f = [r for r in out if r["rid"] == "x-1" and r["alpha"] == 1.0][0]
    gd = unp["G"][i] @ lq["delta_b"][i]
    assert f["d_exact"] == pytest.approx(float(gd.pow(2).mean().sqrt()) / s, rel=1e-12)
    gs = unp["G"][i] @ lq["delta_b"][i][perm]
    assert f["d_shuf"] == pytest.approx(float(gs.pow(2).mean().sqrt()) / s, rel=1e-12)
    with pytest.raises(ValueError, match="floor"):
        feature_rows(cfg, local, m_B, rows, unp, 0.0, perm)


def test_label_norm_is_centred_and_scale_uses_fit_only():
    a = torch.randn(2, 50, dtype=F64)
    assert float(label_norm(a + 3.0, a, a, a).abs().max()) < 1e-12  # constant logit shift is not a response
    assert float((response_norm(a + 1.0, a)).abs().max()) < 1e-12
    rows = [{"split": "fit", "alpha": al, "orig_response_norm": v} for al, v in ((1.0, 3.0), (1.0, 4.0), (0.5, 9.0))]
    assert scale_from_fit(rows) == pytest.approx(math.sqrt(12.5))
    with pytest.raises(ValueError, match="fit rows only"):
        scale_from_fit(rows + [{"split": "validation", "alpha": 1.0, "orig_response_norm": 1.0}])
    with pytest.raises(ValueError, match="fit rows only"):
        scale_from_fit([{**r, "split": "test"} for r in rows])


def test_test_labels_are_locked_outside_evaluate():
    fake = types.SimpleNamespace(labels_allowed=False)
    with pytest.raises(pipeline.LabelLock):
        pipeline.Ctx.labels(fake, [], {}, "test")


# ------------------------------------------------------------------ splits / leakage

def test_real_manifests_structure_and_disjointness(cfg):
    asg_now = data.assign(cfg)
    rows_by = {}
    for s in data.SPLITS:
        rows_by[s], index = data.load_split(s)
        assert json.loads(json.dumps(asg_now)) == index["assignment"]
        info = data.structural_checks(cfg, s, rows_by[s], {k: {kk: [tuple(t) for t in vv] if k == "templates" else vv
                                                              for kk, vv in v.items()} if isinstance(v, dict) else v
                                                          for k, v in asg_now.items()})
        assert info["n_recipients"] == cfg["design"]["expected_recipients"][s]
    data.cross_split_checks(asg_now, rows_by)
    assert len({p["template"] for p in rows_by["test"]}) >= 24
    assert not {p["noun"] for s in rows_by for p in rows_by[s]} & data.historical_nouns(cfg)
    hist, _ = data.historical_texts(cfg)
    for s in data.SPLITS:
        assert data.leakage(rows_by[s], hist) == []
        # donors never leave their template cluster (shared donor rows travel with the template)
        owner = {}
        for p in rows_by[s]:
            assert p["don_text"] == data.prompt(cfg, p["frame"], p["noun"], p["d"])
            owner.setdefault(p["don_text"], set()).add(p["template"])
        assert all(len(v) == 1 for v in owner.values())


def test_leakage_check_flags_historical_prompt(cfg):
    hist, files = data.historical_texts(cfg)
    assert "manifests_day3/test_pairs.csv" in files and "manifests/moment_pool_calibration.csv" in files
    old = next(iter(hist))
    rows = [{"rec_text": "fresh prompt", "don_text": old}]
    assert data.leakage(rows, hist) == [old]


def test_cells_and_gap_rule(cfg):
    assert data.in_cell(10, 20, "up", "10-19") and not data.in_cell(10, 20, "down", "10-19")
    assert data.in_cell(50, 10, "down", "30-40") and not data.in_cell(50, 9, "down", "30-40")
    assert len(data.cells(cfg)) == 6


def test_prepare_refuses_existing_manifests(cfg):
    with pytest.raises(FileExistsError):
        data.prepare(None, cfg)


# ------------------------------------------------------------------ ridge

def test_ridge_standardized_fit_predict_and_lambda_tie():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)) * [1, 10, 100] + [0, 5, -3]
    y = 2.0 + X @ [0.5, -0.1, 0.01]
    m = ridge_fit(X, y, 1e-9)
    assert np.allclose(ridge_predict(m, X, clamp_min=-np.inf), y, atol=1e-6)
    assert (ridge_predict(m, X - 1e4) >= 0).all()  # clamped at zero
    with pytest.raises(ValueError, match="degenerate"):
        ridge_fit(np.c_[X, np.ones(200)], y, 1.0)
    assert select_lambda({0.1: 1.0, 1.0: 1.0, 10.0: 2.0}, 1e-12) == 1.0
    assert select_lambda({0.1: 0.5, 1.0: 1.0, 10.0: 2.0}, 1e-12) == 0.1
    assert select_lambda({0.1: 1.0, 1.0: 1.0, 10.0: 1.0}, 1e-12) == 10.0


# ------------------------------------------------------------------ cluster bootstrap / G / decision

def synth_rows(cfg, frames=("N1", "N2"), n_t=3, seed=0):
    rows = []
    for f in frames:
        for t in range(n_t):
            for direction, gb in data.cells(cfg):
                for a in cfg["intervention"]["alphas"]:
                    rows.append({"frame": f, "template": f"{f}|n{t}", "direction": direction, "diff_bin": gb,
                                 "alpha": a})
    return rows


def brute_mse(cfg, rows, err2, mult):
    """One bootstrap draw by explicit row expansion: each resampled template brings ALL its rows."""
    exp = [(r, e) for r, e in zip(rows, err2) for _ in range(int(mult[r["template"]]))]
    keys = sorted({tuple(r[d] for d in cfg["evaluation"]["cell_dims"]) for r in rows})
    means = [np.mean([e for r, e in exp if tuple(r[d] for d in cfg["evaluation"]["cell_dims"]) == k]) for k in keys]
    return float(np.mean(means))


def test_cluster_bootstrap_fixed_weights_and_rows_travel_together(cfg):
    rows = synth_rows(cfg)
    rng = np.random.default_rng(1)
    e = rng.random(len(rows))
    assert weighted_mse(cfg, rows, e) == pytest.approx(brute_mse(cfg, rows, e, {r["template"]: 1 for r in rows}))
    templates, frames, N, (S,), w = cluster_tables(cfg, rows, [e])
    M = multiplicity_draws(frames, 50, 3)
    for f in set(frames):  # templates resampled within frame only, fixed frame sizes
        assert (M[:, frames == f].sum(1) == (frames == f).sum()).all()
    for b in range(5):
        mult = dict(zip(templates, M[b]))
        vec = float((((M[b:b + 1] @ S) / (M[b:b + 1] @ N)) @ w)[0])
        assert vec == pytest.approx(brute_mse(cfg, rows, e, mult), rel=1e-12)


def test_gain_reports_negative_and_decisions(cfg):
    rows = synth_rows(cfg)
    rng = np.random.default_rng(2)
    eB = rng.random(len(rows)) + 0.5
    res = gain(cfg, rows, 2 * eB, eB, B=500, seed=0)  # A worse than B
    assert res["ratio_valid"] and res["G"] == pytest.approx(-1.0) and res["G_upper_one_sided"] < 0
    assert decision(cfg, res) == "TARGET_GAIN_RULED_OUT"
    res = gain(cfg, rows, 0.5 * eB, eB, B=500, seed=0)
    assert res["G"] == pytest.approx(0.5) and decision(cfg, res) == "GO"
    eA = eB * np.where(np.arange(len(rows)) % 2, 0.6, 1.3)
    res = gain(cfg, rows, eA, eB, B=500, seed=0)
    assert res["G_lower_one_sided"] <= res["G"] <= res["G_upper_one_sided"]
    assert decision(cfg, {"ratio_valid": True, "G_lower_one_sided": 0.05, "G_upper_one_sided": 0.3}) == "INCONCLUSIVE"


def test_gain_denominator_floor_invalidates_without_discarding_draws(cfg):
    rows = synth_rows(cfg)
    eA = np.ones(len(rows))
    res = gain(cfg, rows, eA, np.zeros(len(rows)), B=200, seed=0)
    assert not res["ratio_valid"] and res["G"] is None and decision(cfg, res) == "EVALUATION_INVALID"
    assert res["mse_A"] == 1.0 and res["mse_B"] == 0.0 and res["mse_B_minus_A"] == -1.0
    # nonzero baseline error in ONE template only: some draws drop it -> denominator 0 in those draws
    eB = np.array([1.0 if r["template"] == "N1|n0" else 0.0 for r in rows])
    res = gain(cfg, rows, eA, eB, B=200, seed=0)
    assert res["n_draws_denominator_invalid"] > 0 and not res["ratio_valid"]
    assert res["G_lower_one_sided"] is None and decision(cfg, res) == "EVALUATION_INVALID"
    assert res["G"] is not None  # the point value is still reported


def test_incomplete_design_is_rejected(cfg):
    rows = synth_rows(cfg)
    last = rows[-1]
    gone = [r for r in rows if not (r["frame"] == last["frame"] and r["direction"] == last["direction"]
                                    and r["diff_bin"] == last["diff_bin"] and r["alpha"] == last["alpha"])]
    with pytest.raises(ValueError, match="incomplete"):
        weighted_mse(cfg, gone, np.ones(len(gone)))
    one_missing = rows[:-1]  # the cell still exists, but one template lacks it -> cluster resampling invalid
    with pytest.raises(ValueError, match="every template"):
        gain(cfg, one_missing, np.ones(len(one_missing)), np.ones(len(one_missing)), B=10, seed=0)


# ------------------------------------------------------------------ CLI / locks / hashes

def test_cli_requires_run_arguments():
    import run_followup
    for argv in (["validate"], ["freeze", "--fit-run", "x"], ["register"], ["report"]):
        with pytest.raises(SystemExit):
            run_followup.main(argv)


def test_stages_refuse_without_registration(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "REGISTRATION", tmp_path / "none.json")
    for fn in (pipeline.fit, pipeline.predict_test, pipeline.evaluate_test):
        with pytest.raises(RuntimeError, match="registration"):
            fn(cfg)
    with pytest.raises(RuntimeError, match="registration"):
        pipeline.validate(cfg, "x")


def test_verify_registration_detects_any_bound_change(cfg, monkeypatch, tmp_path):
    reg_path = tmp_path / "reg.json"
    good = {"bound": bound_hashes(cfg)}
    reg_path.write_text(json.dumps(good))
    monkeypatch.setattr(pipeline, "REGISTRATION", reg_path)
    final = {**cfg, "protocol_version": "followup-v1"}
    assert pipeline.verify_registration(final)["bound"] == good["bound"]
    with pytest.raises(RuntimeError, match="DRAFT"):
        pipeline.verify_registration(cfg)
    bad = json.loads(json.dumps(good))
    bad["bound"]["source"]["tree_sha256"] = "0" * 64
    reg_path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="source"):
        pipeline.verify_registration(final)


def test_register_refuses_draft_and_existing(cfg, monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="protocol_version"):
        monkeypatch.setattr(pipeline, "REGISTRATION", tmp_path / "r.json")
        pipeline.register(cfg, "x")
    (tmp_path / "r.json").write_text("{}")
    with pytest.raises(FileExistsError):
        pipeline.register({**cfg, "protocol_version": "followup-v1"}, "x")
    with pytest.raises(FileExistsError):
        write_once(tmp_path / "r.json", {})


def test_predictions_lock_precedes_labels(cfg, monkeypatch, tmp_path):
    """evaluate-test refuses without the lock, and if a locked file changed; predict-test refuses after a
    failed validation gate or an existing lock."""
    monkeypatch.setattr(pipeline, "verify_registration", lambda c: {"bound": {}})
    monkeypatch.setattr(pipeline, "load_frozen", lambda c, r: {"proceed_to_test": True})
    monkeypatch.setattr(pipeline, "TEST_LOCK", tmp_path / "lock.json")
    monkeypatch.setattr(pipeline, "TEST_EVALUATED", tmp_path / "done.json")
    monkeypatch.setattr(pipeline, "FROZEN", tmp_path / "frozen.json")
    (tmp_path / "frozen.json").write_text("{}")
    with pytest.raises(RuntimeError, match="lock"):
        pipeline.evaluate_test(cfg)
    run = tmp_path / "pred"
    run.mkdir()
    for n in ("test_features.csv", "test_predictions.csv", "test_probe_vjps.pt"):
        (run / n).write_text(n)
    from scp.common import sha256_file
    lock = {"run_id": "pred", "test_features_sha256": sha256_file(run / "test_features.csv"),
            "test_predictions_sha256": "f" * 64, "test_probe_vjps_sha256": sha256_file(run / "test_probe_vjps.pt"),
            "frozen_sha256": sha256_file(tmp_path / "frozen.json")}
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    monkeypatch.setattr(pipeline, "load_run", lambda rid, stage: (run, {"bound": {}, "run_id": rid}))
    with pytest.raises(RuntimeError, match="test_predictions.csv changed"):
        pipeline.evaluate_test(cfg)
    with pytest.raises(FileExistsError):
        pipeline.predict_test(cfg)
    monkeypatch.setattr(pipeline, "load_frozen", lambda c, r: {"proceed_to_test": False})
    with pytest.raises(RuntimeError, match="gate"):
        pipeline.predict_test(cfg)
    (tmp_path / "done.json").write_text("{}")
    monkeypatch.setattr(pipeline, "load_frozen", lambda c, r: {"proceed_to_test": True})
    with pytest.raises(FileExistsError):
        pipeline.evaluate_test(cfg)


def test_followup_files_do_not_touch_frozen_day3_tree():
    from scp.common import source_hashes
    reg = json.loads((ROOT / "registration" / "DAY3_REGISTRATION.json").read_text())
    assert source_hashes()["tree_sha256"] == reg["bindings"]["source_tree"]
    assert not any(k.startswith(("followup/", "tests_followup/")) or k == "run_followup.py"
                   for k in source_hashes()["files"])
