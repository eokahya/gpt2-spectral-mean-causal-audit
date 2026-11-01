"""Discarded-state response-floor gate: synthetic/unit tests.

No GPT-2 forward pass and no development outcome: model tests use a tiny random GPT-2 (CPU). The frozen artifact
and safetensors weights are only read from disk."""
import json
import math
import types

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from discarded_response_floor_gate import ROOT, data, pipeline
from discarded_response_floor_gate.aggregate import aggregate
from discarded_response_floor_gate.bootstrap import (confirm_decision, percentile_bounds, resample_theta, rng_for,
                                                     template_bootstrap)
from discarded_response_floor_gate.common import (GROOT, PROTECTED, load_config, manifest_dir, registration_path,
                                                  verify_protected, write_once)
from discarded_response_floor_gate.estimators import (centred_J, frame_moments, norm2_m, scale_s2, template_T,
                                                      u_pairs, u_variance)
from discarded_response_floor_gate.geometry import hybrids, load_projector, projector_checks, support
from discarded_response_floor_gate.hook import branch_mlp, forward
from discarded_response_floor_gate.power import (SCENARIOS, clopper_pearson, decide, gamma_params, guard, run_power,
                                                  scenario_moments, scenario_name, simulate, simulate_dataset,
                                                  validate_rows)
from scp.data import make_batch
from scp.model import wrapped_mlp

F64 = torch.float64
LAYER = 1
needs_manifests = pytest.mark.skipif(not (manifest_dir(load_config()) / "index.json").exists(),
                                     reason="run `prepare` first")
CFG_V1 = GROOT / "config" / "gate_dev_v1.json"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def P(cfg):
    return load_projector(cfg)[0]


@pytest.fixture(scope="module")
def tiny():
    c = GPT2Config(vocab_size=61, n_positions=32, n_embd=24, n_layer=3, n_head=4, attn_implementation="eager",
                   bos_token_id=0, eos_token_id=0)
    torch.manual_seed(0)
    m = GPT2LMHeadModel(c).to(F64).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def tiny_batch(n=3, length=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = [torch.randint(1, 61, (length - i,), generator=g).tolist() for i in range(n)]
    return make_batch(ids, 0, torch.device("cpu"))


# ------------------------------------------------------------------ 1. projector and orientation

def test_frozen_projector_is_symmetric_idempotent_rank192(cfg, P):
    chk = projector_checks(P, 192, cfg["numerics"])
    assert chk["sym_max_abs"] == 0.0 and chk["idem_max_abs"] < 1e-12 and chk["trace"] == pytest.approx(192, abs=1e-9)
    assert chk["n_eig_above_split"] == 192
    x = torch.randn(5, 768, dtype=F64)
    assert float(((P @ x.T).T - x @ P.T).abs().max()) < 1e-12  # column map == row map
    bad = P.clone()
    bad[0, 1] += 1e-6
    with pytest.raises(ValueError):
        projector_checks(bad, 192, cfg["numerics"])


def test_weight_orientation_from_safetensors(cfg):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    path = hf_hub_download(cfg["model"]["name"], "model.safetensors", revision=cfg["model"]["revision"])
    with safe_open(path, "pt") as f:
        Wc = f.get_tensor(f"h.{cfg['layer']}.mlp.c_fc.weight")
    assert tuple(Wc.shape) == (768, 3072)  # Conv1D: x[768] @ Wc -> preactivation[3072]


# ------------------------------------------------------------------ 2. hybrid algebra

def test_hybrid_equals_centred_form_for_any_mean_and_controls(P):
    g = torch.Generator().manual_seed(1)
    ev, U = torch.linalg.eigh(P)
    V = U[:, -192:]
    x_rec, x_a = torch.randn(768, generator=g, dtype=F64), torch.randn(768, generator=g, dtype=F64)
    x_don = torch.randn(8, 768, generator=g, dtype=F64)
    xm, xp = hybrids(P, x_rec, x_don, x_a)
    I = torch.eye(768, dtype=F64)
    for mu in (torch.zeros(768, dtype=F64), 10 * torch.randn(768, generator=g, dtype=F64)):
        z = V.T @ (x_rec - mu)
        d = V.T @ (x_a - x_rec)
        q = (x_don - mu) @ (I - P).T
        assert float((mu + V @ z + q - xm).abs().max()) < 1e-10
        assert float((mu + V @ (z + d) + q - xp).abs().max()) < 1e-10
    own_m, own_p = hybrids(P, x_rec, x_rec[None], x_a)
    assert float((own_m[0] - x_rec).abs().max()) < 1e-12  # own q and own z -> the original input
    d0_m, d0_p = hybrids(P, x_rec, x_don, x_rec)
    assert torch.equal(d0_m, d0_p)  # d = 0 -> identical branch inputs
    # retained part is the recipient's, discarded part is the donor's
    assert float((xm @ P.T - x_rec @ P.T).abs().max()) < 1e-10
    assert float((xm @ (I - P).T - x_don @ (I - P).T).abs().max()) < 1e-10
    sp = support(P, x_rec, x_don, x_a, xm, xp, torch.randn(768, 30, dtype=F64), torch.zeros(30, dtype=F64))
    assert all(v.shape == (8,) for v in sp.values())


# ------------------------------------------------------------------ 3. estimators and synthetic controls

def test_u_two_ways_denominator_8_and_identical_q():
    g = torch.Generator().manual_seed(2)
    R = torch.randn(8, 100, generator=g, dtype=F64)
    u1, u2 = u_variance(R), u_pairs(R)
    assert u1 == pytest.approx(u2, rel=1e-13)
    pop_var = float(R.var(0, unbiased=False).sum()) / 100
    assert u1 == pytest.approx(pop_var, rel=1e-13)  # denominator 8 (population), not 7
    assert u1 != pytest.approx(float(R.var(0, unbiased=True).sum()) / 100, rel=1e-6)
    assert u_variance(R[:1].expand(8, -1)) < 1e-30 and u_pairs(R[:1].expand(8, -1)) == 0.0


def test_centring_and_dimension_normalisation():
    l = torch.randn(3, 500, dtype=F64)
    J = list(range(100, 200))
    c = centred_J(l, J)
    assert float(c.mean(-1).abs().max()) < 1e-14
    assert float((centred_J(l + 7.0, J) - c).abs().max()) < 1e-12
    assert float(norm2_m(torch.ones(2, 100, dtype=F64))[0]) == 1.0
    assert scale_s2(torch.full((4, 100), 2.0, dtype=F64), [0.25] * 4) == pytest.approx(4.0)


def test_whole_map_affine_negative_control_gives_zero(P):
    """F(z, q) = A z + B q + t on the whole map (here: an affine map of the full branch input) -> R = A d, L = 0."""
    g = torch.Generator().manual_seed(3)
    W = torch.randn(768, 100, generator=g, dtype=F64) / 30
    t = torch.randn(100, generator=g, dtype=F64)
    Fmap = lambda x: centred_J(x @ W + t, list(range(100)))  # noqa: E731
    x_rec, x_a = torch.randn(768, generator=g, dtype=F64), torch.randn(768, generator=g, dtype=F64)
    xm, xp = hybrids(P, x_rec, torch.randn(8, 768, generator=g, dtype=F64), x_a)
    R = Fmap(xp) - Fmap(xm)
    assert u_variance(R) < 1e-24 and u_pairs(R) < 1e-24


def test_bilinear_positive_control_matches_analytic_L2():
    """F(z, q) = A z + B q + beta z_1 q_1 v: R_k = A d + beta d_1 q_1k v, so with the uniform 8-point Q
    L^2 = beta^2 d_1^2 Var_Q(q_1) ||v||_m^2 / s^2 (population variance over the 8 donors)."""
    g = torch.Generator().manual_seed(4)
    m, dz, dq = 100, 6, 5
    A = torch.randn(m, dz, generator=g, dtype=F64)
    Bq = torch.randn(m, dq, generator=g, dtype=F64)
    v = torch.randn(m, generator=g, dtype=F64)
    v = v - v.mean()
    beta = 0.7
    F = lambda z, q: z @ A.T + q @ Bq.T + beta * z[:, :1] * q[:, :1] * v  # noqa: E731
    z = torch.randn(dz, generator=g, dtype=F64)
    d = torch.randn(dz, generator=g, dtype=F64)
    q = torch.randn(8, dq, generator=g, dtype=F64)
    R = F((z + d).expand(8, -1), q) - F(z.expand(8, -1), q)
    s2 = 1.9
    analytic = beta ** 2 * float(d[0]) ** 2 * float(q[:, 0].var(unbiased=False)) * float(norm2_m(v)) / s2
    assert u_variance(R) / s2 == pytest.approx(analytic, rel=1e-12)
    assert u_pairs(R) / s2 == pytest.approx(analytic, rel=1e-12)


# ------------------------------------------------------------------ 4. hook (tiny GPT-2)

def test_branch_hook_last_token_only_and_bypass_untouched(tiny):
    b = tiny_batch()
    block = tiny.transformer.h[LAYER]
    cap = {}
    h1 = block.ln_2.register_forward_pre_hook(lambda mod, i: cap.setdefault("resid", []).append(i[0].clone()))
    h2 = block.register_forward_hook(lambda mod, i, o: cap.setdefault("out", []).append((o[0] if isinstance(o, tuple) else o).clone()))
    try:
        with branch_mlp(tiny, LAYER) as w:
            base = forward(tiny, w, b)
            ident = forward(tiny, w, b, branch=base["x_in_last"])
            br = torch.randn(3, 24, dtype=F64)
            pat = forward(tiny, w, b, branch=br)
    finally:
        h1.remove()
        h2.remove()
    assert torch.equal(ident["logits"], base["logits"])  # replacing by the same value is the identity
    assert torch.equal(cap["resid"][0], cap["resid"][2])  # residual bypass unchanged
    ar, li = torch.arange(3), b.last_idx
    out_p = cap["out"][2]
    assert float((out_p[ar, li] - (cap["resid"][2][ar, li] + block.mlp(br.unsqueeze(1))[:, 0])).abs().max()) < 1e-12
    other = torch.arange(out_p.shape[1])[None, :] != li[:, None]
    assert torch.equal(cap["out"][0][other], out_p[other])
    pos = torch.arange(out_p.shape[1])[None, :] < li[:, None]
    assert torch.equal(base["hidden_all"][pos], pat["hidden_all"][pos])
    assert not torch.equal(base["logits"], pat["logits"])


def test_hook_guards(tiny):
    with wrapped_mlp(tiny, LAYER):
        with pytest.raises(RuntimeError, match="nesting"):
            with branch_mlp(tiny, LAYER):
                pass
    b = tiny_batch()
    tiny.train()
    try:
        with branch_mlp(tiny, LAYER) as w, pytest.raises(RuntimeError, match="eval"):
            forward(tiny, w, b)
    finally:
        tiny.eval()


def test_float64_vs_float32_tiny_model_responses(tiny, P):
    """Precision comparison of the full R pipeline on a tiny model (the real MPS-f32 vs CPU-f64 check is in smoke)."""
    t32 = GPT2LMHeadModel(tiny.config)
    t32.load_state_dict({k: v.float() for k, v in tiny.state_dict().items()})
    t32.eval()
    b = tiny_batch(n=1)
    g = torch.Generator().manual_seed(5)
    P24 = torch.linalg.qr(torch.randn(24, 24, generator=g, dtype=F64))[0][:, :10]
    P24 = P24 @ P24.T
    res = {}
    for name, mdl, dt in (("f64", tiny, F64), ("f32", t32, torch.float32)):
        with branch_mlp(mdl, LAYER) as w:
            x_rec = forward(mdl, w, b)["x_in_last"][0].to(F64)
            x_don = torch.randn(8, 24, generator=torch.Generator().manual_seed(6), dtype=F64) + x_rec
            x_a = x_rec + torch.randn(24, generator=torch.Generator().manual_seed(7), dtype=F64)
            xm, xp = hybrids(P24, x_rec, x_don, x_a)
            bb = make_batch([b.ids[0].tolist()] * 16, 0, torch.device("cpu"))
            lg = forward(mdl, w, bb, branch=torch.cat([xm, xp]).to(dt))["logits"].to(F64)
        Fc = centred_J(lg, list(range(61)))
        res[name] = Fc[8:] - Fc[:8]
    rel = float((res["f32"] - res["f64"]).norm() / res["f64"].norm())
    assert rel < 1e-4
    assert u_variance(res["f32"]) == pytest.approx(u_variance(res["f64"]), rel=1e-3)


# ------------------------------------------------------------------ 5. bootstrap

def _groups(frames=(("A", 3), ("B", 5)), seed=0):
    rng = np.random.default_rng(seed)
    gs, U = [], []
    for f, nt in frames:
        for t in range(nt):
            for k in range(2):
                gs.append({"frame": f, "template": f"{f}|t{t}", "group_id": f"{f}{t}{k}"})
                U.append(float(rng.gamma(2.0, 0.01)))
    return gs, np.array(U)


def test_resampler_matches_bruteforce_template_resampling():
    """Resampling templates carries both groups; frame weights stay 1/F even with unequal template counts."""
    gs, U = _groups()
    s2 = 1.3
    T = template_T(gs, U, s2)
    frames = sorted(T)
    vals = [np.array([T[f][t] for t in sorted(T[f])]) for f in frames]
    ts = resample_theta(vals, 20, rng_for(9, 7))
    rng = rng_for(9, 7)  # replay the same index stream explicitly
    idx = [rng.integers(0, len(v), size=(20, len(v)), dtype=np.int16) for v in vals]
    for b in range(20):
        per_frame = []
        for fi, f in enumerate(frames):
            tmpl = sorted(T[f])
            rows = [np.mean([u for g, u in zip(gs, U) if g["template"] == tmpl[ti]]) / s2 for ti in idx[fi][b]]
            per_frame.append(np.mean(rows))  # group-level expansion of the resampled templates
        assert ts[b] == pytest.approx(np.mean(per_frame), rel=1e-12)


def test_resampler_variance_matches_bootstrap_theory():
    rng = np.random.default_rng(3)
    vals = [rng.gamma(2.0, 0.003, size=n) for n in (6, 6, 9, 12)]
    ts = resample_theta(vals, 40000, rng_for(5, 7))
    theory = sum(v.var() / len(v) for v in vals) / 16  # (1/F^2) sum_f sigma_pop^2 / n_f
    assert ts.var() == pytest.approx(theory, rel=0.05)
    assert ts.mean() == pytest.approx(np.mean([v.mean() for v in vals]), rel=0.01)


def test_template_bootstrap_bounds_and_decisions():
    gs, U = _groups((("A", 6), ("B", 6), ("C", 6), ("D", 6)))
    T = template_T(gs, U, 1.0)
    r = template_bootstrap(T, 2000, 1, 0.95)
    assert r["theta_lower"] <= r["theta"] <= r["theta_upper"]
    assert r["L"] == pytest.approx(r["theta"] ** 0.5)
    assert confirm_decision({"L_lower": 0.06, "L_upper": 0.2}, 0.05) == "SUPPORTED_FLOOR"
    assert confirm_decision({"L_lower": 0.01, "L_upper": 0.05}, 0.05) == "TARGET_FLOOR_RULED_OUT"
    assert confirm_decision({"L_lower": 0.01, "L_upper": 0.08}, 0.05) == "INCONCLUSIVE"
    assert confirm_decision(r, 0.05, valid=False) == "EVALUATION_INVALID"
    vals = [np.array([T[f][x] for x in sorted(T[f])]) for f in sorted(T)]
    lo_m, hi_m = percentile_bounds(resample_theta(vals, 2000, rng_for(1, 7)), 0.95)
    assert r["theta_lower"] == lo_m and r["theta_upper"] == hi_m  # same resampler as the power simulation
    lo, hi = percentile_bounds(np.arange(101.0), 0.95)
    assert lo == pytest.approx(5.0) and hi == pytest.approx(95.0)  # linear quantiles


# ------------------------------------------------------------------ 6. power planning

def test_frame_moments_and_template_T():
    gs, U = _groups((("A", 6), ("B", 6)))
    T = template_T(gs, U, 2.0)
    m, v, mbar = frame_moments(T)
    a = [np.mean(U[i:i + 2]) / 2.0 for i in range(0, 12, 2)]
    assert m["A"] == pytest.approx(np.mean(a)) and v["A"] == pytest.approx(np.var(a, ddof=1))
    assert mbar == pytest.approx((m["A"] + m["B"]) / 2)


def test_gamma_matching_and_scenarios():
    m, v = {"A": 0.004, "B": 0.006}, {"A": 1e-5, "B": 2e-5}
    mbar = 0.005
    mu, V, r = scenario_moments(m, v, mbar, 0.0064, 4, "kappa_r2_v")
    assert r == pytest.approx(1.28) and mu["A"] == pytest.approx(1.28 * 0.004) and V["B"] == pytest.approx(4 * 1.28 ** 2 * 2e-5)
    mu, V, _ = scenario_moments(m, v, mbar, 0.0025, 1, "kappa_v")
    assert V["A"] == v["A"]
    sh, sc = gamma_params(0.005, 1e-5)
    x = np.random.default_rng(0).gamma(sh, sc, 400000)
    assert x.mean() == pytest.approx(0.005, rel=0.01) and x.var() == pytest.approx(1e-5, rel=0.03) and (x > 0).all()
    assert len(SCENARIOS) == 4


def test_moment_guard_classes(cfg):
    g = cfg["power"]["guards"]
    assert guard({"A": 0.01, "B": 0.02}, {"A": 1e-5, "B": 1e-5}, 1e-9, g)["ok"]
    r = guard({"A": float("nan"), "B": 0.02}, {"A": 1e-5, "B": 1e-5}, 0.0, g)
    assert r["invalid"] and not r["ok"]
    r = guard({"A": 1e-12, "B": 0.02}, {"A": 1e-5, "B": 1e-5}, 0.0, g)
    assert r["per_frame"]["A"] == "near_zero_floor" and not r["invalid"]
    r = guard({"A": 0.01, "B": 0.02}, {"A": 0.0, "B": 1e-5}, 0.0, g)
    assert r["per_frame"]["A"] == "zero_variance"
    r = guard({"A": 1e-6, "B": 0.02}, {"A": 1e-5, "B": 1e-5}, 1e-7, g)  # floor = 100 x numerical noise
    assert r["per_frame"]["A"] == "near_zero_floor"


def test_clopper_pearson_edges():
    from scipy.stats import beta
    assert clopper_pearson(0, 50, 0.05)[0] == 0.0 and clopper_pearson(50, 50, 0.05)[1] == 1.0
    lo, hi = clopper_pearson(40, 50, 0.05)
    assert lo == pytest.approx(beta.ppf(0.025, 40, 11)) and hi == pytest.approx(beta.ppf(0.975, 41, 10))


def _table(cfg, fn, outer=2000):
    """Full 5 x 4 table; fn(N, scenario) -> (alt_lo, alt_hi, null_hi)."""
    rows = []
    for N in cfg["power"]["candidates"]:
        for k, md in SCENARIOS:
            s = scenario_name(k, md)
            a_lo, a_hi, n_hi = fn(N, s)
            rows.append({"N": N, "scenario": s, "alt": {"n": outer, "cp_lo": a_lo, "cp_hi": a_hi},
                         "null": {"n": outer, "cp_lo": 0.0, "cp_hi": n_hi}})
    return rows


def test_power_decision_rules(cfg):
    good = _table(cfg, lambda N, s: (0.85, 0.95, 0.06) if N >= 96 else (0.5, 0.7, 0.05))
    d = decide(cfg, good, 2000)
    assert d["decision"] == "POWER_PLAN_CANDIDATE" and d["candidate_N"] == 96
    alarm = _table(cfg, lambda N, s: (0.85, 0.95, 0.09 if s == "kappa4_r2v" else 0.05))
    assert decide(cfg, alarm, 2000)["decision"] == "POWER_NOT_ESTABLISHED"  # null alarm in one scenario
    low = _table(cfg, lambda N, s: (0.6, 0.79 if s == "kappa4_v" else 0.95, 0.05))
    assert decide(cfg, low, 2000)["decision"] == "NO_GO_BUDGET"
    assert decide(cfg, good, 2000, prescribed=False)["decision"] == "POWER_NOT_ESTABLISHED"


def test_power_decision_rejects_missing_duplicate_or_short_rows(cfg):
    good = _table(cfg, lambda N, s: (0.85, 0.95, 0.06))
    assert validate_rows(cfg, good, 2000) == []
    for rows in (good[:-1],                                   # missing (N, scenario)
                 good + [good[0]],                            # duplicate
                 [dict(r, alt=dict(r["alt"], n=1999)) if i == 3 else r for i, r in enumerate(good)],  # short
                 [{k: v for k, v in r.items() if k != "null"} if i == 7 else r for i, r in enumerate(good)]):  # missing hypothesis
        assert validate_rows(cfg, rows, 2000)
        d = decide(cfg, rows, 2000)
        assert d["decision"] == "POWER_NOT_ESTABLISHED" and "incomplete" in d["reason"]
    low = _table(cfg, lambda N, s: (0.1, 0.2, 0.05))
    assert decide(cfg, low[:-1], 2000)["decision"] == "POWER_NOT_ESTABLISHED"  # never NO_GO on an incomplete table


def test_run_power_small_is_deterministic_and_never_prescribed(cfg):
    m, v = {f: 0.006 for f in "ABCD"}, {f: 4e-6 for f in "ABCD"}
    a = run_power(cfg, m, v, 0.006, 0.0, outer=30, B=200)
    b = run_power(cfg, m, v, 0.006, 0.0, outer=30, B=200)
    assert [r["alt"]["successes"] for r in a["rows"]] == [r["alt"]["successes"] for r in b["rows"]]
    assert a["complete"] and not a["prescribed"] and a["decision"] == "POWER_NOT_ESTABLISHED"
    assert len(a["rows"]) == 5 * 4 and a["n_comparisons"] == 40
    assert a["cp_alpha_per_comparison"] == pytest.approx(0.05 / 40)


def test_run_power_compute_cap_labels_partial_output(cfg):
    m, v = {f: 0.006 for f in "ABCD"}, {f: 4e-6 for f in "ABCD"}
    r = run_power(cfg, m, v, 0.006, 0.0, cap_s=0.0)
    assert not r["complete"] and r["decision"] == "POWER_NOT_ESTABLISHED" and "cap" in r["reason"]


def test_run_power_invalid_vs_not_estimable(cfg):
    m_ok = {f: 0.006 for f in "ABCD"}
    v = {f: 1e-6 for f in "ABCD"}
    r = run_power(cfg, {**m_ok, "A": float("nan")}, v, 0.006, 0.0, outer=5, B=10)
    assert r["decision"] == "EVALUATION_INVALID"
    r = run_power(cfg, {**m_ok, "B": -1e-3}, v, 0.006, 0.0, outer=5, B=10)
    assert r["decision"] == "EVALUATION_INVALID"
    r = run_power(cfg, {f: 0.0 for f in "ABCD"}, v, 0.0, 0.0, outer=5, B=10)
    assert r["decision"] == "POWER_NOT_ESTABLISHED" and "not estimable" in r["reason"]
    r = run_power(cfg, m_ok, {**v, "C": 0.0}, 0.006, 0.0, outer=5, B=10)
    assert r["decision"] == "POWER_NOT_ESTABLISHED" and "zero_variance" in r["reason"]


def test_simulation_streams_reproducible_independent_and_restartable():
    frames = list("ABCD")
    shapes = [gamma_params(0.005, 1e-5)] * 4
    a = [simulate_dataset(frames, shapes, 6, 500, 0.0025, 0.95, 11, (0, 0, 0, j)) for j in range(12)]
    b = [simulate_dataset(frames, shapes, 6, 500, 0.0025, 0.95, 11, (0, 0, 0, j)) for j in range(12)]
    assert a == b  # reproducible
    assert len({lo for _, lo in a}) == 12  # every dataset has its own data and bootstrap draws
    mu, V = {f: 0.005 for f in frames}, {f: 1e-5 for f in frames}
    full = simulate(frames, mu, V, 6, 12, 500, 0.0025, 0.95, 11, (0, 0, 0), None)
    tail = simulate(frames, mu, V, 6, 12, 500, 0.0025, 0.95, 11, (0, 0, 0), None, start=7)
    assert full == (sum(s for s, _ in a), 12) and tail == (sum(s for s, _ in a[7:]), 5)
    # the bootstrap stream is independent of the data stream: same template values, different bootstrap keys
    T = [np.full(6, 0.004) + 0.001 * np.arange(6)] * 4
    t1, t2 = resample_theta(T, 300, rng_for(11, 3, 0, 0, 0, 0)), resample_theta(T, 300, rng_for(11, 3, 0, 0, 0, 1))
    assert not np.array_equal(t1, t2)
    d1 = rng_for(11, 2, 0, 0, 0, 0).integers(0, 6, 50)
    b1 = rng_for(11, 3, 0, 0, 0, 0).integers(0, 6, 50)
    assert not np.array_equal(d1, b1)


def test_power_null_rejection_rate_near_nominal_small_check(cfg):
    """Sanity of the simulated procedure (not the prescribed size): under the null the rejection rate of the
    one-sided 95% percentile lower bound is in a broad band around 5%."""
    m = {f: 0.0025 for f in "ABCD"}
    v = {f: 2e-6 for f in "ABCD"}
    succ, n = simulate(sorted(m), m, v, 30, 400, 2000, 0.0025, 0.95, 21, (9, 9, 9), None)
    assert n == 400 and 0.01 <= succ / n <= 0.13


# ------------------------------------------------------------------ 7. manifests (outcome-blind prepare)

@needs_manifests
def test_manifest_structure_freshness_and_leakage(cfg):
    groups, index = data.load_groups(cfg)
    asg = index["assignment"]
    prompts = data.long_prompts(cfg, groups)
    hist, files = data.historical_texts(cfg)
    assert "followup/manifests/test_recipients.csv" in files
    info = data.structural_checks(cfg, asg, groups, prompts, hist)
    assert info == {"n_groups": 48, "n_templates": 24, "n_prompts": 480, "n_donor_contexts": 384,
                    "n_historical_texts": len(hist)}
    assert json.loads(json.dumps(data.assign(cfg))) == asg  # deterministic, outcome-blind
    assert len({p["text_sha256"] for p in prompts}) == 480 and len({p["ids_sha256"] for p in prompts}) == 480
    for g in groups:
        assert all(g[f"don{k}_text"].count(f"17{g['r']:02d}") == 1 for k in range(8))  # same year as the recipient
        assert len({g[f"don{k}_text"] for k in range(8)}) == 8  # 8 distinct enumerated donors


@needs_manifests
def test_leakage_and_duplicate_detection(cfg):
    groups, index = data.load_groups(cfg)
    prompts = data.long_prompts(cfg, groups)
    hist, _ = data.historical_texts(cfg)
    with pytest.raises(AssertionError, match="historical"):
        data.structural_checks(cfg, index["assignment"], groups, prompts, hist | {prompts[5]["text"]})
    dup = [dict(p) for p in prompts]
    dup[1]["text"] = dup[0]["text"]
    with pytest.raises(AssertionError):
        data.structural_checks(cfg, index["assignment"], groups, dup, hist)


@needs_manifests
def test_prepare_refuses_existing_manifests(cfg):
    with pytest.raises(FileExistsError):
        data.prepare(None, cfg)


@needs_manifests
def test_separate_aggregation_matches_pipeline_on_synthetic_raw(cfg, tmp_path):
    groups, _ = data.load_groups(cfg)
    g = torch.Generator().manual_seed(8)
    raw = torch.randn(48, 19, 100, generator=g, dtype=F64) * 3
    raw[:, 18] = raw[:, 0]  # duplicate control row
    torch.save({"row_kinds": pipeline.ROW_KINDS, "group_ids": [x["group_id"] for x in groups],
                "J_ids": list(range(100)), "raw_J_logits": raw}, tmp_path / "raw_J_logits.pt")
    stats = [pipeline.group_stats(raw[i]) for i in range(48)]
    ctl = [{"identity_max_abs": 0.0, "duplicate_max_abs": 0.0, "finite": True} for _ in range(48)]
    summ = pipeline.summarise(cfg, groups, stats, ctl)
    data.write_csv(tmp_path / "groups_U.csv", [{"group_id": x["group_id"], "U": s["U"]} for x, s in zip(groups, stats)])
    from discarded_response_floor_gate.common import bound_hashes
    bh = bound_hashes(cfg)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"manifest_dir": bh["manifest_dir"], "bound": bh}))
    agg = aggregate(tmp_path)
    assert agg["groups_csv"] == "discarded_response_floor_gate/manifests_v2/groups.csv"
    assert summ["valid"]
    assert agg["Lhat2"] == pytest.approx(summ["Lhat2"], rel=1e-12) and agg["s2"] == pytest.approx(summ["s2"], rel=1e-12)
    assert agg["U_max_abs_diff_vs_groups_U_csv"] < 1e-12 and agg["U_two_way_max_rel"] < 1e-12
    for f in agg["frame_mean_T"]:
        assert agg["frame_mean_T"][f] == pytest.approx(summ["frame_mean_T"][f], rel=1e-12)
        assert agg["frame_var_T"][f] == pytest.approx(summ["frame_var_T"][f], rel=1e-12)


@needs_manifests
def test_summarise_flags_invalid_measurements(cfg):
    groups, _ = data.load_groups(cfg)
    raw = torch.randn(48, 19, 100, dtype=F64)
    raw[:, 18] = raw[:, 0]
    stats = [pipeline.group_stats(raw[i]) for i in range(48)]
    bad_ctl = [{"identity_max_abs": 1.0, "duplicate_max_abs": 0.0, "finite": True} for _ in range(48)]
    assert not pipeline.summarise(cfg, groups, stats, bad_ctl)["valid"]
    raw[:, 16] = raw[:, 17]  # natural response exactly zero -> scale invalid
    stats = [pipeline.group_stats(raw[i]) for i in range(48)]
    ok_ctl = [{"identity_max_abs": 0.0, "duplicate_max_abs": 0.0, "finite": True} for _ in range(48)]
    out = pipeline.summarise(cfg, groups, stats, ok_ctl)
    assert not out["valid"] and "scale" in out["invalid_reasons"][0]


# ------------------------------------------------------------------ 8. stage guards and preserved files

def test_stages_refuse_without_registration(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "REGISTRATION", tmp_path / "none.json")
    final = {**cfg, "protocol_version": "drfg-dev-v2"}
    with pytest.raises(RuntimeError, match="registration"):
        pipeline.develop(final)
    with pytest.raises(RuntimeError, match="registration"):
        pipeline.power(final, "x")


def test_register_refuses_draft_and_existing(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "REGISTRATION", tmp_path / "r.json")
    with pytest.raises(RuntimeError, match="protocol_version"):
        pipeline.register({**cfg, "protocol_version": "drfg-dev-v1-DRAFT"}, "x", "y")
    (tmp_path / "r.json").write_text("{}")
    with pytest.raises(FileExistsError):
        pipeline.register({**cfg, "protocol_version": "drfg-dev-v2"}, "x", "y")
    with pytest.raises(FileExistsError):
        write_once(tmp_path / "r.json", {})


def test_verify_registration_detects_changes(cfg, monkeypatch, tmp_path):
    from discarded_response_floor_gate.common import bound_hashes
    p = tmp_path / "reg.json"
    final = {**cfg, "protocol_version": "drfg-dev-v2"}
    good = {"bound": bound_hashes(cfg), "protocol_version": "drfg-dev-v2"}
    p.write_text(json.dumps(good))
    monkeypatch.setattr(pipeline, "REGISTRATION", p)
    assert pipeline.verify_registration(final)
    p.write_text(json.dumps({**good, "protocol_version": "drfg-dev-v1"}))
    with pytest.raises(RuntimeError, match="another protocol version"):
        pipeline.verify_registration(final)
    p.write_text(json.dumps(good))
    assert pipeline.verify_registration(final)
    with pytest.raises(RuntimeError, match="DRAFT"):
        pipeline.verify_registration({**cfg, "protocol_version": "drfg-dev-v1-DRAFT"})
    bad = json.loads(json.dumps(good))
    bad["bound"]["scp_dependency_tree_sha256"] = "0" * 64
    p.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="scp_dependency"):
        pipeline.verify_registration(final)


def test_cli_requires_arguments():
    from discarded_response_floor_gate import cli
    for argv in (["register"], ["register", "--smoke-run", "x"], ["power"], ["report"]):
        with pytest.raises(SystemExit):
            cli.main(argv)


def test_old_work_untouched(cfg):
    from scp.common import source_hashes
    assert source_hashes()["tree_sha256"] == cfg["scp_dependency_tree_sha256"]
    from followup.common import bound_hashes as fb, load_config as flc
    reg = json.loads((ROOT / "followup" / "registration" / "FOLLOWUP_REGISTRATION.json").read_text())
    assert fb(flc()) == reg["bound"]
    if PROTECTED.exists():
        r = verify_protected()
        assert r["ok"], r


# ------------------------------------------------------------------ 9. development resource caps (fake model)

class _FakeClock:
    def __init__(self, dt):
        self.t, self.dt = 0.0, dt
        self.start = 0.0

    def __call__(self):
        self.t += self.dt
        return self.t


class _FakeCtx:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        Q = torch.linalg.qr(torch.randn(24, 24, generator=g, dtype=F64))[0][:, :10]
        self.P = Q @ Q.T
        self.Wc, self.b_fc = torch.randn(24, 30, dtype=F64), torch.zeros(30, dtype=F64)
        self.calls = 0

    def group_rows(self, rec_ids, rows):
        self.calls += 1
        return torch.randn(len(pipeline.ROW_KINDS), 100, dtype=F64), None


def _fake_dev(n=6):
    groups = [{"group_id": f"g{i}", "rec_ids": [1, 2, 3]} for i in range(n)]
    return groups, torch.randn(n * 10, 24, dtype=F64), torch.randn(n * 10, 100, dtype=F64)


BUDGET = {"model_stage_cap_s": 100.0, "memory_cap_gib": 24.0, "timed_groups": 3, "projection_overhead_factor": 1.25}


def test_resource_status_rules():
    assert pipeline.resource_status(10, 1, BUDGET) is None
    assert pipeline.resource_status(101, 1, BUDGET) == "resource_stop_time_cap"
    assert pipeline.resource_status(10, 25, BUDGET) == "resource_stop_memory_cap"
    assert pipeline.resource_status(10, 1, BUDGET, projected=150) == "resource_stop_projection"


def test_run_groups_stops_before_next_group_and_keeps_partial_outputs():
    groups, X, LJ = _fake_dev()
    ctx = _FakeCtx()
    # each group consumes 3 clock calls of 15 s -> elapsed 45, 90, 135: the time cap stops after group 3
    res = pipeline.run_groups(ctx, groups, X, LJ, {**BUDGET, "timed_groups": 99}, _FakeClock(15.0), lambda: 1.0)
    assert res["status"] == "resource_stop_time_cap" and ctx.calls == 3 and len(res["raw"]) == 3
    assert len(res["controls"]) == 3 and len(res["support"]) == 3 * 8 and res["budget"]["stopped_after_group"] == 3
    ctx = _FakeCtx()
    res = pipeline.run_groups(ctx, groups, X, LJ, BUDGET, _FakeClock(0.1), lambda: 30.0)
    assert res["status"] == "resource_stop_memory_cap" and ctx.calls == 1  # memory checked after every group
    ctx = _FakeCtx()
    res = pipeline.run_groups(ctx, groups, X, LJ, BUDGET, _FakeClock(4.0), lambda: 1.0)
    # after group 3: elapsed 36, projection 36 + 4 x 3 x 1.25 = 51 -> no stop; all groups done
    assert res["status"] == "ok" and ctx.calls == 6 and res["budget"]["projected_model_stage_s"] == pytest.approx(51.0)
    ctx = _FakeCtx()
    # 60 groups, 2 s per group forward: after group 3 elapsed 18, projection 18 + 2 x 57 x 1.25 = 160.5 > 100
    res = pipeline.run_groups(ctx, _fake_dev(60)[0], *_fake_dev(60)[1:], BUDGET, _FakeClock(2.0), lambda: 1.0)
    assert res["status"] == "resource_stop_projection" and ctx.calls == 3


def test_run_groups_records_cap_exceeded_after_final_group():
    groups, X, LJ = _fake_dev(2)
    ctx = _FakeCtx()
    res = pipeline.run_groups(ctx, groups, X, LJ, {**BUDGET, "timed_groups": 99}, _FakeClock(30.0), lambda: 1.0)
    assert res["status"] == "ok" and ctx.calls == 2
    assert res["budget"]["cap_exceeded_after_final_group"] == "resource_stop_time_cap"


# ------------------------------------------------------------------ 10. v2 donor assignment (GPT Pro option B)

@pytest.fixture(scope="module")
def tok(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])


def test_v2_config_paths_and_offsets(cfg):
    assert cfg["protocol_version"] == "drfg-dev-v2" and cfg["_path"].endswith("gate_dev_v2.json")
    assert manifest_dir(cfg) == GROOT / "manifests_v2"
    assert registration_path(cfg) == GROOT / "registration" / "DEV_REGISTRATION_v2.json"
    v1 = load_config(CFG_V1)
    assert manifest_dir(v1) == GROOT / "manifests" and registration_path(v1).name == "DEV_REGISTRATION.json"
    assert data.donor_offsets(cfg) == [0, 1, 4, 9, 15, 22, 32, 34]
    assert data.donor_offsets(v1) == [6 * k for k in range(8)]
    prof = data.offset_profile(data.donor_offsets(cfg), 48)
    assert prof["positive_differences_distinct"] and len(prof["positive_differences"]) == 28
    assert prof["max_modular_multiplicity"] == 2
    old = data.offset_profile(data.donor_offsets(v1), 48)
    assert old["max_modular_multiplicity"] == 8 and not old["positive_differences_distinct"]
    # v1 records preserved with the hashes recorded in the v2 config
    from scp.common import sha256_file
    rec = cfg["amendment"]["v1_records_preserved"]
    assert sha256_file(CFG_V1) == rec["config_sha256"] and sha256_file(GROOT / "DEV_PROTOCOL.md") == rec["protocol_sha256"]
    for n, s in rec["manifests"].items():
        assert sha256_file(GROOT / "manifests" / n) == s


@needs_manifests
def test_v2_manifest_acceptance_and_overlap_matrices(cfg):
    groups, index = data.load_groups(cfg)
    acc = index["acceptance"]
    assert acc["pair_overlap_counts"] == {"0": 24, "1": 864, "2": 240} or acc["pair_overlap_counts"] == {0: 24, 1: 864, 2: 240}
    assert acc["n_group_pairs"] == 1128 and acc["total_shared_noun_incidences"] == 1344
    assert acc["mean_pair_overlap"] == pytest.approx(56 / 47)
    M, tmpl, TM = data.noun_overlaps(groups, 8)
    import csv
    with open(manifest_dir(cfg) / "group_noun_overlap.csv") as f:
        saved = [[int(v) for k, v in r.items() if k != "group_id"] for r in csv.DictReader(f)]
    assert np.array_equal(np.array(saved), M) and M.max() == 8 and M[~np.eye(48, dtype=bool)].max() == 2
    with open(manifest_dir(cfg) / "template_noun_overlap.csv") as f:
        saved_t = [[int(v) for k, v in r.items() if k != "template"] for r in csv.DictReader(f)]
    assert np.array_equal(np.array(saved_t), TM)
    # recipients / retained donors / cells / years / donor frames identical to v1
    ref = index["reference_comparison"]
    assert ref["donor_slots_changed"] > 0 and "rec_text" in ref["identical_fields"]
    old, _ = data.load_groups(load_config(CFG_V1))
    assert [g["rec_text"] for g in old] == [g["rec_text"] for g in groups]
    assert [g["ret_text"] for g in old] == [g["ret_text"] for g in groups]
    assert {g["don0_noun"] for g in old} == {g["don0_noun"] for g in groups}  # same global noun pool


def test_v1_manifests_rebuild_byte_identically(tok, tmp_path):
    from scp.common import sha256_file
    v1 = load_config(CFG_V1)
    asg, groups, prompts, *_ = data.build_all(tok, v1)
    data.write_csv(tmp_path / "groups.csv", groups)
    data.write_csv(tmp_path / "prompts.csv", prompts)
    for n in ("groups.csv", "prompts.csv"):
        assert sha256_file(tmp_path / n) == sha256_file(GROOT / "manifests" / n)


def test_acceptance_blocks_aliasing_offsets_and_reference_mismatch(cfg, tok):
    import copy
    bad = copy.deepcopy(cfg)
    bad["design"]["donor_offsets"] = [6 * k for k in range(8)]
    asg = data.assign(bad)
    groups = data.build(tok, bad, asg)
    prompts = data.long_prompts(bad, groups)
    with pytest.raises(AssertionError):
        data.acceptance_checks(bad, groups, prompts, asg)
    good_groups = data.build(tok, cfg, data.assign(cfg))
    good_groups[3] = dict(good_groups[3], r=good_groups[3]["r"] + 1)
    with pytest.raises(ValueError, match="differs from the v1 reference"):
        data.compare_with_reference(cfg, good_groups)
    with pytest.raises(ValueError):
        data.donor_offsets({"design": {"K": 8, "n_groups": 48, "donor_offsets": [0, 1, 2, 3, 4, 5, 6, 54]}})


def test_register_requires_smoke_under_same_config_and_manifest_dir(cfg, monkeypatch, tmp_path):
    from discarded_response_floor_gate.common import bound_hashes
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    (smoke / "smoke.json").write_text(json.dumps({"all_gates_pass": True}))
    bh = bound_hashes(cfg)
    man = {"config_path": "discarded_response_floor_gate/config/gate_dev_v1.json", "bound": bh}
    monkeypatch.setattr(pipeline, "load_run", lambda rid, stage: (smoke, man))
    monkeypatch.setattr(pipeline, "REGISTRATION", tmp_path / "reg.json")
    with pytest.raises(RuntimeError, match="different config file"):
        pipeline.register(cfg, "x", None)
    man2 = {"config_path": bh["config_path"], "bound": {**bh, "manifest_dir": "discarded_response_floor_gate/manifests"}}
    monkeypatch.setattr(pipeline, "load_run", lambda rid, stage: (smoke, man2))
    with pytest.raises(RuntimeError, match="different config file or manifest"):
        pipeline.register(cfg, "x", None)
