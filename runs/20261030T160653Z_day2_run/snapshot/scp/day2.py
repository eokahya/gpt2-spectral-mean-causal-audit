"""Day-2 stages (PROTOCOL_DAY2): quadcheck, smoke (incl. Day-1 replay), run (geometric controls + compensation)."""
import csv
import math
import time

import numpy as np
import torch

from . import ROOT
from .bases import build_bases
from .experiment import (DONORS, BudgetExceeded, Context, RowWriter, moment_pool_name, orig_runs,
                         positive_control, response)
from .matching import apply_match, l2_norm, preact_norm, scale_quantiles
from .metrics import centered, kl, prob_diff, rms, t_score
from .model import forward_last, make_reduction, wrapped_mlp
from .spectral import calibrated_moments, gauss_expect_gelu, weights_only_moments

F64 = torch.float64
REGIMES = ("native", "l2", "preact")


# ---------------------------------------------------------------- shared helpers

def boot(x, stat, B, seed, clusters=None, ci=0.95):
    """Vectorised percentile bootstrap of mean or RMS over pairs (iid) or template clusters.

    Returns (point, lo, hi, n_units), the same semantics as metrics.bootstrap_ci."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("non-finite values in bootstrap input")
    v = x ** 2 if stat == "rms" else x
    rng = np.random.default_rng(seed)
    if clusters is None:
        units = len(v)
        s = v[rng.integers(0, units, (B, units))].mean(1)
    else:
        _, inv = np.unique(np.asarray(clusters), return_inverse=True)
        units = int(inv.max()) + 1
        sums = np.bincount(inv, weights=v, minlength=units)
        cnt = np.bincount(inv, minlength=units).astype(float)
        pick = rng.integers(0, units, (B, units))
        s = sums[pick].sum(1) / cnt[pick].sum(1)
    point = v.mean()
    if stat == "rms":
        point, s = np.sqrt(point), np.sqrt(s)
    a = (1 - ci) / 2
    return float(point), float(np.quantile(s, a)), float(np.quantile(s, 1 - a)), units


def ci2(x, stat, B, seed, clusters):
    """Pair-iid and template-cluster CIs together."""
    return {f"{stat}_ci": boot(x, stat, B, seed), f"{stat}_cluster_ci": boot(x, stat, B, seed, clusters)}

def pool_inputs(ctx, w):
    x_last, x_all = ctx.moment_activations(w)
    return x_last.to(F64), x_all.to(F64)


def basis_moments(ctx, basis, x_last, x_all):
    """Fixed moments of the discarded pre-activation q = (x_row @ Q^T) @ Wc, from the moment pool only."""
    _, B_row = basis.row_maps()
    Wc = ctx.mlp.c_fc.weight.detach().cpu().to(F64)
    ln = ctx.model.transformer.h[ctx.layer].ln_2
    return {"final_token": calibrated_moments(x_last, B_row, Wc),
            "all_token": calibrated_moments(x_all, B_row, Wc),
            "weights_only": weights_only_moments(B_row, Wc, ln.weight, ln.bias)}


def basis_reduction(ctx, basis, mom, method, scope):
    R_row, _ = basis.row_maps()
    pool = moment_pool_name(method, scope)
    mu, sigma = mom["final_token"] if pool == "none" else mom[pool]
    return make_reduction(R_row, ctx.mlp, mu, sigma, method, scope, ctx.cfg["quadrature_nodes"],
                          ctx.cfg["quadrature_sigma_max_checked"], ctx.device)


def indicator_alignment(r, d, w):
    """<v_rw, v_rd> / ||v_rd||^2 per pair (0 when w == r); v uncentred-then-centred as in metrics.v_rd."""
    j = np.arange(100)

    def v(a, b):
        x = (j[None, :] > b[:, None]).astype(float) - (j[None, :] > a[:, None]).astype(float)
        return x - x.mean(1, keepdims=True)
    vd, vw = v(r, d), v(r, w)
    return (vd * vw).sum(1) / (vd * vd).sum(1)


def fwd(ctx, w, b, delta=None, reduction=None):
    return forward_last(ctx.model, w, b["rec"], delta=None if delta is None else delta.to(ctx.device),
                        reduction=reduction)


# ---------------------------------------------------------------- quadcheck

def quadcheck(ctx):
    """Fix the Day-2 Gauss-Hermite node count and sigma bound from calibration moments of all bases."""
    run, cfg = ctx.run, ctx.cfg
    rule = cfg["quadrature_rule"]
    spec = ctx.spectrum()
    rng = np.random.default_rng(cfg["seed"])
    Wc = ctx.mlp.c_fc.weight.detach().cpu().to(F64)
    bias = ctx.mlp.c_fc.bias.detach().cpu().to(F64)
    with wrapped_mlp(ctx.model, ctx.layer) as w:
        x_last, x_all = pool_inputs(ctx, w)
    bases = build_bases(cfg, spec, x_last)
    samples, ranges = [], []
    for (bid, k), basis in bases.items():
        mom = basis_moments(ctx, basis, x_last, x_all)
        R_row, _ = basis.row_maps()
        for pool, x in (("final_token", x_last), ("all_token", x_all), ("weights_only", x_last)):
            mu, sd = mom[pool]
            xs = x[rng.choice(len(x), size=32, replace=False)]
            m = (xs @ R_row) @ Wc + bias + mu
            samples.append((m, sd.expand_as(m)))
            ranges.append({"basis": bid, "k": k, "pool": pool, "sigma_max": float(sd.max()),
                           "sigma_q99": float(torch.quantile(sd, 0.99)), "m_min": float(m.min()),
                           "m_max": float(m.max())})
    sig_max = max(r["sigma_max"] for r in ranges)
    step = rule["sigma_bound_round"]
    bound = math.ceil(rule["sigma_bound_factor"] * sig_max / step) * step
    m = torch.linspace(-20, 20, 801, dtype=F64).reshape(-1, 1)
    s = torch.linspace(0, bound, 41, dtype=F64).reshape(1, -1)
    ref = gauss_expect_gelu(m, s, rule["ref_nodes"])
    grid = {n: float((gauss_expect_gelu(m, s, n) - ref).abs().max()) for n in rule["candidates"]}
    real = {n: 0.0 for n in rule["candidates"]}
    for mm, ss in samples:
        r_ = gauss_expect_gelu(mm, ss, rule["ref_nodes"])
        for n in rule["candidates"]:
            real[n] = max(real[n], float((gauss_expect_gelu(mm, ss, n) - r_).abs().max()))
    ok = [n for n in rule["candidates"] if grid[n] < rule["tol_max_abs"] and real[n] < rule["tol_max_abs"]]
    out = {"observed_sigma_max": sig_max, "sigma_bound": bound, "grid_err": grid, "real_err": real,
           "chosen_nodes": min(ok) if ok else None, "ranges": ranges,
           "configured": {"nodes": cfg["quadrature_nodes"], "bound": cfg["quadrature_sigma_max_checked"]}}
    out["configured_ok"] = (cfg["quadrature_nodes"] in ok and cfg["quadrature_sigma_max_checked"] is not None
                            and sig_max <= cfg["quadrature_sigma_max_checked"])
    run.save_json("quadcheck.json", out)
    run.log.info(f"sigma_max={sig_max:.3f} bound={bound} grid={grid} real={real} chosen={out['chosen_nodes']}")
    return out


# ---------------------------------------------------------------- original-model geometric controls

def control_targets(deltas, bases, Wc):
    """Per (basis, k, S): natural-delta norms used as matching targets."""
    tg = {}
    for (bid, k), basis in bases.items():
        R_row, B_row = basis.row_maps()
        for S, M in (("R", R_row), ("B", B_row)):
            dn = deltas["natural"] @ M
            tg[(bid, k, S)] = {"l2": l2_norm(dn).numpy(), "preact": preact_norm(dn, Wc).numpy()}
    tg[("any", 0, "RB")] = {"l2": l2_norm(deltas["natural"]).numpy(),
                            "preact": preact_norm(deltas["natural"], Wc).numpy()}
    return tg


def run_controls(ctx, w, b, base_o, deltas, bases, writer, Wc, align, guard=lambda cell: None,
                 done=lambda cell: None):
    """Original model: all bases x k x donors x S x regimes (+ basis-independent RB). Returns native responses."""
    cfg, J = ctx.cfg, ctx.J
    mcfg = cfg["matching"]
    tg = control_targets(deltas, bases, Wc)
    native_resp, match_log = {}, []
    cells = [(bid, k, basis) for (bid, k), basis in bases.items()]
    cells.append(("any", 0, None))
    for bid, k, basis in cells:
        guard((bid, k))
        if basis is None:
            Ss = [("RB", None)]
        else:
            R_row, B_row = basis.row_maps()
            Ss = [("R", R_row), ("B", B_row)]
        for donor in DONORS:
            for S, M in Ss:
                native = deltas[donor] if M is None else deltas[donor] @ M
                for regime in REGIMES:
                    if regime == "native":
                        dl, info, target_name = native.float(), None, "none"
                    else:
                        if donor == "natural" and bid in ("svd", "any"):
                            target_name = "self_natural_same_S_k"
                            if bid == "any":  # natural RB matched to itself is the identity; not re-run
                                continue
                        if donor == "natural":
                            key, target_name = ("svd", k, S), "svd_natural_same_S_k"
                        else:
                            key, target_name = (bid, k, S), "same_basis_natural_same_S_k"
                        dl, info = apply_match(native, regime, tg[key][regime], mcfg, Wc)
                        match_log.append({"basis": bid, "k": k, "donor": donor, "S": S, "regime": regime,
                                          "n_degenerate": int(info["degenerate"].sum()),
                                          "n_oos": int(info["oos"].sum()),
                                          "scale_quantiles": scale_quantiles(info["scale"], info["degenerate"]),
                                          "max_rel_err_f64": info["max_rel_err_f64"],
                                          "max_rel_err_f32": info["max_rel_err_f32"]})
                    r = response(fwd(ctx, w, b, dl), base_o, J)
                    r["T"] = t_score(r["dJ"], ctx.v).numpy()
                    if regime == "native" and donor == "natural":
                        native_resp[(bid, k, S)] = r
                    if regime == "native" and bid == "any":
                        native_resp[("any", 0, f"RB_{donor}")] = r
                    writer.add(control_rows(ctx, bid, basis, k, donor, S, regime, target_name, r, dl, info,
                                            Wc, align[donor]))
        done((bid, k))
    return native_resp, match_log


def control_rows(ctx, bid, basis, k, donor, S, regime, target_name, r, dl, info, Wc, align):
    T = r["T"]
    loc = r["dout"].norm(dim=-1).numpy()
    lg = centered(r["dlog"]).norm(dim=-1).numpy()
    dl2 = l2_norm(dl).numpy()
    dpre = preact_norm(dl, Wc).numpy()
    deg = np.zeros(len(T), bool) if info is None else info["degenerate"]
    oos = np.zeros(len(T), bool) if info is None else info["oos"]
    scale = np.ones(len(T)) if info is None else info["scale"]
    rows = []
    for i, p in enumerate(ctx.pairs):
        bad = bool(deg[i])
        rows.append({
            "basis": bid, "family": "any" if basis is None else basis.family,
            "seed": "" if basis is None or basis.seed is None else basis.seed, "k": k, "donor": donor, "S": S,
            "regime": regime, "match_target": target_name, "pair_id": p["pair_id"], "template": p["template"],
            "r": p["r"], "d": p["d"], "donor_yy": p[f"{DONORS[donor]}_yy"], "v_alignment": float(align[i]),
            "scale": float(scale[i]), "degenerate": int(bad), "oos": int(oos[i]),
            "delta_l2": float(dl2[i]), "delta_preact": float(dpre[i]),
            "T": math.nan if bad else float(T[i]),
            "local_resp_norm": math.nan if bad else float(loc[i]),
            "logit_resp_norm": math.nan if bad else float(lg[i]),
        })
    return rows


# ---------------------------------------------------------------- reduced models

def run_reduced_basis(ctx, w, b, base_o, deltas, basis, mom, resp_o, writer, uninterv, vec, run, budget, reserve,
                      pd_orig, unp_writer):
    cfg, J = ctx.cfg, ctx.J
    R_row, B_row = basis.row_maps()
    phys = {"R": (deltas["natural"] @ R_row).float(), "B": (deltas["natural"] @ B_row).float(),
            "RB": deltas["natural"].float()}
    for method in cfg["methods"]:
        for scope in cfg["scopes"]:
            if run.elapsed() > budget - reserve:
                raise BudgetExceeded(f"walltime {run.elapsed():.0f}s > budget {budget}s - reserve {reserve}s")
            t0 = time.perf_counter()
            red = basis_reduction(ctx, basis, mom, method, scope)
            base = fwd(ctx, w, b, reduction=red)
            dx = float((base["x_pre_last"] - base_o["x_pre_last"]).abs().max())
            if dx > 1e-5:
                raise RuntimeError(f"x_rec differs between reduced and original model: {dx}")
            lo, lr = base_o["logits"].cpu(), base["logits"].cpu()
            if not torch.isfinite(lr).all():
                raise RuntimeError(f"non-finite logits for {basis.id} k={basis.k} {method} {scope}")
            kl_v, pd_v = kl(lo, lr).numpy(), prob_diff(lr, J, ctx.r).numpy()
            top1 = (lo.argmax(-1) == lr.argmax(-1)).numpy()
            unp_writer.add([{"basis": basis.id, "family": basis.family,
                             "seed": "" if basis.seed is None else basis.seed, "k": basis.k, "method": method,
                             "scope": scope, "moment_pool": moment_pool_name(method, scope), "pair_id": p["pair_id"],
                             "template": p["template"], "r": p["r"], "kl_orig_to_M": float(kl_v[i]),
                             "pd_M": float(pd_v[i]), "pd_orig": float(pd_orig[i]), "top1_agree": int(top1[i])}
                            for i, p in enumerate(ctx.pairs)])
            uninterv.append({"basis": basis.id, "family": basis.family, "seed": basis.seed, "k": basis.k,
                             "method": method, "scope": scope, "kl": kl_v, "pd_diff": pd_v - pd_orig,
                             "top1_agree": float((lo.argmax(-1) == lr.argmax(-1)).float().mean())})
            for S in ("R", "B", "RB"):
                r = response(fwd(ctx, w, b, phys[S], red), base, J)
                r["T"] = t_score(r["dJ"], ctx.v).numpy()
                ro = resp_o[(basis.id, basis.k, S)] if S != "RB" else resp_o[("any", 0, "RB_natural")]
                e_out = r["dout"] - ro["dout"]
                e_log = centered(r["dlog"] - ro["dlog"])
                vec.append({"basis": basis.id, "k": basis.k, "method": method, "scope": scope, "S": S,
                            "local_rms_err": float(e_out.norm(dim=-1).pow(2).mean().sqrt()),
                            "local_rms_orig": float(ro["dout"].norm(dim=-1).pow(2).mean().sqrt()),
                            "local_mean_err_vec_norm": float(e_out.mean(0).norm()),
                            "logit_rms_err": float(e_log.norm(dim=-1).pow(2).mean().sqrt()),
                            "logit_rms_orig": float(centered(ro["dlog"]).norm(dim=-1).pow(2).mean().sqrt())})
                en, el = e_out.norm(dim=-1).numpy(), e_log.norm(dim=-1).numpy()
                writer.add([{
                    "basis": basis.id, "family": basis.family, "seed": "" if basis.seed is None else basis.seed,
                    "k": basis.k, "method": method, "scope": scope, "moment_pool": moment_pool_name(method, scope),
                    "S": S, "pair_id": p["pair_id"], "template": p["template"], "r": p["r"], "d": p["d"],
                    "T": float(r["T"][i]), "T_orig": float(ro["T"][i]), "e_T": float(r["T"][i] - ro["T"][i]),
                    "local_resp_norm": float(r["dout"][i].norm()), "local_err_norm": float(en[i]),
                    "logit_err_norm": float(el[i])} for i, p in enumerate(ctx.pairs)])
            run.walltimes[f"red_{basis.id}_k{basis.k}_{method}_{scope}"] = time.perf_counter() - t0


# ---------------------------------------------------------------- main run

def setup(ctx, w, b):
    o = orig_runs(ctx, w, b)
    x_rec = o["rec"]["x_pre_last"]
    deltas = {donor: (o[role]["x_pre_last"] - x_rec).cpu().to(F64) for donor, role in DONORS.items()}
    return o, deltas


def run(ctx):
    run_, cfg = ctx.run, ctx.cfg
    Bn, seed = cfg["bootstrap"]["B"], cfg["seed"]
    budget, reserve = cfg["budget"]["walltime_budget_s"], cfg["budget"]["aggregate_reserve_s"]
    b = ctx.batches()
    spec = ctx.spectrum()
    Wc = ctx.mlp.c_fc.weight.detach().cpu().to(F64)
    ctl_w = RowWriter(run_.dir / "control_rows.csv")
    red_w = RowWriter(run_.dir / "reduced_rows.csv")
    unp_w = RowWriter(run_.dir / "unperturbed_rows.csv")
    progress = {"done": [], "status": "running"}
    uninterv, vec = [], []
    state = {}

    def checkpoint(status="running"):
        progress.update(status=status, elapsed_s=run_.elapsed())
        run_.save_json("partial_results.json", {"progress": progress, **state})

    with wrapped_mlp(ctx.model, ctx.layer) as w:
        with run_.timer("orig_forward"):
            o, deltas = setup(ctx, w, b)
        with run_.timer("positive_control_gate"):
            gate, _, _ = positive_control(ctx, w, b, o, Bn, seed)
        state["positive_control"] = gate
        if not gate["valid"]:
            raise RuntimeError(f"positive control invalid: {gate}")
        with run_.timer("bases"):
            x_last, x_all = pool_inputs(ctx, w)
            bases = build_bases(cfg, spec, x_last)
        state["bases"] = {f"{bid}|{k}": bs.meta for (bid, k), bs in bases.items()}
        align = {"natural": np.ones(len(ctx.pairs)),
                 "same_year": indicator_alignment(ctx.r, ctx.d, ctx.r),
                 "wrong_donor": indicator_alignment(ctx.r, ctx.d, np.array([p["wrong_donor_yy"] for p in ctx.pairs]))}
        base_o = o["rec"]
        def guard(cell):
            if run_.elapsed() > budget - reserve:
                ctl_w.close()
                checkpoint("budget_exceeded")
                raise BudgetExceeded(f"walltime {run_.elapsed():.0f}s > budget - reserve before control cell {cell}")

        def cell_done(cell):
            progress["done"].append(f"controls:{cell[0]}|{cell[1]}")
            checkpoint()

        with run_.timer("original_controls"):
            resp_o, match_log = run_controls(ctx, w, b, base_o, deltas, bases, ctl_w, Wc, align, guard, cell_done)
        state["matching"] = match_log
        progress["done"].append("original_controls")
        checkpoint()
        pd_orig = prob_diff(base_o["logits"].cpu(), ctx.J, ctx.r).numpy()
        try:
            for bid in dict.fromkeys(bid for bid, _ in bases):
                t0 = time.perf_counter()
                for k in cfg["ranks"]:
                    basis = bases[(bid, k)]
                    mom = basis_moments(ctx, basis, x_last, x_all)
                    run_reduced_basis(ctx, w, b, base_o, deltas, basis, mom, resp_o, red_w, uninterv, vec, run_,
                                      budget, reserve, pd_orig, unp_w)
                progress["done"].append(bid)
                state["unintervened_partial"] = [_uninterv_summary(u) for u in uninterv]
                checkpoint()
                run_.log.info(f"basis {bid} done in {time.perf_counter() - t0:.1f}s (elapsed {run_.elapsed():.0f}s)")
        except BudgetExceeded:
            ctl_w.close(), red_w.close(), unp_w.close()
            checkpoint("budget_exceeded")
            raise
    ctl_w.close(), red_w.close(), unp_w.close()
    checkpoint("variants_done")
    with run_.timer("aggregate"):
        results = aggregate(ctx, ctl_w.rows, red_w.rows, uninterv, vec, match_log, Bn, seed)
    results.update(state)
    run_.save_json("results.json", results)
    if run_.elapsed() > budget:
        raise BudgetExceeded(f"finished but walltime {run_.elapsed():.0f}s exceeded budget {budget}s")
    return results


def _uninterv_summary(u):
    return {k: v for k, v in u.items() if k not in ("kl", "pd_diff")} | {
        "kl_mean": float(np.mean(u["kl"])), "pd_diff_mean": float(np.mean(u["pd_diff"]))}


# ---------------------------------------------------------------- aggregation

def _col(rows, c):
    return np.array([r[c] for r in rows], dtype=float)


def aggregate(ctx, ctl_rows, red_rows, uninterv, vec, match_log, B, seed):
    out = {"label": ctx.cfg["label"], "n_pairs": len(ctx.pairs),
           "n_template_clusters": int(len(np.unique(ctx.templates))), "n_frames": 2}
    # controls
    groups = {}
    for r in ctl_rows:
        groups.setdefault((r["basis"], r["k"], r["donor"], r["S"], r["regime"]), []).append(r)
    ctl = []
    for (bid, k, donor, S, regime), rs in groups.items():
        valid = [r for r in rs if not r["degenerate"]]
        a = {"basis": bid, "family": rs[0]["family"], "seed": rs[0]["seed"], "k": k, "donor": donor, "S": S,
             "regime": regime, "match_target": rs[0]["match_target"], "n_valid": len(valid),
             "n_degenerate": len(rs) - len(valid), "n_oos": int(sum(r["oos"] for r in rs)),
             "scale_quantiles": scale_quantiles(_col(rs, "scale"), _col(rs, "degenerate").astype(bool))
             if regime != "native" else None,
             "v_alignment_mean": float(_col(rs, "v_alignment").mean())}
        if valid:
            T = _col(valid, "T")
            tm = np.array([r["template"] for r in valid])
            a.update(T_mean_ci=boot(T, "mean", B, seed), T_mean_cluster_ci=boot(T, "mean", B, seed, tm),
                     local_resp_rms=rms(_col(valid, "local_resp_norm")),
                     logit_resp_rms=rms(_col(valid, "logit_resp_norm")),
                     delta_l2_rms=rms(_col(valid, "delta_l2")), delta_preact_rms=rms(_col(valid, "delta_preact")))
            inn = [r for r in valid if not r["oos"]]
            if a["n_oos"] and inn:
                a["T_mean_ci_excl_oos"] = boot(_col(inn, "T"), "mean", B, seed)
        ctl.append(a)
    out["controls"] = ctl
    out["controls_random_seed_spread"] = _seed_spread(ctl, ("k", "donor", "S", "regime"),
                                                      lambda a: a.get("T_mean_ci", [np.nan])[0], "T_mean")
    # original-model interaction per basis/k (native, natural)
    by = {}
    for r in ctl_rows:
        if r["regime"] == "native" and r["donor"] == "natural":
            by[(r["basis"], r["k"], r["S"], r["pair_id"])] = r["T"]
    inter = []
    keys = sorted({(r["basis"], r["k"]) for r in ctl_rows if r["basis"] != "any"}, key=str)
    pids = [p["pair_id"] for p in ctx.pairs]
    for bid, k in keys:
        TR = np.array([by[(bid, k, "R", p)] for p in pids])
        TB = np.array([by[(bid, k, "B", p)] for p in pids])
        TRB = np.array([by[("any", 0, "RB", p)] for p in pids])
        I = TRB - TR - TB
        inter.append({"basis": bid, "k": k, "T_R_mean": float(TR.mean()), "T_B_mean": float(TB.mean()),
                      "T_RB_mean": float(TRB.mean()), "I_T_mean_ci": boot(I, "mean", B, seed),
                      "I_T_mean_cluster_ci": boot(I, "mean", B, seed, np.array([p["template"] for p in ctx.pairs])),
                      "I_T_rms": rms(I)})
    out["interaction_original"] = inter
    out["contrasts"] = contrasts(ctx, ctl_rows, B, seed)
    # reduced
    rg = {}
    for r in red_rows:
        rg.setdefault((r["basis"], r["k"], r["method"], r["scope"], r["S"]), []).append(r)
    vmap = {(v["basis"], v["k"], v["method"], v["scope"], v["S"]): v for v in vec}
    red = []
    for key, rs in rg.items():
        bid, k, method, scope, S = key
        e = _col(rs, "e_T")
        tm = np.array([r["template"] for r in rs])
        v = vmap[key]
        T = _col(rs, "T")
        red.append({"basis": bid, "family": rs[0]["family"], "seed": rs[0]["seed"], "k": k, "method": method,
                    "scope": scope, "S": S, "T_orig_mean": float(_col(rs, "T_orig").mean()),
                    "T_mean_ci": boot(T, "mean", B, seed), "T_mean_cluster_ci": boot(T, "mean", B, seed, tm),
                    "e_T_mean_ci": boot(e, "mean", B, seed), "e_T_mean_cluster_ci": boot(e, "mean", B, seed, tm),
                    "e_T_rms_ci": boot(e, "rms", B, seed), "e_T_rms_cluster_ci": boot(e, "rms", B, seed, tm),
                    "local_err_ratio": v["local_rms_err"] / v["local_rms_orig"],
                    "local_mean_err_vec_norm": v["local_mean_err_vec_norm"],
                    "logit_err_ratio": v["logit_rms_err"] / v["logit_rms_orig"]})
    out["reduced"] = red
    out["reduced_random_seed_spread"] = _seed_spread(red, ("k", "method", "scope", "S"),
                                                     lambda a: a["e_T_rms_ci"][0], "e_T_rms")
    # signed decomposition e_RB = e_R - C, C = T_RB^orig - T_R^orig = T_B^orig + I^orig
    dec = []
    for (bid, k, method, scope, S), rs in rg.items():
        if S != "R":
            continue
        rR = {r["pair_id"]: r for r in rs}
        rRB = {r["pair_id"]: r for r in rg[(bid, k, method, scope, "RB")]}
        eR = np.array([rR[p]["e_T"] for p in pids])
        eRB = np.array([rRB[p]["e_T"] for p in pids])
        C = np.array([rRB[p]["T_orig"] - rR[p]["T_orig"] for p in pids])
        tm = np.array([r["template"] for r in rs])
        dec.append({"basis": bid, "k": k, "method": method, "scope": scope,
                    "identity_max_residual": float(np.abs(eRB - (eR - C)).max()),
                    "e_R_mean_ci": boot(eR, "mean", B, seed), "e_R_mean_cluster_ci": boot(eR, "mean", B, seed, tm),
                    "e_RB_mean_ci": boot(eRB, "mean", B, seed),
                    "e_RB_mean_cluster_ci": boot(eRB, "mean", B, seed, tm),
                    "e_R_rms_ci": boot(eR, "rms", B, seed), "e_RB_rms_ci": boot(eRB, "rms", B, seed),
                    "MSE_R": float(np.mean(eR ** 2)), "cross_minus2_E_C_eR": float(-2 * np.mean(C * eR)),
                    "E_C2": float(np.mean(C ** 2)), "MSE_RB": float(np.mean(eRB ** 2)), "C_mean": float(C.mean())})
    out["decomposition"] = dec
    tmpl = np.array([p["template"] for p in ctx.pairs])
    out["unintervened"] = [{**_uninterv_summary(u), "kl_mean_ci": boot(u["kl"], "mean", B, seed),
                            "kl_mean_cluster_ci": boot(u["kl"], "mean", B, seed, tmpl),
                            "pd_diff_mean_ci": boot(u["pd_diff"], "mean", B, seed),
                            "pd_diff_mean_cluster_ci": boot(u["pd_diff"], "mean", B, seed, tmpl)} for u in uninterv]
    out["matching_summary"] = {"max_rel_err_f64": max(m["max_rel_err_f64"] for m in match_log),
                               "max_rel_err_f32": max(m["max_rel_err_f32"] for m in match_log),
                               "n_degenerate_total": sum(m["n_degenerate"] for m in match_log),
                               "n_oos_total": sum(m["n_oos"] for m in match_log)}
    return out


def paired_contrast(lhs, rhs, pids, templates, B, seed):
    """Paired mean difference lhs - rhs over a common mask (both sides non-degenerate); oos-filtered variant
    removes pairs flagged oos on either side. lhs/rhs: {pair_id: row}."""
    out = {}
    for name, keep in (("all", lambda r: not r["degenerate"]),
                       ("excl_oos", lambda r: not r["degenerate"] and not r["oos"])):
        idx = [i for i, p in enumerate(pids) if keep(lhs[p]) and keep(rhs[p])]
        if not idx:
            out[name] = {"n": 0}
            continue
        dlt = np.array([lhs[pids[i]]["T"] - rhs[pids[i]]["T"] for i in idx])
        tm = np.array([templates[i] for i in idx])
        out[name] = {"n": len(idx), "diff_mean_ci": boot(dlt, "mean", B, seed),
                     "diff_mean_cluster_ci": boot(dlt, "mean", B, seed, tm),
                     "lhs_mean": float(np.mean([lhs[pids[i]]["T"] for i in idx])),
                     "rhs_mean": float(np.mean([rhs[pids[i]]["T"] for i in idx]))}
    return out


def contrasts(ctx, ctl_rows, B, seed):
    """PROTOCOL_DAY2 §C: (1) cross-basis: basis natural (matched to SVD / native) - SVD natural native;
    (2) donor-control: basis natural native - donor control (matched to that basis's natural / native)."""
    pids = [p["pair_id"] for p in ctx.pairs]
    templates = [p["template"] for p in ctx.pairs]
    idx = {}
    for r in ctl_rows:
        idx.setdefault((r["basis"], r["k"], r["donor"], r["S"], r["regime"]), {})[r["pair_id"]] = r
    out = []
    for (bid, k, donor, S, regime), lhs in idx.items():
        if bid == "any":
            continue
        if donor == "natural" and bid != "svd":
            rhs = idx[("svd", k, "natural", S, "native")]
            out.append({"contrast": "cross_basis", "basis": bid, "k": k, "S": S, "regime": regime,
                        "lhs": f"{bid}/natural/{regime}" + ("/matched_to_svd_natural" if regime != "native" else ""),
                        "rhs": "svd/natural/native", **paired_contrast(lhs, rhs, pids, templates, B, seed)})
        if donor != "natural":
            nat = idx[(bid, k, "natural", S, "native")]
            out.append({"contrast": "donor_control", "basis": bid, "k": k, "S": S, "regime": regime,
                        "donor": donor, "lhs": f"{bid}/natural/native",
                        "rhs": f"{bid}/{donor}/{regime}" + ("/matched_to_basis_natural" if regime != "native" else ""),
                        **paired_contrast(nat, lhs, pids, templates, B, seed)})
    for donor in ("same_year", "wrong_donor"):  # basis-independent RB
        nat = idx[("any", 0, "natural", "RB", "native")]
        for regime in REGIMES:
            out.append({"contrast": "donor_control", "basis": "any", "k": 0, "S": "RB", "regime": regime,
                        "donor": donor, "lhs": "natural/RB/native", "rhs": f"{donor}/RB/{regime}",
                        **paired_contrast(nat, idx[("any", 0, donor, "RB", regime)], pids, templates, B, seed)})
    return out


def _seed_spread(aggs, keys, stat, name):
    groups = {}
    for a in aggs:
        if a["family"] == "random":
            groups.setdefault(tuple(a[k] for k in keys), []).append((a["seed"], stat(a)))
    out = []
    for key, vals in groups.items():
        x = np.array([v for _, v in vals], dtype=float)
        out.append({**dict(zip(keys, key)), "statistic": name, "per_seed": {str(s): float(v) for s, v in vals},
                    "mean": float(np.nanmean(x)), "sd": float(np.nanstd(x, ddof=1)) if len(x) > 1 else None,
                    "min": float(np.nanmin(x)), "max": float(np.nanmax(x)), "n_seeds": len(x)})
    return out


# ---------------------------------------------------------------- smoke (Day-1 replay + basis sanity)

def smoke(ctx):
    run_, cfg = ctx.run, ctx.cfg
    Bn, seed = cfg["bootstrap"]["B"], cfg["seed"]
    b = ctx.batches()
    spec = ctx.spectrum()
    Wc = ctx.mlp.c_fc.weight.detach().cpu().to(F64)
    out = {}
    with wrapped_mlp(ctx.model, ctx.layer) as w:
        o, deltas = setup(ctx, w, b)
        gate, _, _ = positive_control(ctx, w, b, o, Bn, seed)
        out["positive_control_valid"] = gate["valid"]
        out["T_RB_natural_mean_ci"] = gate["T_RB_natural_mean_ci"]
        x_last, x_all = pool_inputs(ctx, w)
        bases = build_bases(cfg, spec, x_last)
        out["bases"] = {f"{bid}|{k}": bs.meta for (bid, k), bs in bases.items()}
        # Day-1 replay: SVD k=192
        day1 = {}
        with open(ROOT / cfg["day1"]["pilot_run"] / "per_pair_rows.csv") as f:
            for r in csv.DictReader(f):
                if r["k"] == "192" and r["donor"] == "natural":
                    day1[(r["model"], r["scope"], r["S"], r["pair_id"])] = float(r["T"])
        svd = bases[("svd", 192)]
        R_row, B_row = svd.row_maps()
        mom = basis_moments(ctx, svd, x_last, x_all)
        phys = {"R": (deltas["natural"] @ R_row).float(), "B": (deltas["natural"] @ B_row).float(),
                "RB": deltas["natural"].float()}
        replay = {}
        for S, dl in phys.items():
            T = t_score(response(fwd(ctx, w, b, dl), o["rec"], ctx.J)["dJ"], ctx.v).numpy()
            replay[f"orig_{S}"] = max(abs(T[i] - day1[("orig", "none", S, p["pair_id"])])
                                      for i, p in enumerate(ctx.pairs))
        for method, scope in (("bias", "last_token"), ("gauss", "last_token"), ("zero", "full_layer"),
                              ("gauss", "full_layer")):
            red = basis_reduction(ctx, svd, mom, method, scope)
            base = fwd(ctx, w, b, reduction=red)
            for S in ("R", "RB"):
                T = t_score(response(fwd(ctx, w, b, phys[S], red), base, ctx.J)["dJ"], ctx.v).numpy()
                replay[f"{method}_{scope}_{S}"] = max(abs(T[i] - day1[(method, scope, S, p["pair_id"])])
                                                      for i, p in enumerate(ctx.pairs))
        out["day1_replay_max_abs_T_diff"] = replay
        # matching sanity (pca k=192, natural & wrong donor, both regimes)
        pca = bases[("pca", 192)]
        tg = control_targets(deltas, {("svd", 192): svd, ("pca", 192): pca}, Wc)
        ms = {}
        for donor, key in (("natural", ("svd", 192, "R")), ("wrong_donor", ("pca", 192, "R"))):
            for regime in ("l2", "preact"):
                _, info = apply_match(deltas[donor] @ pca.row_maps()[0], regime, tg[key][regime], cfg["matching"], Wc)
                ms[f"{donor}_{regime}"] = {"max_rel_err_f32": info["max_rel_err_f32"],
                                           "scale_quantiles": scale_quantiles(info["scale"], info["degenerate"]),
                                           "n_oos": int(info["oos"].sum()), "n_degenerate": int(info["degenerate"].sum())}
        out["matching_sanity"] = ms
        # whitened reduced-model B null and RB = donor
        ws = bases[("wsvd", 192)]
        momw = basis_moments(ctx, ws, x_last, x_all)
        red = basis_reduction(ctx, ws, momw, "gauss", "full_layer")
        base = fwd(ctx, w, b, reduction=red)
        dB = (deltas["natural"] @ ws.row_maps()[1]).float()
        pat = fwd(ctx, w, b, dB, red)
        out["wsvd_reduced_B_null_max_abs_logit"] = float((pat["logits"] - base["logits"]).abs().max())
        full = fwd(ctx, w, b, deltas["natural"].float())
        out["RB_patch_vs_donor_mlp_out_max_abs"] = float((full["out_last"] - o["don"]["out_last"]).abs().max())
    gates = cfg["smoke_gates"]
    fails = []
    if not out["positive_control_valid"]:
        fails.append("positive control invalid")
    for name, v in replay.items():
        tol = gates["replay_orig_max_abs_T"] if name.startswith("orig_") else gates["replay_reduced_max_abs_T"]
        if not v <= tol:
            fails.append(f"Day-1 replay {name}: {v:.3g} > {tol}")
    if not out["wsvd_reduced_B_null_max_abs_logit"] <= gates["reduced_B_null_max_abs_logit"]:
        fails.append(f"wsvd B null {out['wsvd_reduced_B_null_max_abs_logit']:.3g}")
    if not out["RB_patch_vs_donor_mlp_out_max_abs"] <= gates["RB_vs_donor_max_abs"]:
        fails.append(f"RB vs donor {out['RB_patch_vs_donor_mlp_out_max_abs']:.3g}")
    for name, m in ms.items():
        if m["max_rel_err_f32"] > cfg["matching"]["assert_rel_tol_f32"]:
            fails.append(f"matching {name}")
    out["gate_failures"] = fails
    run_.save_json("smoke.json", out)
    if fails:
        raise AssertionError(f"Day-2 smoke gates failed: {fails}")
    return out
