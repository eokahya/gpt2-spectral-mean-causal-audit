"""Pipeline stages: smoke (behaviour + positive control + parity), benchmark, exploratory pilot."""
import csv
import json
import time

import numpy as np
import torch

from . import ROOT
from .common import get_device, memory_snapshot, set_seed, sync
from .data import load_pilot, make_batch
from .metrics import bootstrap_ci, centered, kl, prob_diff, rms, t_score, v_rd
from .model import forward_last, load_model, make_reduction, model_file_info, wrapped_mlp
from .spectral import calibrated_moments, svd_w_in, weights_only_moments

ROLES = ("rec", "don", "same_year", "wrong_donor")
DONORS = {"natural": "don", "same_year": "same_year", "wrong_donor": "wrong_donor"}
SUBSPACES = ("R", "B", "RB")


def moment_pool_name(method, scope):
    """Which moments a variant uses (PROTOCOL §3/§4): scopes use different calibrated pools."""
    if method in ("orig", "zero"):
        return "none"
    if method == "gauss_wo":
        return "weights_only"
    return "final_token" if scope == "last_token" else "all_token"


class BudgetExceeded(RuntimeError):
    pass


class RowWriter:
    """Appends per-pair rows to CSV as they are produced (checkpoint), and keeps them for aggregation."""

    def __init__(self, path):
        self.path, self.rows, self._w, self._f = path, [], None, None

    def add(self, rows):
        if self._w is None:
            self._f = open(self.path, "w", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=list(rows[0].keys()))
            self._w.writeheader()
        self._w.writerows(rows)
        self._f.flush()
        self.rows += rows

    def close(self):
        if self._f:
            self._f.close()


class Context:
    def __init__(self, cfg, run, layer=None, device=None):
        self.cfg = cfg
        self.run = run
        self.layer = cfg["layer"] if layer is None else layer
        self.device = get_device(device or cfg["model"]["device"])
        set_seed(cfg["seed"])
        with run.timer("load_model"):
            self.model, self.tok = load_model(cfg, self.device)
        run.manifest["model"].update(model_file_info(cfg))
        run.manifest["layer"] = self.layer
        run.manifest["device_used"] = str(self.device)
        self.meta, self.pairs, self.moment = load_pilot()
        self.J = self.meta["J_token_ids"]
        self.pad = self.tok.eos_token_id
        self.mlp = self.model.transformer.h[self.layer].mlp
        self.r = np.array([p["r"] for p in self.pairs])
        self.d = np.array([p["d"] for p in self.pairs])
        self.v = v_rd(self.r, self.d)
        self.templates = np.array([p["template"] for p in self.pairs])

    def batches(self, idx=None):
        rows = self.pairs if idx is None else [self.pairs[i] for i in idx]
        return {role: make_batch([p[f"{role}_ids"] for p in rows], self.pad, self.device) for role in ROLES}

    def spectrum(self):
        with self.run.timer("svd_cpu_float64"):
            return svd_w_in(self.mlp.c_fc.weight)

    def moment_activations(self, wrapper):
        """Original-model MLP inputs on the GT-moment pool: final-token rows and all real-position rows."""
        bs = self.cfg["budget"]["batch_size"]
        last, allpos = [], []
        for s in range(0, len(self.moment), bs):
            rows = self.moment[s:s + bs]
            b = make_batch([p["ids"] for p in rows], self.pad, self.device)
            out = forward_last(self.model, wrapper, b, capture_all=True)
            last.append(out["x_pre_last"].cpu())
            allpos.append(out["x_all"][b.mask.bool()].cpu())
        return torch.cat(last), torch.cat(allpos)

    def moments(self, spec, k, x_last, x_all):
        P_R, P_B = spec.projectors(k)
        Wc = self.mlp.c_fc.weight.detach().cpu()
        ln = self.model.transformer.h[self.layer].ln_2
        return P_R, P_B, {
            "final_token": calibrated_moments(x_last, P_B, Wc),
            "all_token": calibrated_moments(x_all, P_B, Wc),
            "weights_only": weights_only_moments(P_B, Wc, ln.weight, ln.bias),
        }

    def reduction(self, P_R, mom, k, method, scope):
        mu, sigma = mom[moment_pool_name(method, scope)] if method != "zero" else mom["final_token"]
        return make_reduction(P_R, self.mlp, mu, sigma, method, scope, self.cfg["quadrature_nodes"],
                              self.cfg["quadrature_sigma_max_checked"], self.device)


def response(res, base, J):
    dlog = (res["logits"] - base["logits"]).cpu()
    return {"dlog": dlog, "dJ": dlog[:, J], "dout": (res["out_last"] - base["out_last"]).cpu()}


def orig_runs(ctx, w, b):
    return {role: forward_last(ctx.model, w, b[role]) for role in ROLES}


def positive_control(ctx, w, b, o, B, seed):
    """Validity gate (PROTOCOL §5) from original-model runs on the calibration recipients/donors."""
    x_rec = o["rec"]["x_pre_last"]
    pd = prob_diff(o["rec"]["logits"].cpu(), ctx.J, ctx.r).numpy()
    T = {}
    for donor, role in DONORS.items():
        res = forward_last(ctx.model, w, b["rec"], delta=o[role]["x_pre_last"] - x_rec)
        T[donor] = t_score(response(res, o["rec"], ctx.J)["dJ"], ctx.v).numpy()
    pd_ci = bootstrap_ci(np.mean, [pd], B, seed)
    t_ci = bootstrap_ci(np.mean, [T["natural"]], B, seed)
    diff_ci = bootstrap_ci(lambda a, b: np.mean(a - b), [T["natural"], T["same_year"]], B, seed)
    pc = ctx.cfg["positive_control"]
    valid = pd_ci[0] > pc["min_mean_pd"] and t_ci[1] > pc["ci_lower_T_gt"] and diff_ci[1] > 0
    return {
        "valid": bool(valid),
        "rule": "mean PD > 0 AND CI_lo(mean T_RB natural) > 0 AND CI_lo(mean[T_nat - T_same_year]) > 0",
        "pd_mean_ci": pd_ci, "pd_frac_positive": float(np.mean(pd > 0)),
        "T_RB_natural_mean_ci": t_ci,
        "T_RB_same_year_mean_ci": bootstrap_ci(np.mean, [T["same_year"]], B, seed),
        "T_RB_wrong_donor_mean_ci": bootstrap_ci(np.mean, [T["wrong_donor"]], B, seed),
        "T_natural_minus_same_year_ci": diff_ci,
        "T_RB_natural_cluster_ci": bootstrap_ci(np.mean, [T["natural"]], B, seed, clusters=ctx.templates),
    }, pd, T


def smoke(ctx):
    run, cfg = ctx.run, ctx.cfg
    B, seed = cfg["bootstrap"]["B"], cfg["seed"]
    out = {}
    b = ctx.batches()
    with wrapped_mlp(ctx.model, ctx.layer) as w:
        with run.timer("orig_forward"):
            o = orig_runs(ctx, w, b)
        with run.timer("positive_control"):
            out["positive_control"], pd, T = positive_control(ctx, w, b, o, B, seed)
        x_rec = o["rec"]["x_pre_last"]
        same = forward_last(ctx.model, w, b["rec"], delta=torch.zeros_like(x_rec))
        full = forward_last(ctx.model, w, b["rec"], delta=o["don"]["x_pre_last"] - x_rec)
    # Behaviour: top-1 tokens on the first prompts, fraction of top-1 in J.
    top1 = o["rec"]["logits"].argmax(-1).cpu()
    out["top1_examples"] = [{"prompt": ctx.pairs[i]["rec_text"], "top1": ctx.tok.decode([int(top1[i])])}
                            for i in range(8)]
    out["frac_top1_in_J"] = float(np.isin(top1.numpy(), ctx.J).mean())
    out["frac_top1_year_gt_r"] = float(np.mean([
        (int(t) in ctx.J and ctx.J.index(int(t)) > r) for t, r in zip(top1.tolist(), ctx.r)]))
    out["identity_same_example_max_abs_logit_diff"] = float((same["logits"] - o["rec"]["logits"]).abs().max())
    out["RB_patch_vs_donor_mlp_out_max_abs_diff"] = float((full["out_last"] - o["don"]["out_last"]).abs().max())
    out["RB_patch_vs_donor_mlp_out_rel"] = float((full["out_last"] - o["don"]["out_last"]).norm()
                                                 / o["don"]["out_last"].norm())
    with run.timer("mps_cpu_parity"):
        out["mps_cpu_parity"] = parity(ctx)
    run.save_json("smoke.json", out)
    np.savez(run.dir / "smoke_per_pair.npz", pd=pd, **{f"T_RB_{k}": v for k, v in T.items()})
    return out


def parity(ctx, n=8, k=192):
    """Small MPS-vs-CPU logits comparison for the original model and one reduced model."""
    if ctx.device.type != "mps":
        return {"skipped": "device is not mps"}
    from .model import load_model as _load
    cpu = torch.device("cpu")
    model_cpu, _ = _load(ctx.cfg, cpu)
    idx = list(range(n))
    rows = [ctx.pairs[i]["rec_ids"] for i in idx]
    spec = svd_w_in(ctx.mlp.c_fc.weight)
    P_R, P_B = spec.projectors(k)
    mu, sigma = weights_only_moments(P_B, ctx.mlp.c_fc.weight.cpu(), ctx.model.transformer.h[ctx.layer].ln_2.weight,
                                     ctx.model.transformer.h[ctx.layer].ln_2.bias)
    res = {}
    for name, model, dev in (("mps", ctx.model, ctx.device), ("cpu", model_cpu, cpu)):
        b = make_batch(rows, ctx.pad, dev)
        mlp = model.transformer.h[ctx.layer].mlp
        red = make_reduction(P_R, mlp, mu, sigma, "gauss", "full_layer", ctx.cfg["quadrature_nodes"],
                             ctx.cfg["quadrature_sigma_max_checked"], dev)
        with wrapped_mlp(model, ctx.layer) as w:
            res[name] = (forward_last(model, w, b)["logits"].cpu(),
                         forward_last(model, w, b, reduction=red)["logits"].cpu())
    del model_cpu
    return {"orig_max_abs_logit_diff": float((res["mps"][0] - res["cpu"][0]).abs().max()),
            "reduced_gauss_full_layer_k192_max_abs_logit_diff": float((res["mps"][1] - res["cpu"][1]).abs().max()),
            "n_prompts": n}


def bench(ctx, sizes, k=192, reps=5):
    run = ctx.run
    spec = ctx.spectrum()
    out = {"k": k, "reps": reps, "sizes": {}}
    with wrapped_mlp(ctx.model, ctx.layer) as w:
        x_last, x_all = ctx.moment_activations(w)
        P_R, _, mom = ctx.moments(spec, k, x_last, x_all)
        for n in sizes:
            b = ctx.batches(range(n))
            base = forward_last(ctx.model, w, b["rec"])
            delta = forward_last(ctx.model, w, b["don"])["x_pre_last"] - base["x_pre_last"]
            cases = {
                "orig_baseline": dict(),
                "orig_patched": dict(delta=delta),
                "gauss_full_layer_patched": dict(delta=delta, reduction=ctx.reduction(P_R, mom, k, "gauss", "full_layer")),
                "gauss_last_token_patched": dict(delta=delta, reduction=ctx.reduction(P_R, mom, k, "gauss", "last_token")),
            }
            res = {}
            for name, kw in cases.items():
                for _ in range(2):
                    forward_last(ctx.model, w, b["rec"], **kw)
                sync(ctx.device)
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    forward_last(ctx.model, w, b["rec"], **kw)
                    sync(ctx.device)
                    ts.append(time.perf_counter() - t0)
                res[name] = {"mean_ms": 1e3 * float(np.mean(ts)), "min_ms": 1e3 * float(np.min(ts))}
            res["memory"] = memory_snapshot(ctx.device)
            res["seq_len_max"] = int(b["rec"].ids.shape[1])
            out["sizes"][n] = res
            run.log.info(f"bench n={n}: {json.dumps(res)}")
    run.save_json("bench.json", out)
    return out


def pilot(ctx):
    """Exploratory pilot on the calibration split: all ranks x methods x scopes x donors x subspaces."""
    run, cfg = ctx.run, ctx.cfg
    B, seed, J = cfg["bootstrap"]["B"], cfg["seed"], ctx.J
    b = ctx.batches()
    spec = ctx.spectrum()
    run.manifest["svd_singular_values_top5"] = spec.S[:5].tolist()
    vec_aggs, uninterv, interaction, checks = [], [], [], {}
    budget = cfg["budget"]["walltime_budget_s"]
    reserve = cfg["budget"]["aggregate_reserve_s"]
    writer = RowWriter(run.dir / "per_pair_rows.csv")
    progress = {"completed_variants": [], "status": "running"}

    def checkpoint(status="running"):
        progress.update(status=status, elapsed_s=run.elapsed(), budget_s=budget)
        run.save_json("partial_results.json", {"progress": progress, "positive_control": gate, "checks": checks,
                                               "unintervened": uninterv, "interaction_original": interaction,
                                               "vector_aggregates": vec_aggs})

    with wrapped_mlp(ctx.model, ctx.layer) as w:
        with run.timer("orig_forward"):
            o = orig_runs(ctx, w, b)
        with run.timer("positive_control_gate"):
            gate, _, _ = positive_control(ctx, w, b, o, B, seed)
        run.save_json("positive_control.json", gate)
        if not gate["valid"]:
            raise RuntimeError(f"positive control invalid; no measurements produced: {gate}")
        with run.timer("moment_pool_forward"):
            x_mom_last, x_mom_all = ctx.moment_activations(w)
        checks["moment_pool_rows"] = {"final_token": len(x_mom_last), "all_token": len(x_mom_all)}

        base_o = o["rec"]
        x_rec = base_o["x_pre_last"]
        raw = {donor: (o[role]["x_pre_last"] - x_rec) for donor, role in DONORS.items()}
        pd_orig = prob_diff(base_o["logits"].cpu(), J, ctx.r).numpy()
        cut = {}
        for k in cfg["ranks"]:
            P_R, P_B, mom = ctx.moments(spec, k, x_mom_last, x_mom_all)
            deltas = {}
            for donor, d in raw.items():
                d64 = d.cpu().double()
                deltas[(donor, "R")] = (d64 @ P_R).float().to(ctx.device)
                deltas[(donor, "B")] = (d64 @ P_B).float().to(ctx.device)
                deltas[(donor, "RB")] = d  # full difference: x' = x_don exactly (positive control)
            cut[k] = (P_R, P_B, mom, deltas)
            run.manifest.setdefault("retained_energy", {})[k] = spec.retained_energy(k)

        # ---- original model responses (each reduced response is compared with its own counterpart) ----
        resp_o, energy = {}, {}
        Wc64 = ctx.mlp.c_fc.weight.detach().cpu().double()
        with run.timer("orig_patched"):
            for k, (P_R, P_B, mom, deltas) in cut.items():
                for (donor, S), dl in deltas.items():
                    d64 = dl.cpu().double()
                    energy[(k, donor, S)] = {
                        "delta_norm": d64.norm(dim=-1).numpy(),
                        "preact_energy_orig": (d64 @ Wc64).norm(dim=-1).numpy(),
                        "preact_energy_reduced": ((d64 @ P_R) @ Wc64).norm(dim=-1).numpy(),
                    }
                    if S == "RB" and ("RB", donor) in resp_o:  # RB delta does not depend on k
                        resp_o[(k, donor, S)] = resp_o[("RB", donor)]
                        continue
                    r = response(forward_last(ctx.model, w, b["rec"], delta=dl), base_o, J)
                    r["T"] = t_score(r["dJ"], ctx.v).numpy()
                    resp_o[(k, donor, S)] = r
                    if S == "RB":
                        resp_o[("RB", donor)] = r
            for k in cut:
                for donor in DONORS:
                    R, Bb, RB = (resp_o[(k, donor, S)] for S in SUBSPACES)
                    I_T = RB["T"] - R["T"] - Bb["T"]
                    I_out = (RB["dout"] - R["dout"] - Bb["dout"]).norm(dim=-1).numpy()
                    interaction.append({
                        "k": k, "donor": donor,
                        "I_T_mean_ci": bootstrap_ci(np.mean, [I_T], B, seed),
                        "I_T_rms": rms(I_T),
                        "I_local_norm_rms": rms(I_out),
                        "RB_local_norm_rms": rms(RB["dout"].norm(dim=-1).numpy()),
                        "T_R_mean": float(np.mean(R["T"])), "T_B_mean": float(np.mean(Bb["T"])),
                        "T_RB_mean": float(np.mean(RB["T"])),
                    })
                    for S, r in zip(SUBSPACES, (R, Bb, RB)):
                        writer.add(_rows(ctx, "orig", "none", k, donor, S, r, r, energy[(k, donor, S)]))
        same = forward_last(ctx.model, w, b["rec"], delta=torch.zeros_like(x_rec))
        checks["same_example_max_abs_logit_diff"] = float((same["logits"] - base_o["logits"]).abs().max())
        checkpoint()

        # ---- reduced models ----
        x_rec_dev = []
        for k, (P_R, P_B, mom, deltas) in cut.items():
            for method in cfg["methods"]:
                for scope in cfg["scopes"]:
                    if run.elapsed() > budget - reserve:
                        checks["x_rec_reduced_vs_orig_max_abs"] = max(x_rec_dev, default=None)
                        checkpoint("budget_exceeded")
                        writer.close()
                        raise BudgetExceeded(f"walltime {run.elapsed():.0f}s > budget {budget}s - reserve {reserve}s; "
                                             f"partial rows/JSON kept")
                    t0 = time.perf_counter()
                    red = ctx.reduction(P_R, mom, k, method, scope)
                    base = forward_last(ctx.model, w, b["rec"], reduction=red)
                    x_rec_dev.append(float((base["x_pre_last"] - x_rec).abs().max()))
                    if x_rec_dev[-1] > 1e-5:  # upstream of block L is unchanged, so x_rec must be identical
                        raise RuntimeError(f"x_rec differs between reduced and original model: {x_rec_dev[-1]}")
                    lo, lr = base_o["logits"].cpu(), base["logits"].cpu()
                    kl_v = kl(lo, lr).numpy()
                    pd_v = prob_diff(lr, J, ctx.r).numpy()
                    uninterv.append({
                        "k": k, "method": method, "scope": scope,
                        "KL_mean_ci": bootstrap_ci(np.mean, [kl_v], B, seed),
                        "KL_median": float(np.median(kl_v)),
                        "PD_mean": float(pd_v.mean()), "PD_orig_mean": float(pd_orig.mean()),
                        "PD_diff_mean_ci": bootstrap_ci(np.mean, [pd_v - pd_orig], B, seed),
                        "top1_agree": float((lo.argmax(-1) == lr.argmax(-1)).float().mean()),
                    })
                    for (donor, S), dl in deltas.items():
                        r = response(forward_last(ctx.model, w, b["rec"], delta=dl, reduction=red), base, J)
                        r["T"] = t_score(r["dJ"], ctx.v).numpy()
                        ro = resp_o[(k, donor, S)]
                        writer.add(_rows(ctx, method, scope, k, donor, S, r, ro, energy[(k, donor, S)]))
                        e_out = r["dout"] - ro["dout"]
                        e_log = centered(r["dlog"] - ro["dlog"])
                        vec_aggs.append({
                            "k": k, "method": method, "scope": scope, "donor": donor, "S": S,
                            "local_mean_error_vec_norm": float(e_out.mean(0).norm()),
                            "local_rms_error_norm": float(e_out.norm(dim=-1).pow(2).mean().sqrt()),
                            "local_rms_orig_response_norm": float(ro["dout"].norm(dim=-1).pow(2).mean().sqrt()),
                            "logit_mean_error_vec_norm": float(e_log.mean(0).norm()),
                            "logit_rms_error_norm": float(e_log.norm(dim=-1).pow(2).mean().sqrt()),
                            "logit_rms_orig_response_norm": float(
                                centered(ro["dlog"]).norm(dim=-1).pow(2).mean().sqrt()),
                        })
                    dt_v = time.perf_counter() - t0
                    run.walltimes[f"variant_k{k}_{method}_{scope}"] = dt_v
                    progress["completed_variants"].append({"k": k, "method": method, "scope": scope, "walltime_s": dt_v})
                    checkpoint()
                    run.log.info(f"k={k} {method:8s} {scope:10s} done in {dt_v:.2f}s (elapsed {run.elapsed():.0f}s)")
        checks["x_rec_reduced_vs_orig_max_abs"] = max(x_rec_dev)

    writer.close()
    rows = writer.rows
    checkpoint("variants_done")
    with run.timer("aggregate"):
        aggs = aggregate(ctx, rows, B, seed)
        primary = primary_candidate(ctx, rows, B, seed)
    results = {
        "label": cfg["label"], "split": "calibration", "n_pairs": len(ctx.pairs),
        "n_template_clusters": int(len(np.unique(ctx.templates))),
        "positive_control": gate, "checks": checks,
        "retained_energy": run.manifest["retained_energy"],
        "unintervened": uninterv, "interaction_original": interaction,
        "vector_aggregates": vec_aggs, "scalar_aggregates": aggs,
        "primary_candidate_exploratory": primary,
        "epsilon_reference_scales": epsilon_scales(ctx, rows),
    }
    run.save_json("results.json", results)
    if run.elapsed() > budget:
        raise BudgetExceeded(f"finished but walltime {run.elapsed():.0f}s exceeded budget {budget}s")
    return results


def _rows(ctx, method, scope, k, donor, S, r, ro, en):
    T, To = r["T"], ro["T"]
    e_out = (r["dout"] - ro["dout"]).norm(dim=-1).numpy()
    e_log = centered(r["dlog"] - ro["dlog"]).norm(dim=-1).numpy()
    e_J = centered(r["dJ"] - ro["dJ"]).norm(dim=-1).numpy()
    out_n = r["dout"].norm(dim=-1).numpy()
    log_n = centered(r["dlog"]).norm(dim=-1).numpy()
    pool = moment_pool_name(method, scope)
    rows = []
    for i, p in enumerate(ctx.pairs):
        rows.append({
            "pair_id": p["pair_id"], "template": p["template"], "r": p["r"], "d": p["d"],
            "direction": p["direction"], "diff_bin": p["diff_bin"],
            "model": method, "scope": scope, "moment_pool": pool, "k": k, "donor": donor, "S": S,
            "T": float(T[i]), "T_orig": float(To[i]), "e_T": float(T[i] - To[i]),
            "local_resp_norm": float(out_n[i]), "local_err_norm": float(e_out[i]),
            "logit_resp_norm": float(log_n[i]), "logit_err_norm": float(e_log[i]), "J_err_norm": float(e_J[i]),
            "delta_norm": float(en["delta_norm"][i]), "preact_energy_orig": float(en["preact_energy_orig"][i]),
            "preact_energy_reduced": float(en["preact_energy_reduced"][i]),
        })
    return rows


def _group(rows):
    g = {}
    for r in rows:
        g.setdefault((r["model"], r["scope"], r["k"], r["donor"], r["S"]), []).append(r)
    return g


def aggregate(ctx, rows, B, seed):
    out = []
    for (model, scope, k, donor, S), rs in _group(rows).items():
        col = lambda c: np.array([r[c] for r in rs])
        cl = col("template")
        a = {"model": model, "scope": scope, "k": k, "donor": donor, "S": S, "n": len(rs),
             "T_mean_ci": bootstrap_ci(np.mean, [col("T")], B, seed),
             "T_orig_mean": float(col("T_orig").mean()),
             "local_resp_norm_rms": rms(col("local_resp_norm")),
             "logit_resp_norm_rms": rms(col("logit_resp_norm")),
             "delta_norm_rms": rms(col("delta_norm")),
             "preact_energy_orig_rms": rms(col("preact_energy_orig")),
             "preact_energy_reduced_rms": rms(col("preact_energy_reduced"))}
        if model != "orig":
            a["e_T_mean_ci"] = bootstrap_ci(np.mean, [col("e_T")], B, seed)
            a["e_T_rms_ci"] = bootstrap_ci(rms, [col("e_T")], B, seed)
            a["e_T_rms_cluster_ci"] = bootstrap_ci(rms, [col("e_T")], B, seed, clusters=cl)
            a["local_err_rms"] = rms(col("local_err_norm"))
            a["logit_err_rms"] = rms(col("logit_err_norm"))
            a["J_err_rms"] = rms(col("J_err_norm"))
        out.append(a)
    return out


def primary_candidate(ctx, rows, B, seed):
    """EXPLORATORY view of the candidate primary comparison: gauss vs bias, same k, natural donor, S=RB.

    Not confirmatory: calibration split, epsilon not frozen."""
    g = _group(rows)
    out = []
    for scope in ctx.cfg["scopes"]:
        for k in ctx.cfg["ranks"]:
            ga, bi = g[("gauss", scope, k, "natural", "RB")], g[("bias", scope, k, "natural", "RB")]
            assert [r["pair_id"] for r in ga] == [r["pair_id"] for r in bi]
            eg, eb = np.array([r["e_T"] for r in ga]), np.array([r["e_T"] for r in bi])
            cl = np.array([r["template"] for r in ga])
            diff = lambda a, b: rms(a) - rms(b)
            out.append({
                "scope": scope, "k": k, "primary_scope_candidate": scope == "last_token",
                "rms_e_T_gauss_ci": bootstrap_ci(rms, [eg], B, seed),
                "rms_e_T_bias_ci": bootstrap_ci(rms, [eb], B, seed),
                "rms_gauss_minus_bias_ci": bootstrap_ci(diff, [eg, eb], B, seed),
                "rms_gauss_minus_bias_cluster_ci": bootstrap_ci(diff, [eg, eb], B, seed, clusters=cl),
                "mean_e_T_gauss": float(eg.mean()), "mean_e_T_bias": float(eb.mean()),
            })
    return out


def epsilon_scales(ctx, rows):
    """Reference scales for choosing epsilon later (NOT a frozen epsilon)."""
    T = np.array([r["T"] for r in rows if r["model"] == "orig" and r["donor"] == "natural" and r["S"] == "RB"
                  and r["k"] == ctx.cfg["ranks"][0]])
    return {"T_orig_natural_RB_rms": rms(T), "T_orig_natural_RB_sd": float(T.std(ddof=1)),
            "T_orig_natural_RB_mean": float(T.mean()), "note": "epsilon not frozen; reference scales only"}


def quadcheck(ctx, nodes=(32, 64, 128, 256), ref_nodes=512, n_rows=256, tol=1e-7):
    """Node-count convergence of the Gauss-Hermite expectation on the *calibration* moments and z distribution
    (GT-moment pool only; no recipient/donor data), plus a synthetic grid up to the configured sigma bound."""
    from .spectral import gauss_expect_gelu
    run, cfg = ctx.run, ctx.cfg
    spec = ctx.spectrum()
    rng = np.random.default_rng(cfg["seed"])
    bound = cfg["quadrature_sigma_max_checked"]
    out = {"tol_max_abs": tol, "ref_nodes": ref_nodes, "sigma_bound": bound, "real": [], "grid": {}}
    Wc = ctx.mlp.c_fc.weight.detach().cpu().double()
    b = ctx.mlp.c_fc.bias.detach().cpu().double()
    with wrapped_mlp(ctx.model, ctx.layer) as w:
        x_last, x_all = ctx.moment_activations(w)
    for k in cfg["ranks"]:
        P_R, P_B, mom = ctx.moments(spec, k, x_last, x_all)
        for pool, x in (("final_token", x_last), ("all_token", x_all), ("weights_only", x_last)):
            xs = x[rng.choice(len(x), size=min(n_rows, len(x)), replace=False)].double()
            mu, sd = mom[pool]
            m = (xs @ P_R) @ Wc + b + mu
            s = sd.expand_as(m)
            ref = gauss_expect_gelu(m, s, ref_nodes)
            errs = {n: float((gauss_expect_gelu(m, s, n) - ref).abs().max()) for n in nodes}
            out["real"].append({"k": k, "pool": pool, "sigma_max": float(sd.max()),
                                "m_range": [float(m.min()), float(m.max())], "max_abs_err": errs})
            run.log.info(f"quadcheck k={k} {pool}: sigma_max={float(sd.max()):.3f} errs={errs}")
    m = torch.linspace(-20, 20, 801, dtype=torch.float64).reshape(-1, 1)
    s = torch.linspace(0, bound, 41, dtype=torch.float64).reshape(1, -1)
    ref = gauss_expect_gelu(m, s, ref_nodes)
    out["grid"] = {n: float((gauss_expect_gelu(m, s, n) - ref).abs().max()) for n in nodes}
    sig_max = max(r["sigma_max"] for r in out["real"])
    ok = [n for n in nodes if out["grid"][n] < tol and all(r["max_abs_err"][n] < tol for r in out["real"])]
    out["observed_sigma_max"] = sig_max
    out["sigma_bound_covers_observed"] = bool(sig_max <= bound)
    out["smallest_converged_nodes"] = min(ok) if ok else None
    out["configured_nodes"] = cfg["quadrature_nodes"]
    out["configured_nodes_ok"] = cfg["quadrature_nodes"] in ok and sig_max <= bound
    run.save_json("quadcheck.json", out)
    return out
