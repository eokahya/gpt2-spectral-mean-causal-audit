"""Day-3 held-out validation/test (PROTOCOL_DAY3): datasets, frozen calibration artifact, held-out run,
pre-specified analysis, KL matching, epsilon, freeze/verify."""
import csv
import hashlib
import json
import math
import time

import numpy as np
import torch

from . import ROOT
from .bases import Basis, build_bases, check_projector
from .common import sha256_file, source_hashes, utcnow
from .data import _write_csv, check_prompt, read_csv
from .day2 import (aggregate, basis_moments, basis_reduction, boot, fwd, indicator_alignment, pool_inputs, response,
                   run_controls, run_reduced_basis, setup)
from .experiment import DONORS, BudgetExceeded, Context, RowWriter, positive_control
from .metrics import prob_diff, t_score, v_rd
from .model import wrapped_mlp

F64 = torch.float64
REG = ROOT / "registration" / "DAY3_REGISTRATION.json"
AMEND = ROOT / "registration" / "DAY3_AMENDMENTS.json"
FROZEN_SEL = ROOT / "registration" / "DAY3_FROZEN_SELECTION.json"
FROZEN_EPS = ROOT / "registration" / "DAY3_FROZEN_EPSILON.json"
POOLS = ("final_token", "all_token", "weights_only")


class GateInvalid(RuntimeError):
    pass


# ================================================================ held-out datasets

def load_sealed(path):
    pool = {}
    for r in read_csv(path):
        pool[(r["frame"], r["noun"], r["yy"])] = (r["text"], r["ids"])
    return pool


def build_pairs(pool, split, dcfg, seed):
    """Day-1 pair algorithm on a sealed prompt pool: 6 strata x n, same_year and wrong_donor controls.

    Output-independent: uses only the pool keys and the seed."""
    rng = np.random.default_rng(seed)
    years = sorted({y for (_, _, y) in pool})
    templates = sorted({(f, n) for (f, n, _) in pool})
    n_per = dcfg["pairs_per_stratum"]
    out = []
    for direction in ("up", "down"):
        for lo, hi in dcfg["diff_bins"]:
            cands = [(r, d) for r in years for d in years if lo <= (d - r if direction == "up" else r - d) <= hi]
            if len(cands) < n_per:
                raise ValueError(f"{split}: stratum {direction} {lo}-{hi} has only {len(cands)} year pairs")
            chosen = [cands[i] for i in rng.choice(len(cands), size=n_per, replace=False)]
            order = [templates[i] for i in rng.permutation(len(templates))]
            for j, (r, d) in enumerate(chosen):
                for off in range(len(templates)):
                    f, n = order[(j + off) % len(templates)]
                    if (f, n, r) in pool and (f, n, d) in pool:
                        break
                else:
                    raise ValueError(f"{split}: no valid template for years {r},{d}")
                others = [t for t in templates if t != (f, n)]
                sy = [t for t in others if (*t, r) in pool]
                sy_t = sy[rng.integers(len(sy))]
                wd = [(*t, y) for t in others for y in years if y not in (r, d) and (*t, y) in pool]
                wd_k = wd[rng.integers(len(wd))]
                p = {"rec": (f, n, r), "don": (f, n, d), "same_year": (*sy_t, r), "wrong_donor": wd_k}
                row = {"pair_id": f"{split[:3]}-{len(out):04d}", "direction": direction, "diff_bin": f"{lo}-{hi}",
                       "r": r, "d": d, "template": f"{f}|{n}", "frame": f, "noun": n}
                for role, key in p.items():
                    text, ids = pool[key]
                    row.update({f"{role}_template": f"{key[0]}|{key[1]}", f"{role}_yy": key[2],
                                f"{role}_text": text, f"{role}_ids": ids})
                out.append(row)
    return out


def dataset_checks(rows, pool, n_per):
    """Validity of one split's pair set (raises on violation; returns counts)."""
    strata = {}
    for p in rows:
        strata[(p["direction"], p["diff_bin"])] = strata.get((p["direction"], p["diff_bin"]), 0) + 1
        diff = p["d"] - p["r"]
        lo, hi = map(int, p["diff_bin"].split("-"))
        assert lo <= abs(diff) <= hi and (diff > 0) == (p["direction"] == "up"), p["pair_id"]
        assert p["rec_template"] == p["don_template"] == p["template"]
        assert p["same_year_yy"] == p["r"] and p["same_year_template"] != p["template"]
        assert p["wrong_donor_template"] != p["template"] and p["wrong_donor_yy"] not in (p["r"], p["d"])
        for role in ("rec", "don", "same_year", "wrong_donor"):
            f, n = p[f"{role}_template"].split("|")
            key = (f, n, p[f"{role}_yy"])
            assert key in pool, "control outside split pool"
            assert (p[f"{role}_text"], p[f"{role}_ids"]) == pool[key], "pair text/ids differ from sealed record"
    assert set(strata.values()) == {n_per} and len(strata) == 6, strata
    assert len({p["pair_id"] for p in rows}) == len(rows)
    return {"n_pairs": len(rows), "strata": {f"{a}|{b}": v for (a, b), v in strata.items()},
            "design": design_check(rows)}


def write_datasets(cfg, tok):
    d = ROOT / cfg["datasets"]["dir"]
    if d.exists():
        raise FileExistsError(f"{d} exists; held-out pair manifests are never overwritten")
    for split in ("validation", "test"):
        pool = load_sealed(ROOT / cfg["datasets"]["source"][split])
        fails = sum(check_prompt(tok, {"task": {"century": "17"}}, t, y) != ids for (_, _, y), (t, ids) in pool.items())
        if fails:
            raise ValueError(f"{split}: {fails} sealed prompts fail tokenization; refusing before any measurement")
    d.mkdir()
    index = {"created_utc": utcnow().isoformat(), "rule": "Day-1 pair algorithm on sealed prompt pools; output-independent",
             "splits": {}}
    for split in ("validation", "test"):
        src = ROOT / cfg["datasets"]["source"][split]
        pool = load_sealed(src)
        fails = sum(check_prompt(tok, {"task": {"century": "17"}}, t, y) != ids for (_, _, y), (t, ids) in pool.items())
        rows = build_pairs(pool, split, cfg["datasets"], cfg["datasets"]["seeds"][split])
        info = dataset_checks(rows, pool, cfg["datasets"]["pairs_per_stratum"])
        path = d / f"{split}_pairs.csv"
        _write_csv(path, rows)
        index["splits"][split] = {**info, "file": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                                  "source": str(src.relative_to(ROOT)), "source_sha256": sha256_file(src),
                                  "seed": cfg["datasets"]["seeds"][split], "pool_size": len(pool),
                                  "tokenization_failures": int(fails),
                                  "n_templates": len({p["template"] for p in rows}),
                                  "frames": sorted({p["frame"] for p in rows})}
    (d / "datasets_index.json").write_text(json.dumps(index, indent=2))
    return index


def dataset_hashes(cfg):
    d = ROOT / cfg["datasets"]["dir"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(d.glob("*")) if p.is_file()}


# ================================================================ frozen calibration artifact

def _tsha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def artifact_paths(cfg):
    a = cfg["calibration_artifact"]
    return ROOT / a["dir"] / a["file"], ROOT / a["dir"] / a["index"]


def fit_settings(cfg):
    return {"ridge_rel": cfg["bases"]["ridge_rel"], "fit_pool": cfg["bases"]["fit_pool"], "layer": cfg["layer"],
            "quadrature_sigma_max_checked": cfg["quadrature_sigma_max_checked"], "svd": cfg["svd"]}


def build_artifact(ctx):
    """Calibration-only: fit bases + moments on the Day-2 moment pool and serialise them (CPU float64)."""
    cfg = ctx.cfg
    path, ipath = artifact_paths(cfg)
    if path.exists() or ipath.exists():
        raise FileExistsError("calibration artifact exists; never overwritten")
    spec = ctx.spectrum()
    with wrapped_mlp(ctx.model, ctx.layer) as w:
        x_last, x_all = pool_inputs(ctx, w)
    bases = build_bases(cfg, spec, x_last)
    tensors, energy, meta = {}, {}, {}
    for (bid, k), b in bases.items():
        tensors[f"P|{bid}|{k}"] = b.P.contiguous()
        mom = basis_moments(ctx, b, x_last, x_all)
        for pool in POOLS:
            tensors[f"mu|{bid}|{k}|{pool}"] = mom[pool][0].contiguous()
            tensors[f"sigma|{bid}|{k}|{pool}"] = mom[pool][1].contiguous()
        _, B_row = b.row_maps()
        energy[f"{bid}|{k}"] = float((x_last @ B_row).norm(dim=-1).pow(2).mean().sqrt())
        meta[f"{bid}|{k}"] = {kk: v for kk, v in b.meta.items() if isinstance(v, (int, float))}
    path.parent.mkdir(exist_ok=True)
    torch.save({"tensors": tensors}, path)
    index = {"created_utc": utcnow().isoformat(), "file_sha256": sha256_file(path),
             "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": _tsha(t)} for n, t in tensors.items()},
             "fit_pool": {"moment_pool_manifest_sha256": sha256_file(ROOT / "manifests" / "moment_pool_calibration.csv"),
                          "x_last_sha256": _tsha(x_last), "x_all_sha256": _tsha(x_all),
                          "n_final_token": len(x_last), "n_all_token": len(x_all)},
             "calib_B_energy_rms": energy, "basis_meta": meta,
             "model_safetensors_sha256": ctx.run.manifest["model"].get("safetensors_sha256"),
             "config_day3_sha256": sha256_file(cfg["_path"]), "fit_settings": fit_settings(cfg),
             "families": cfg["bases"]["families"],
             "random_seeds": cfg["bases"]["random_seeds"], "ranks": cfg["ranks"]}
    ipath.write_text(json.dumps(index, indent=2))
    return index


def expected_tensor_names(cfg):
    ids = [f for f in cfg["bases"]["families"] if f != "random"] + [f"random_s{s}" for s in cfg["bases"]["random_seeds"]
                                                                   if "random" in cfg["bases"]["families"]]
    names = set()
    for bid in ids:
        for k in cfg["ranks"]:
            names.add(f"P|{bid}|{k}")
            for pool in POOLS:
                names |= {f"mu|{bid}|{k}|{pool}", f"sigma|{bid}|{k}|{pool}"}
    return names


def load_artifact(cfg, expected_file_sha=None, model_sha=None):
    """Load and verify the frozen artifact against its index AND the runtime config/model/pool. Returns
    (bases, moments, index). Raises on any inconsistency."""
    path, ipath = artifact_paths(cfg)
    index = json.loads(ipath.read_text())
    if index["families"] != cfg["bases"]["families"] or index["random_seeds"] != cfg["bases"]["random_seeds"] \
            or index["ranks"] != cfg["ranks"]:
        raise ValueError("artifact fit settings differ from runtime config")
    if index.get("fit_settings") != fit_settings(cfg):
        raise ValueError("artifact fit settings (ridge/fit pool/quadrature) differ from runtime config")
    if index["fit_pool"]["moment_pool_manifest_sha256"] != sha256_file(ROOT / "manifests" / "moment_pool_calibration.csv"):
        raise ValueError("artifact moment pool manifest hash mismatch")
    if model_sha is not None and index["model_safetensors_sha256"] != model_sha:
        raise ValueError("artifact was fitted with different model weights")
    if set(index["tensors"]) != expected_tensor_names(cfg):
        raise ValueError("artifact tensor set differs from the set required by the config")
    sha = sha256_file(path)
    if sha != index["file_sha256"] or (expected_file_sha is not None and sha != expected_file_sha):
        raise ValueError("calibration artifact hash mismatch")
    tensors = torch.load(path, map_location="cpu", weights_only=True)["tensors"]
    if set(tensors) != set(index["tensors"]):
        raise ValueError("artifact tensor set differs from index")
    for n, t in tensors.items():
        spec = index["tensors"][n]
        if t.dtype != F64 or t.device.type != "cpu" or list(t.shape) != spec["shape"]:
            raise ValueError(f"artifact tensor {n}: bad dtype/device/shape")
        if not torch.isfinite(t).all():
            raise ValueError(f"artifact tensor {n}: non-finite")
        if _tsha(t) != spec["sha256"]:
            raise ValueError(f"artifact tensor {n}: content hash mismatch")
        if n.startswith("P|") and tuple(t.shape) != (768, 768):
            raise ValueError(f"{n}: wrong basis shape")
        if not n.startswith("P|") and tuple(t.shape) != (3072,):
            raise ValueError(f"{n}: wrong moment shape")
        if n.startswith("sigma|") and (t < 0).any():
            raise ValueError(f"{n}: negative sigma")
    bases, moments = {}, {}
    for n, t in tensors.items():
        if n.startswith("P|"):
            _, bid, k = n.split("|")
            k = int(k)
            fam, seed = ("random", int(bid.split("_s")[1])) if bid.startswith("random_s") else (bid, None)
            b = Basis(fam, k, t, seed=seed)
            check_projector(b)
            bases[(bid, k)] = b
            moments[(bid, k)] = {pool: (tensors[f"mu|{bid}|{k}|{pool}"], tensors[f"sigma|{bid}|{k}|{pool}"])
                                 for pool in POOLS}
    return bases, moments, index


# ================================================================ held-out context and run

class HeldoutContext(Context):
    """Context whose pairs come from manifests_day3/{split}_pairs.csv. The moment pool is unreachable."""

    def __init__(self, cfg, run, split):
        super().__init__(cfg, run)
        self.split = split
        self.pairs = read_csv(ROOT / cfg["datasets"]["dir"] / f"{split}_pairs.csv")
        self.r = np.array([p["r"] for p in self.pairs])
        self.d = np.array([p["d"] for p in self.pairs])
        self.v = v_rd(self.r, self.d)
        self.templates = np.array([p["template"] for p in self.pairs])
        self.moment = None
        run.manifest["split"] = split

    def moment_activations(self, wrapper):
        raise RuntimeError("held-out split: moment-pool forwards are forbidden (no refit)")


def frame_of(p):
    return p["template"].split("|")[0]


def per_frame_gate(ctx, pd, T, B, seed):
    out = {}
    frames = np.array([frame_of(p) for p in ctx.pairs])
    for f in sorted(set(frames)):
        m = frames == f
        out[f] = {"n": int(m.sum()), "pd_mean_ci": boot(pd[m], "mean", B, seed),
                  "T_RB_natural_mean_ci": boot(T["natural"][m], "mean", B, seed),
                  "T_nat_minus_same_year_ci": boot(T["natural"][m] - T["same_year"][m], "mean", B, seed)}
    return out


def heldout_run(ctx, bases, moments, art_sha, frozen_selection=None):
    """Gate -> original geometric controls -> reduced models (frozen calibration bases/moments) -> analysis."""
    run_, cfg = ctx.run, ctx.cfg
    Bn, seed = cfg["bootstrap"]["B"], cfg["seed"]
    budget, reserve = cfg["budget"]["walltime_budget_s"], cfg["budget"]["aggregate_reserve_s"]
    state_design = design_check(ctx.pairs)
    b = ctx.batches()
    Wc = ctx.mlp.c_fc.weight.detach().cpu().to(F64)
    ctl_w = RowWriter(run_.dir / "control_rows.csv")
    red_w = RowWriter(run_.dir / "reduced_rows.csv")
    unp_w = RowWriter(run_.dir / "unperturbed_rows.csv")
    progress, state, uninterv, vec = {"done": [], "status": "running"}, {"split": ctx.split,
                                                                          "design": state_design}, [], []

    def checkpoint(status="running"):
        progress.update(status=status, elapsed_s=run_.elapsed())
        run_.save_json("partial_results.json", {"progress": progress, **state})

    def guard(cell):
        if run_.elapsed() > budget - reserve:
            checkpoint("budget_exceeded")
            raise BudgetExceeded(f"budget before control cell {cell}")

    with wrapped_mlp(ctx.model, ctx.layer) as w:
        o, deltas = setup(ctx, w, b)
        gate, pd, T = positive_control(ctx, w, b, o, Bn, seed)
        gw = RowWriter(run_.dir / "gate_rows.csv")
        gw.add([{"pair_id": p["pair_id"], "frame": frame_of(p), "template": p["template"], "direction": p["direction"],
                 "diff_bin": p["diff_bin"], "r": p["r"], "d": p["d"], "PD": float(pd[i]),
                 "T_natural": float(T["natural"][i]), "T_same_year": float(T["same_year"][i]),
                 "T_wrong_donor": float(T["wrong_donor"][i])} for i, p in enumerate(ctx.pairs)])
        gw.close()
        gate["per_frame"] = per_frame_gate(ctx, pd, T, Bn, seed)
        state["positive_control"] = gate
        run_.save_json("positive_control.json", gate)
        if not gate["valid"]:
            checkpoint("invalid_gate")
            raise GateInvalid(f"{ctx.split}: positive gate failed -> assay invalid, no reduced-model claims")
        x_rec = o["rec"]["x_pre_last"].cpu().to(F64)
        art_idx = json.loads(artifact_paths(cfg)[1].read_text())
        state["input_shift_diagnostic"] = {
            f"{bid}|{k}": {"heldout_B_energy_rms": float((x_rec @ bs.row_maps()[1]).norm(dim=-1).pow(2).mean().sqrt()),
                           "calib_B_energy_rms": art_idx["calib_B_energy_rms"][f"{bid}|{k}"]}
            for (bid, k), bs in bases.items()}
        align = {"natural": np.ones(len(ctx.pairs)), "same_year": indicator_alignment(ctx.r, ctx.d, ctx.r),
                 "wrong_donor": indicator_alignment(ctx.r, ctx.d, np.array([p["wrong_donor_yy"] for p in ctx.pairs]))}
        base_o = o["rec"]

        def cell_done(cell):
            progress["done"].append(f"controls:{cell[0]}|{cell[1]}")
            checkpoint()
        try:
            resp_o, match_log = run_controls(ctx, w, b, base_o, deltas, bases, ctl_w, Wc, align, guard, cell_done)
            state["matching"] = match_log
            pd_orig = prob_diff(base_o["logits"].cpu(), ctx.J, ctx.r).numpy()
            for bid in dict.fromkeys(bid for bid, _ in bases):
                t0 = time.perf_counter()
                for k in cfg["ranks"]:
                    run_reduced_basis(ctx, w, b, base_o, deltas, bases[(bid, k)], moments[(bid, k)], resp_o, red_w,
                                      uninterv, vec, run_, budget, reserve, pd_orig, unp_w)
                progress["done"].append(bid)
                checkpoint()
                run_.log.info(f"basis {bid} done in {time.perf_counter() - t0:.1f}s (elapsed {run_.elapsed():.0f}s)")
        except BudgetExceeded:
            ctl_w.close(), red_w.close(), unp_w.close()
            checkpoint("budget_exceeded")
            raise
    ctl_w.close(), red_w.close(), unp_w.close()
    if sha256_file(artifact_paths(cfg)[0]) != art_sha:
        raise RuntimeError("calibration artifact changed during the run")
    checkpoint("variants_done")
    results = aggregate(ctx, ctl_w.rows, red_w.rows, uninterv, vec, match_log, Bn, seed)
    results.update(state)
    eps = epsilon_from_day1(cfg)
    ref_split = float(np.sqrt(np.mean(np.square(
        [r["T"] for r in ctl_w.rows if r["basis"] == "any" and r["donor"] == "natural" and r["regime"] == "native"]))))
    results["analysis"] = analyze(ctx.pairs, red_w.rows, cfg["analysis"], eps["epsilon"], eps["reference_rms"],
                                  ref_split, seed)
    results["epsilon_proposal"] = eps
    if ctx.split == "validation":
        results["kl_matching_this_split"] = kl_match(unp_w.rows, cfg["analysis"]["kl_matching"], cfg["ranks"])
    if frozen_selection is not None:
        results["kl_matched_contrasts_frozen"] = kl_matched_contrasts(
            ctx.pairs, red_w.rows, unp_w.rows, frozen_selection["kl_matching"], cfg["analysis"]["kl_matching"],
            cfg["analysis"]["bootstrap_B"], seed)
    results["orig_RB_rms_this_split"] = ref_split
    run_.save_json("results.json", results)
    if run_.elapsed() > budget:
        raise BudgetExceeded(f"finished but walltime {run_.elapsed():.0f}s exceeded budget {budget}s")
    return results


# ================================================================ pre-specified analysis

def cell_key(p):
    return (frame_of(p), p["direction"], p["diff_bin"])


def design_check(pairs):
    """Every frame x direction x diff_bin cell must contain every template of its frame (needed so that each
    cluster-bootstrap draw has a positive weight in every cell). Raises otherwise."""
    tmpl_by_frame, cell_t = {}, {}
    for p in pairs:
        tmpl_by_frame.setdefault(frame_of(p), set()).add(p["template"])
        cell_t.setdefault(cell_key(p), set()).add(p["template"])
    for c, ts in cell_t.items():
        if ts != tmpl_by_frame[c[0]]:
            raise AssertionError(f"cell {c} lacks templates {sorted(tmpl_by_frame[c[0]] - ts)}")
    if len(cell_t) != 2 * len(tmpl_by_frame) * 3:
        raise AssertionError(f"expected {6 * len(tmpl_by_frame)} cells, got {len(cell_t)}")
    return {"n_cells": len(cell_t), "templates_per_frame": {f: len(t) for f, t in tmpl_by_frame.items()}}


def strat_mse_draws(errs, pairs, B, seed, mode, frame=None):
    """Stratified bootstrap of MSE for paired error arrays (PROTOCOL_DAY3 §5).

    Cells = frame x direction x diff_bin with fixed weights N_c/N. mode='cluster': within each frame, draw that
    frame's templates with replacement; cell mean = multiplicity-weighted mean of squared errors.
    mode='iid': resample N_c pairs within each cell. Returns (point MSEs, draws [len(errs), B])."""
    sel = np.array([frame is None or frame_of(p) == frame for p in pairs])
    sp = [p for p, s_ in zip(pairs, sel) if s_]
    e2 = [np.asarray(e, float)[sel] ** 2 for e in errs]
    n = len(sp)
    cells = {}
    for i, p in enumerate(sp):
        cells.setdefault(cell_key(p), []).append(i)
    rng = np.random.default_rng(seed)
    draws = np.zeros((len(errs), B))
    if mode == "cluster":
        tlist = sorted({p["template"] for p in sp})
        tix = {t: j for j, t in enumerate(tlist)}
        M = np.zeros((B, len(tlist)))
        for f in sorted({frame_of(p) for p in sp}):
            ft = [tix[t] for t in tlist if t.split("|")[0] == f]
            pick = rng.integers(0, len(ft), (B, len(ft)))
            np.add.at(M, (np.arange(B)[:, None], np.asarray(ft)[pick]), 1.0)
        W = M[:, [tix[p["template"]] for p in sp]]
        for c, idx in cells.items():
            den = W[:, idx].sum(1)
            if not (den > 0).all():
                raise AssertionError(f"zero bootstrap weight in cell {c}")
            for a, v in enumerate(e2):
                draws[a] += (len(idx) / n) * (W[:, idx] * v[idx]).sum(1) / den
    elif mode == "iid":
        for c, idx in cells.items():
            idx = np.asarray(idx)
            r = rng.integers(0, len(idx), (B, len(idx)))
            for a, v in enumerate(e2):
                draws[a] += (len(idx) / n) * v[idx][r].mean(1)
    else:
        raise ValueError(mode)
    return [float(v.mean()) for v in e2], draws


def rms_diff_stat(a, b, pairs, B, seed, mode, frame=None):
    (ma, mb), d = strat_mse_draws([a, b], pairs, B, seed, mode, frame)
    return float(np.sqrt(ma) - np.sqrt(mb)), np.sqrt(d[0]) - np.sqrt(d[1])


def rms_stat(a, pairs, B, seed, mode, frame=None):
    (m,), d = strat_mse_draws([a], pairs, B, seed, mode, frame)
    return float(np.sqrt(m)), np.sqrt(d[0])


def two_sided(point, draws, level):
    q = (1 - level) / 2
    return [point, float(np.quantile(draws, q)), float(np.quantile(draws, 1 - q))]


def upper(point, draws, level):
    return [point, float(np.quantile(draws, level))]


def direction(ci):
    """Direction flag of a two-sided paired-difference CI. No equivalence flag: epsilon is an absolute
    reconstruction tolerance, not a margin for differences."""
    lo, hi = ci[1], ci[2]
    return {"direction": "lhs_lower" if hi < 0 else ("lhs_higher" if lo > 0 else "not_distinguished")}


def _errs(red_rows, pids, basis, method, k, scope, S):
    m = {r["pair_id"]: float(r["e_T"]) for r in red_rows if r["basis"] == basis and r["method"] == method
         and int(r["k"]) == k and r["scope"] == scope and r["S"] == S}
    if set(m) != set(pids):
        raise ValueError(f"missing rows for {basis}/{method}/{k}/{scope}/{S}")
    return np.array([m[p] for p in pids], float)


def analyze(pairs, red_rows, acfg, eps, ref_cal, ref_split, seed):
    """Pre-specified Day-3 inference: P1 two-sided 95%; S1/S2 one-sided 97.5% upper bounds (Bonferroni within the
    secondary family). Stratified frame-within template-cluster bootstrap is primary; pair-iid supports."""
    B = acfg["bootstrap_B"]
    pids = [p["pair_id"] for p in pairs]
    P, sec = acfg["primary"], acfg["secondary"]
    lvl = 1 - acfg["multiplicity"]["family_alpha"] / len(sec)
    out = {"design": design_check(pairs), "bootstrap_B": B, "epsilon": eps, "secondary_upper_level": lvl,
           "orig_RB_rms_calibration": ref_cal, "orig_RB_rms_this_split": ref_split,
           "epsilon_over_orig_RB_rms_this_split": eps / ref_split,
           "notes": ["Bonferroni within the secondary family {S1,S2} only; no global FWER including P1.",
                     "Bootstrap intervals are nominal, approximate and conditional on fitted bases/moments and the "
                     "2 frames of this split.",
                     "The positive gate does not guarantee against weak-signal splits.",
                     "If the test-split orig RB scale approaches epsilon, passing S2 is not strong mechanism "
                     "preservation."]}

    def run_all(frame=None):
        res = {}
        eg = _errs(red_rows, pids, P["basis"], P["contrast"][0], P["k"], P["scope"], P["S"])
        eb = _errs(red_rows, pids, P["basis"], P["contrast"][1], P["k"], P["scope"], P["S"])
        r = {}
        for mode in ("cluster", "iid"):
            pt, d = rms_diff_stat(eg, eb, pairs, B, seed, mode, frame)
            r[f"diff_{mode}_ci"] = two_sided(pt, d, P["ci"])
        r.update(direction(r["diff_cluster_ci"]))
        r["rms_gauss"] = rms_stat(eg, pairs, 1, seed, "iid", frame)[0]
        r["rms_bias"] = rms_stat(eb, pairs, 1, seed, "iid", frame)[0]
        r["reading"] = "lhs=gauss; a positive or undistinguished difference does not support Gaussian superiority"
        res["P1"] = r
        for spec in sec:
            r = {"spec": spec}
            if spec["kind"] == "basis_difference":
                a = _errs(red_rows, pids, spec["lhs"][0], spec["lhs"][1], spec["k"], spec["scope"], spec["S"])
                b = _errs(red_rows, pids, spec["rhs"][0], spec["rhs"][1], spec["k"], spec["scope"], spec["S"])
                for mode in ("cluster", "iid"):
                    pt, d = rms_diff_stat(a, b, pairs, B, seed, mode, frame)
                    r[f"diff_{mode}_upper"] = upper(pt, d, lvl)
                r["supported_lhs_lower"] = bool(r["diff_cluster_upper"][1] < 0)
            else:
                a = _errs(red_rows, pids, spec["basis"], spec["method"], spec["k"], spec["scope"], spec["S"])
                for mode in ("cluster", "iid"):
                    pt, d = rms_stat(a, pairs, B, seed, mode, frame)
                    r[f"rms_{mode}_upper"] = upper(pt, d, lvl)
                r["preserved_upper_lt_eps"] = bool(r["rms_cluster_upper"][1] < eps)
                r["nrmse_fixed_cal_denominator"] = r["rms_cluster_upper"][0] / ref_cal
                r["nrmse_this_split_denominator"] = r["rms_cluster_upper"][0] / ref_split
            res[spec["id"]] = r
        return res

    out.update(run_all())
    out["P1"]["spec"] = P
    out["per_frame_descriptive"] = {f: run_all(f) for f in sorted({frame_of(p) for p in pairs})}
    return out


def epsilon_from_day1(cfg):
    rule = cfg["analysis"]["epsilon_rule"]
    path = ROOT / rule["reference_csv"]
    T = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["model"] == "orig" and r["donor"] == "natural" and r["S"] == "RB" and r["k"] == "96":
                T.append(float(r["T"]))
    if len(T) != 192:
        raise ValueError(f"epsilon reference rows: {len(T)} != 192")
    ref = float(np.sqrt(np.mean(np.square(T))))
    return {"epsilon": rule["fraction"] * ref, "reference_rms": ref, "fraction": rule["fraction"], "n": len(T),
            "reference_csv": rule["reference_csv"], "reference_csv_sha256": sha256_file(path)}


def kl_gap(a, b):
    """Symmetric relative KL gap g = 2|a-b|/(a+b); 0 when both are 0."""
    return 0.0 if a == 0 and b == 0 else 2 * abs(a - b) / (a + b)


def kl_means(unp_rows, ranks):
    out = {}
    for r in unp_rows:
        key = (r["basis"], r["method"], r["scope"], int(r["k"]))
        out.setdefault(key, []).append(float(r["kl_orig_to_M"]))
    res = {}
    for key, v in out.items():
        m = float(np.mean(np.asarray(v, dtype=np.float64)))
        if not (math.isfinite(m) and m >= 0):
            raise ValueError(f"non-finite KL for {key}")
        res[key] = m
    return res


def kl_match(unp_rows, kcfg, ranks):
    """Discrete same-scope KL matching at existing ranks (PROTOCOL_DAY3 §6). Matched within rule, not equal quality."""
    kl = kl_means(unp_rows, ranks)
    out = {"rule": kcfg, "families": {}}
    for fam in kcfg["families"]:
        for scope in kcfg["scopes"]:
            cands = []
            for k1 in ranks:
                for k2 in ranks:
                    a = kl[(fam["lhs"][0], fam["lhs"][1], scope, k1)]
                    b = kl[(fam["rhs"][0], fam["rhs"][1], scope, k2)]
                    g = kl_gap(a, b)
                    if g <= kcfg["max_gap"]:
                        cands.append({"lhs_k": k1, "rhs_k": k2, "kl_lhs": a, "kl_rhs": b, "gap": g,
                                      "abs_gap": abs(a - b), "configuration_comparison": k1 != k2})
            cands.sort(key=lambda c: (c["gap"], max(c["lhs_k"], c["rhs_k"]), c["lhs_k"] + c["rhs_k"],
                                      c["lhs_k"], c["rhs_k"]))
            out["families"][f"{fam['id']}|{scope}"] = {
                "lhs": fam["lhs"], "rhs": fam["rhs"], "scope": scope,
                "kl": {f"lhs|{k}": kl[(fam["lhs"][0], fam["lhs"][1], scope, k)] for k in ranks}
                | {f"rhs|{k}": kl[(fam["rhs"][0], fam["rhs"][1], scope, k)] for k in ranks},
                "matched": cands, "selected": cands[0] if cands else None,
                "claim": "matched_within_rule" if cands else "no_iso_kl_claim"}
    return out


def kl_matched_contrasts(pairs, red_rows, unp_rows, selection, kcfg, B, seed):
    """Descriptive RMS e_RB differences for frozen KL-matched pairs on test. The KL gap is recomputed on this split;
    if it exceeds the rule, no near-quality interpretation and no reselection."""
    pids = [p["pair_id"] for p in pairs]
    kl = kl_means(unp_rows, None)
    out = {}
    for key, fam in selection["families"].items():
        sel = fam["selected"]
        if sel is None:
            out[key] = {"claim": "no_iso_kl_claim"}
            continue
        a = _errs(red_rows, pids, fam["lhs"][0], fam["lhs"][1], sel["lhs_k"], fam["scope"], "RB")
        b = _errs(red_rows, pids, fam["rhs"][0], fam["rhs"][1], sel["rhs_k"], fam["scope"], "RB")
        ka = kl[(fam["lhs"][0], fam["lhs"][1], fam["scope"], sel["lhs_k"])]
        kb = kl[(fam["rhs"][0], fam["rhs"][1], fam["scope"], sel["rhs_k"])]
        g = kl_gap(ka, kb)
        r = {"selected_on_validation": sel, "descriptive": True, "kl_lhs_this_split": ka, "kl_rhs_this_split": kb,
             "gap_this_split": g, "abs_gap_this_split": abs(ka - kb),
             "near_quality_interpretation": bool(g <= kcfg["max_gap"]),
             "label": "configuration comparison (across ranks), not a pure Gaussian effect"
             if sel["lhs_k"] != sel["rhs_k"] else "same-rank comparison"}
        for mode in ("cluster", "iid"):
            pt, d = rms_diff_stat(a, b, pairs, B, seed, mode)
            r[f"diff_{mode}_ci"] = two_sided(pt, d, 0.95)
        out[key] = r
    return out


# ================================================================ registration / freeze / verify

def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def current_bindings(cfg):
    path, ipath = artifact_paths(cfg)
    return {"protocol_day3": sha256_file(ROOT / "PROTOCOL_DAY3.md"), "config_day3": sha256_file(cfg["_path"]),
            "source_tree": source_hashes()["tree_sha256"], "calibration_artifact": sha256_file(path),
            "calibration_index": sha256_file(ipath), "datasets": dataset_hashes(cfg)}


def check_source_allowed(tree_sha, reg):
    if tree_sha == reg["bindings"]["source_tree"]:
        return "registered"
    if AMEND.exists():
        for a in json.loads(AMEND.read_text()):
            if a["tree_sha256"] == tree_sha and (ROOT / a["archive"]).is_dir():
                return f"amendment:{a['utc']}"
    raise RuntimeError("source tree differs from Day-3 registration and no archived amendment covers it")


def verify_registration(cfg):
    if not REG.exists():
        raise RuntimeError("no Day-3 registration")
    reg = json.loads(REG.read_text())
    if reg.get("kind") != "day3_registration" or reg.get("checksum") != canonical_sha(
            {k: v for k, v in reg.items() if k != "checksum"}):
        raise RuntimeError("Day-3 registration malformed or checksum mismatch")
    now = current_bindings(cfg)
    for key in ("protocol_day3", "config_day3", "calibration_artifact", "calibration_index", "datasets"):
        if now[key] != reg["bindings"][key]:
            raise RuntimeError(f"Day-3 registered {key} changed")
    if reg.get("epsilon") != epsilon_from_day1(cfg):
        raise RuntimeError("registered epsilon object differs from its recomputation (value or reference hash)")
    status = check_source_allowed(now["source_tree"], reg)
    return reg, status


def write_registration(cfg):
    if REG.exists():
        raise FileExistsError("Day-3 registration exists; not overwriting")
    reg = {"kind": "day3_registration", "registered_utc": utcnow().isoformat(),
           "protocol_version": cfg["protocol_version"], "bindings": current_bindings(cfg),
           "epsilon": epsilon_from_day1(cfg),
           "note": "validation/test unseen; epsilon fixed here (calibration-informed); KL selection frozen after "
                   "validation"}
    reg["checksum"] = canonical_sha(reg)
    REG.write_text(json.dumps(reg, indent=2))
    return reg


def run_bindings(run_dir):
    run_dir = ROOT / run_dir
    files = ["results.json", "run_manifest.json", "control_rows.csv", "reduced_rows.csv", "unperturbed_rows.csv"]
    return {f: sha256_file(run_dir / f) for f in files}


def freeze(cfg, val_run_dir):
    if FROZEN_SEL.exists() or FROZEN_EPS.exists():
        raise FileExistsError("Day-3 freeze files exist; never overwritten")
    reg, _ = verify_registration(cfg)
    vdir = ROOT / val_run_dir
    man = json.loads((vdir / "run_manifest.json").read_text())
    res = json.loads((vdir / "results.json").read_text())
    if man.get("status") != "ok" or man.get("split") != "validation" or res.get("split") != "validation":
        raise RuntimeError("validation run is not a completed, valid validation run")
    if not res["positive_control"]["valid"]:
        raise RuntimeError("validation gate invalid; nothing to freeze")
    now = current_bindings(cfg)
    checks = {"source": (man["source"]["tree_sha256"], now["source_tree"]),
              "config": (man["config_sha256"], now["config_day3"]),
              "protocol_day3": (man.get("day3_protocol_sha256"), now["protocol_day3"]),
              "calibration_artifact": (man.get("calibration_artifact_sha256"), now["calibration_artifact"]),
              "registration": (man.get("day3_registration", {}).get("sha256"), sha256_file(REG))}
    bad = [k for k, (a, b_) in checks.items() if a != b_]
    if bad:
        raise RuntimeError(f"validation run does not match current {bad}; archive the change and re-run validation")
    unp = read_rows(vdir / "unperturbed_rows.csv")
    sel = kl_match(unp, cfg["analysis"]["kl_matching"], cfg["ranks"])
    eps = reg["epsilon"]
    bind = {**now, "registration_day3": sha256_file(REG),
            "validation_run": {"dir": str(val_run_dir), **run_bindings(val_run_dir)}}
    common = {"created_utc": utcnow().isoformat(), "bindings": bind}
    s = {"kind": "day3_frozen_selection", **common, "anchor_rank": cfg["analysis"]["anchor_rank"],
         "confirmatory_rank": cfg["analysis"]["anchor_rank"], "primary": cfg["analysis"]["primary"],
         "secondary": cfg["analysis"]["secondary"], "multiplicity": cfg["analysis"]["multiplicity"],
         "kl_matching": sel}
    e = {"kind": "day3_frozen_epsilon", **common, **eps}
    for obj, path in ((s, FROZEN_SEL), (e, FROZEN_EPS)):
        obj["checksum"] = canonical_sha(obj)
        path.write_text(json.dumps(obj, indent=2))
    return s, e


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def verify_frozen(cfg):
    """Hard checks before test: real content, self-checksums, every binding recomputed, epsilon and KL selection
    recomputed from their sources."""
    if not (FROZEN_SEL.exists() and FROZEN_EPS.exists()):
        raise RuntimeError("freeze files missing")
    s, e = json.loads(FROZEN_SEL.read_text()), json.loads(FROZEN_EPS.read_text())
    for obj, kind in ((s, "day3_frozen_selection"), (e, "day3_frozen_epsilon")):
        if obj.get("kind") != kind or obj.get("checksum") != canonical_sha({k: v for k, v in obj.items() if k != "checksum"}):
            raise RuntimeError(f"{kind}: malformed or checksum mismatch")
    if s["bindings"] != e["bindings"]:
        raise RuntimeError("selection/epsilon bindings differ")
    reg, status = verify_registration(cfg)
    now = current_bindings(cfg)
    bind = s["bindings"]
    for key in ("protocol_day3", "config_day3", "calibration_artifact", "calibration_index", "datasets"):
        if now[key] != bind[key]:
            raise RuntimeError(f"frozen binding {key} changed")
    if now["source_tree"] != bind["source_tree"]:
        raise RuntimeError("source tree changed after freeze; test forbidden without an explicit invalidate/new version")
    if sha256_file(REG) != bind["registration_day3"]:
        raise RuntimeError("Day-3 registration changed after freeze")
    vr = bind["validation_run"]
    if run_bindings(vr["dir"]) != {k: v for k, v in vr.items() if k != "dir"}:
        raise RuntimeError("validation run artifacts changed after freeze")
    if json.loads((ROOT / vr["dir"] / "run_manifest.json").read_text()).get("split") != "validation":
        raise RuntimeError("frozen run is not a validation run (split mix)")
    eps = epsilon_from_day1(cfg)
    if {k: e.get(k) for k in eps} != eps or reg["epsilon"] != eps:
        raise RuntimeError("frozen epsilon object (value or reference hash) does not match registration/recomputation")
    sel = kl_match(read_rows(ROOT / vr["dir"] / "unperturbed_rows.csv"), cfg["analysis"]["kl_matching"],
                   cfg["ranks"])
    if canonical_sha(sel) != canonical_sha(s["kl_matching"]):
        raise RuntimeError("frozen KL selection does not match recomputation")
    if s["primary"] != cfg["analysis"]["primary"] or s["secondary"] != cfg["analysis"]["secondary"]:
        raise RuntimeError("frozen analysis spec differs from config")
    return s, e


# ================================================================ calibration-only smoke

def smoke(ctx, bases, moments, art_sha):
    """Calibration only: artifact == fresh fit; Day-2 numerical replay; positive gate. Raises on failure."""
    cfg = ctx.cfg
    out, fails = {}, []
    spec = ctx.spectrum()
    b = ctx.batches()
    with wrapped_mlp(ctx.model, ctx.layer) as w:
        x_last, x_all = pool_inputs(ctx, w)
        fresh = build_bases(cfg, spec, x_last)
        out["artifact_vs_refit_max_abs_P"] = max(float((fresh[key].P - bases[key].P).abs().max()) for key in bases)
        mdiff = 0.0
        for key in bases:
            m2 = basis_moments(ctx, fresh[key], x_last, x_all)
            for pool in POOLS:
                for i in (0, 1):
                    mdiff = max(mdiff, float((m2[pool][i] - moments[key][pool][i]).abs().max()))
        out["artifact_vs_refit_max_abs_moments"] = mdiff
        if out["artifact_vs_refit_max_abs_P"] != 0 or mdiff != 0:
            fails.append("artifact differs from deterministic calibration refit")
        o, deltas = setup(ctx, w, b)
        gate, _, _ = positive_control(ctx, w, b, o, cfg["bootstrap"]["B"], cfg["seed"])
        out["positive_control_valid"] = gate["valid"]
        if not gate["valid"]:
            fails.append("calibration positive gate invalid")
        d2 = ROOT / cfg["day2"]["run"]
        ctl = {(r["basis"], r["S"], r["pair_id"]): float(r["T"]) for r in read_rows(d2 / "control_rows.csv")
               if r["k"] == "192" and r["donor"] == "natural" and r["regime"] == "native"}
        red = {(r["basis"], r["method"], r["scope"], r["S"], r["pair_id"]): float(r["T"])
               for r in read_rows(d2 / "reduced_rows.csv") if r["k"] == "192"}
        replay = {}
        pids = [p["pair_id"] for p in ctx.pairs]
        for bid in ("svd", "pca", "wsvd", "random_s101"):
            bs = bases[(bid, 192)]
            R_row, B_row = bs.row_maps()
            phys = {"R": (deltas["natural"] @ R_row).float(), "B": (deltas["natural"] @ B_row).float(),
                    "RB": deltas["natural"].float()}
            for S in ("R", "B"):
                T = t_score(response(fwd(ctx, w, b, phys[S]), o["rec"], ctx.J)["dJ"], ctx.v).numpy()
                replay[f"orig|{bid}|{S}"] = max(abs(T[i] - ctl[(bid, S, p)]) for i, p in enumerate(pids))
            for method, scope in (("bias", "last_token"), ("gauss", "last_token"), ("zero", "full_layer"),
                                  ("gauss", "full_layer")):
                rd = basis_reduction(ctx, bs, moments[(bid, 192)], method, scope)
                base = fwd(ctx, w, b, reduction=rd)
                for S in ("R", "RB"):
                    T = t_score(response(fwd(ctx, w, b, phys[S], rd), base, ctx.J)["dJ"], ctx.v).numpy()
                    replay[f"{bid}|{method}|{scope}|{S}"] = max(abs(T[i] - red[(bid, method, scope, S, p)])
                                                                for i, p in enumerate(pids))
        out["day2_replay_max_abs_T"] = replay
        g = cfg["smoke_gates"]
        for name, v in replay.items():
            tol = g["replay_orig_max_abs_T"] if name.startswith("orig|") else g["replay_reduced_max_abs_T"]
            if not v <= tol:
                fails.append(f"replay {name}: {v:.3g} > {tol}")
    if sha256_file(artifact_paths(cfg)[0]) != art_sha:
        fails.append("artifact changed during smoke")
    out["gate_failures"] = fails
    ctx.run.save_json("smoke.json", out)
    if fails:
        raise AssertionError(f"Day-3 calibration smoke failed: {fails}")
    return out
