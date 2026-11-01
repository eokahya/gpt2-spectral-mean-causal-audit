"""Read-only numerical diagnostic for the v2 development identity-control failure (EVALUATION_INVALID). Version 2.

Not part of the bound gate source (lives under notes/). It uses OLD Day-1 calibration prompts only (pilot recipients,
retained donors, same-year and wrong donors, and the moment pool: lengths 12 and 14). It uses no development prompt
and writes only under notes/gate_numerical_diagnostic/runs/. The first run (20261031T103857Z_diag) is kept unchanged.

Why v2: in the first run every length-12 target sat at row 0 of its mixed batch with natural padding 2. Pads 1, 3, 4
and other batch rows, which occur in the development extraction (pads 0-4; recipients at rows 0..47), were not
exercised. Here every target is evaluated in mixed 48-prompt batches for each padding level P in --pads and each
target row in --rows:
  * the other 47 prompts all have length <= the target length (so the natural maximum is the target length);
  * P extra masked pad positions are appended to every row, so the target's padding is exactly P;
  * an additional 'natural' variant for length-12 targets mixes in length-14 prompts (natural padding 2, no extra
    pad), as in development extraction.
Both padded (P > 0) and unpadded (P = 0) cases are guaranteed and counted (asserted).

Quantities (logit differences reported raw, in float32 ulps, and as ||Delta C_J||^2_m):
  H_mix  hook identity in the SAME mixed layout (branch = every row's own x)            -> 0 unless the hook is wrong
  D      development control: F_19(branch = x_mix[row]) vs F_mix[row], split exactly into
         injected-input part F_19(branch=x_mix) - F_19 and layout part F_19 - F_mix (no hook)
  X      branch-input drift x_mix[row] vs x_19 (input-extraction drift, no hook)
  per target: H in single / group19; drift vs single for group19 / same48; padding-only (batch 1, K masked pads);
  M      lm_head GEMM shape only (same final hidden row in row blocks of 1 / 19 / 48)
  CPU64  for the first --cpu64-spots targets, the SAME suites in float64 on CPU (same-layout H, cross-layout X/G/D)
         plus every float32 layout vs the float64 single-prompt reference.

A finite set of layouts can only rule out the mechanisms it actually exercises; it cannot exclude every
padding-conditional or row-conditional bug.

    .venv/bin/python notes/gate_numerical_diagnostic/diagnose.py --device mps --n-targets 8 --cpu64-spots 2
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import discarded_response_floor_gate  # noqa: E402,F401  (HF cache/offline env, thread caps)
from discarded_response_floor_gate.hook import branch_mlp, forward  # noqa: E402
from scp.common import get_device, sha256_file  # noqa: E402
from scp.data import make_batch, read_csv, year_token_ids  # noqa: E402
from scp.model import load_model  # noqa: E402

F64 = torch.float64
MODEL_CFG = {"model": {"name": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                       "dtype": "float32", "device": "mps", "attn_implementation": "eager"}}


def padded_batch(id_lists, pad, device, extra_pad=0):
    """Right-padded batch; extra_pad appends K masked pad positions to every row beyond the longest prompt."""
    b = make_batch([x + [pad] * extra_pad for x in id_lists], pad, device)
    if extra_pad:
        for i, x in enumerate(id_lists):
            b.mask[i, len(x):] = 0
        b.last_idx[:] = torch.tensor([len(x) - 1 for x in id_lists], device=device)
    return b


def fwd(model, layer, batch, branch=None):
    with branch_mlp(model, layer) as w:
        o = forward(model, w, batch, branch=branch)
    return o["logits"].detach().cpu().to(F64), o["x_in_last"].detach().cpu().to(F64), o


def mdtype(model):
    return next(model.parameters()).dtype


def stats(a, b, J, ulp=None):
    d = a - b
    dj = d[J]
    cj = dj - dj.mean()
    out = {"max_abs_all": float(d.abs().max()), "max_abs_J": float(dj.abs().max()),
           "centred_J_norm2_m": float(cj.pow(2).mean())}
    if ulp:
        out["max_abs_J_in_f32_ulps"] = float(dj.abs().max() / ulp)
    return out


def x_stats(a, b):
    return {"max_abs": float((a - b).abs().max()), "rel_norm": float((a - b).norm() / b.norm().clamp_min(1e-300))}


def f32_ulp(v):
    return float(np.spacing(np.float32(abs(v))))


def mixed_cases(target, pool, pads, rows, rng):
    """[(label, P, row, id_lists, extra_pad)]: controlled cases (others no longer than the target, P extra pads) and,
    for a target shorter than some pool prompt, one natural case (longer prompts mixed in, no extra pad)."""
    n = len(target)
    shorter_eq = [x for x in pool if len(x) <= n]
    longer = [x for x in pool if len(x) > n]
    out = []
    for P in pads:
        for r in rows:
            others = [shorter_eq[i] for i in rng.choice(len(shorter_eq), 47, replace=False)]
            ids = others[:r] + [target] + others[r:]
            out.append((f"controlled_P{P}_row{r}", P, r, ids, P))
    if longer:
        for r in rows:
            k = min(8, len(longer))
            others = [longer[i] for i in rng.choice(len(longer), k, replace=False)] + \
                     [shorter_eq[i] for i in rng.choice(len(shorter_eq), 47 - k, replace=False)]
            others = [others[i] for i in rng.permutation(47)]
            ids = others[:r] + [target] + others[r:]
            out.append((f"natural_row{r}", max(len(x) for x in ids) - n, r, ids, 0))
    return out


def mixed_suite(model, layer, pad, device, target, cases, J, ulp, l19, x19):
    """H_mix, D and X for every mixed case of one target."""
    res = []
    dtype = mdtype(model)
    for label, P, r, ids, extra in cases:
        b = padded_batch(ids, pad, device, extra)
        assert int(b.last_idx[r]) == len(target) - 1 and int(b.mask[r].sum()) == len(target)
        assert b.ids.shape[1] - len(target) == P, (label, P)
        lmix, xmix, _ = fwd(model, layer, b)
        lh, _, _ = fwd(model, layer, b, branch=xmix.to(dtype).to(device))
        g19 = padded_batch([target] * 19, pad, device)
        linj, _, _ = fwd(model, layer, g19, branch=xmix[r].to(dtype).to(device).expand(19, -1))
        res.append({"case": label, "pad": int(P), "row": int(r),
                    "H_mix_hook_identity_same_layout": stats(lh[r], lmix[r], J, ulp),
                    "H_mix_all_rows_max_abs": float((lh - lmix).abs().max()),
                    "X_input_drift_mix_vs_group19": x_stats(xmix[r], x19),
                    "D_total_F19_branch_xmix_vs_Fmix": stats(linj[0], lmix[r], J, ulp),
                    "D_injected_input_part": stats(linj[0], l19, J, ulp),
                    "D_layout_part_no_hook_F19_vs_Fmix": stats(l19, lmix[r], J, ulp),
                    "D_decomposition_residual": float(((linj[0] - l19) + (l19 - lmix[r]) - (linj[0] - lmix[r])).abs().max())})
    return res


def target_suite(model, layer, pad, device, target, cases, J, pads_only=(1, 2, 3, 4)):
    dtype = mdtype(model)
    single = padded_batch([target], pad, device)
    l1, x1, o1 = fwd(model, layer, single)
    ulp = f32_ulp(float(l1[0][J].abs().max())) if dtype == torch.float32 else None
    g19 = padded_batch([target] * 19, pad, device)
    l19, x19, _ = fwd(model, layer, g19)
    s48 = padded_batch([target] * 48, pad, device)
    l48, x48, _ = fwd(model, layer, s48)
    out = {"len": len(target), "dtype": str(dtype), "max_abs_logit_J": float(l1[0][J].abs().max()), "f32_ulp": ulp}
    h = {}
    for name, b, l, x in (("single", single, l1, x1), ("group19", g19, l19, x19)):
        lh, _, _ = fwd(model, layer, b, branch=x.to(dtype).to(device))
        h[name] = stats(lh[0], l[0], J, ulp)
    out["H_hook_identity_same_layout"] = h
    out["G_drift_vs_single"] = {"group19": stats(l19[0], l1[0], J, ulp), "same48": stats(l48[0], l1[0], J, ulp)}
    out["X_drift_vs_single"] = {"group19": x_stats(x19[0], x1[0]), "same48": x_stats(x48[0], x1[0])}
    out["padding_only_vs_single"] = {}
    for k in pads_only:
        lk, xk, _ = fwd(model, layer, padded_batch([target], pad, device, extra_pad=k))
        out["padding_only_vs_single"][f"pad{k}"] = {"x": x_stats(xk[0], x1[0]), "logits": stats(lk[0], l1[0], J, ulp)}
    hrow = o1["hidden_all"][0, single.last_idx[0]].detach()
    with torch.inference_mode():
        m1 = model.lm_head(hrow[None])[0].cpu().to(F64)
        mm = {n: model.lm_head(hrow[None].expand(n, -1).contiguous())[0].cpu().to(F64) for n in (19, 48)}
    out["M_lm_head_gemm_shape_vs_1row"] = {f"rows{n}": stats(v, m1, J, ulp) for n, v in mm.items()}
    out["mixed"] = mixed_suite(model, layer, pad, device, target, cases, J, ulp, l19[0], x19[0])
    out["_single_logits"] = l1[0]
    out["_layout_logits"] = {"group19": l19[0], "same48": l48[0]}
    return out


def diagnose(model, layer, pad, device, targets, pool, J, model64=None, spots=0, pads=(0, 1, 2, 3, 4),
             rows=(0, 24, 47), seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for ti, t in enumerate(targets):
        cases = mixed_cases(t, pool, pads, rows, rng)
        r32 = target_suite(model, layer, pad, device, t, cases, J)
        res = {"target": ti, "len": len(t), "float32": r32}
        if model64 is not None and ti < spots:
            r64 = target_suite(model64, layer, pad, torch.device("cpu"), t, cases, J)
            ref = r64["_single_logits"]
            ulp = r32["f32_ulp"]
            res["float64_cpu"] = r64
            res["f32_vs_cpu64_single_ref"] = {"single": stats(r32["_single_logits"], ref, J, ulp),
                                              **{k: stats(v, ref, J, ulp) for k, v in r32["_layout_logits"].items()}}
        results.append(res)
    for res in results:
        for k in ("float32", "float64_cpu"):
            if k in res:
                res[k].pop("_single_logits", None)
                res[k].pop("_layout_logits", None)
    return results


def _cases(results, key):
    return [c for r in results if key in r for c in r[key]["mixed"]]


def summarise(results):
    def mx(vals):
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    out = {"n_targets": len(results)}
    for key in ("float32", "float64_cpu"):
        cs = _cases(results, key)
        if not cs:
            continue
        padded = [c for c in cs if c["pad"] > 0]
        unpadded = [c for c in cs if c["pad"] == 0]
        by_pad = {}
        for c in cs:
            by_pad.setdefault(c["pad"], []).append(c)
        rs = [r[key] for r in results if key in r]
        out[key] = {
            "n_mixed_cases": len(cs), "n_padded": len(padded), "n_unpadded": len(unpadded),
            "H_hook_identity_single_group19_max_abs": mx(r["H_hook_identity_same_layout"][n]["max_abs_all"]
                                                         for r in rs for n in ("single", "group19")),
            "H_mix_all_rows_max_abs": mx(c["H_mix_all_rows_max_abs"] for c in cs),
            "D_total_by_pad_max_abs_J": {str(p): mx(c["D_total_F19_branch_xmix_vs_Fmix"]["max_abs_J"] for c in v)
                                         for p, v in sorted(by_pad.items())},
            "D_total_by_pad_n_nonzero": {str(p): sum(c["D_total_F19_branch_xmix_vs_Fmix"]["max_abs_all"] > 0 for c in v)
                                         for p, v in sorted(by_pad.items())},
            "D_injected_input_max_abs_J": mx(c["D_injected_input_part"]["max_abs_J"] for c in cs),
            "D_layout_part_max_abs_J": mx(c["D_layout_part_no_hook_F19_vs_Fmix"]["max_abs_J"] for c in cs),
            "X_input_drift_mix_vs_group19_max_abs": mx(c["X_input_drift_mix_vs_group19"]["max_abs"] for c in cs),
            "G_drift_vs_single_max_abs_J": {n: mx(r["G_drift_vs_single"][n]["max_abs_J"] for r in rs) for n in ("group19", "same48")},
            "padding_only_logits_max_abs_J": {k: mx(r["padding_only_vs_single"][k]["logits"]["max_abs_J"] for r in rs)
                                              for k in ("pad1", "pad2", "pad3", "pad4")},
            "M_gemm_shape_max_abs_J": {k: mx(r["M_lm_head_gemm_shape_vs_1row"][k]["max_abs_J"] for r in rs)
                                       for k in ("rows19", "rows48")},
            "max_abs_logit_J": mx(r["max_abs_logit_J"] for r in rs)}
    sp = [r["f32_vs_cpu64_single_ref"] for r in results if "f32_vs_cpu64_single_ref" in r]
    if sp:
        out["f32_layouts_vs_cpu64_single_max_abs_J"] = {k: mx(s[k]["max_abs_J"] for s in sp) for k in sp[0]}
    out["reading_guide"] = (
        "H (single/group19/mix, float32 and float64) != 0 -> the hook is not an identity in that layout. "
        "D by pad/row localises when the development control is non-zero; its two parts separate the injected "
        "extraction input from the forward layout. float64 cross-layout values show whether the effect is "
        "precision-related. Finite layouts only rule out the mechanisms they exercise.")
    return out


def old_cal_pool():
    pairs = read_csv(ROOT / "manifests" / "pilot_pairs_calibration.csv")
    pool = read_csv(ROOT / "manifests" / "moment_pool_calibration.csv")
    roles = ("rec_ids", "don_ids", "same_year_ids", "wrong_donor_ids")
    seen, out = set(), []
    for x in [p[k] for p in pairs for k in roles] + [q["ids"] for q in pool]:
        if tuple(x) not in seen:
            seen.add(tuple(x))
            out.append(x)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--cpu64-spots", type=int, default=2)
    ap.add_argument("--pads", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rows", type=int, nargs="+", default=[0, 24, 47])
    ap.add_argument("--seed", type=int, default=20261034)
    ap.add_argument("--out", default=str(HERE / "runs"))
    a = ap.parse_args(argv)
    device = get_device(a.device)  # never falls back silently
    model, tok = load_model(MODEL_CFG, device)
    J = year_token_ids(tok)
    model64 = None
    if a.cpu64_spots:
        c = json.loads(json.dumps(MODEL_CFG))
        c["model"]["dtype"] = "float64"
        model64, _ = load_model(c, torch.device("cpu"))
    pool = old_cal_pool()
    rng = np.random.default_rng(a.seed)
    by_len = {}
    for x in pool:
        by_len.setdefault(len(x), []).append(x)
    lens = sorted(by_len)
    # targets alternate across lengths; the shortest-length targets also get a natural-padding variant
    chosen = []
    for i in range(a.n_targets):
        L = lens[i % len(lens)]
        chosen.append(by_len[L][rng.integers(len(by_len[L]))])
    results = diagnose(model, 9, tok.eos_token_id, device, chosen, pool, J, model64, a.cpu64_spots,
                       tuple(a.pads), tuple(a.rows), a.seed)
    s = summarise(results)
    assert s["float32"]["n_padded"] > 0 and s["float32"]["n_unpadded"] > 0, "need padded and unpadded cases"
    out = Path(a.out) / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_diag_v2"
    out.mkdir(parents=True, exist_ok=False)
    summary = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "device": str(device),
               "torch": torch.__version__, "data": "old Day-1 calibration prompts only (all pilot roles + moment pool)",
               "pool_lengths": {str(k): len(v) for k, v in by_len.items()},
               "script_sha256": sha256_file(Path(__file__)), "args": vars(a), **s}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_target.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(summary, indent=2))
    print(out)


if __name__ == "__main__":
    main()
