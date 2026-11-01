"""Read-only numerical diagnostic for the v2 development identity-control failure (EVALUATION_INVALID).

Not part of the bound gate source (lives under notes/). It uses OLD Day-1 calibration prompts only (no development
prompt, no new scientific data), writes only under notes/gate_numerical_diagnostic/runs/, and never touches the
registration, manifests or runs of discarded_response_floor_gate/.

Question: is the development identity control `F_group(branch = x_extract) vs F_extract(unpatched)` failing because
of float32 layout numerics (padding / sequence length / batch size), or because of a hook bug? It separates:

  H  hook identity in the SAME layout: branch = that layout's own x -> must be exactly 0 if the hook is an identity;
  X  input-extraction drift: the MLP-branch input x at the final token, across layouts (no hook);
  G  logits drift across layouts with no hook (the whole forward, incl. the lm_head GEMM);
  M  lm_head GEMM shape alone: identical final hidden rows multiplied in row blocks of 1 / 19 / 48;
  D  the development control decomposed: D = [F_19(branch=x_mix) - F_19(unpatched)]  (injected-input drift)
                                            + [F_19(unpatched) - F_mix(unpatched)]   (layout drift, no hook);
  C  a CPU float64 single-prompt reference: |F_layout(f32) - F(f64)| for every layout.

Layouts for a target prompt p (all with the target's own last-token index):
  single   batch 1, no padding
  group19  19 identical copies (the development group-forward layout)
  same48   48 identical copies (extraction batch size, no padding)
  padK     batch 1, p right-padded with K pad tokens (attention mask 0), K = 1..4  (padding only)
  mix48    the target inside a 48-prompt batch of mixed-length old calibration prompts (padding + mixing,
           the development extraction layout)

    .venv/bin/python notes/gate_numerical_diagnostic/diagnose.py --device mps --n-targets 16 --cpu64-spots 3
"""
import argparse
import csv
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
from scp.common import get_device, load_config as load_cfg3, sha256_file  # noqa: E402
from scp.data import make_batch, read_csv, year_token_ids  # noqa: E402
from scp.model import load_model  # noqa: E402

F64 = torch.float64
MODEL_CFG = {"model": {"name": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                       "dtype": "float32", "device": "mps", "attn_implementation": "eager"}}


def padded_batch(id_lists, pad, device, extra_pad=0):
    """Right-padded batch; extra_pad adds K masked pad positions beyond the longest prompt."""
    b = make_batch([x + [pad] * extra_pad for x in id_lists], pad, device)
    if extra_pad:
        for i, x in enumerate(id_lists):
            b.mask[i, len(x):] = 0
        b.last_idx[:] = torch.tensor([len(x) - 1 for x in id_lists], device=device)
    return b


def run_layout(model, layer, batch, row, branch=None):
    """Logits [V] and branch input x [d] of one row of a batch; optional branch for every row."""
    with branch_mlp(model, layer) as w:
        o = forward(model, w, batch, branch=branch)
    return o["logits"][row].detach().cpu().to(F64), o["x_in_last"][row].detach().cpu().to(F64), o


def stats(a, b, J, ulp_ref=None):
    d = (a - b)
    dj = d[J]
    cj = dj - dj.mean()
    out = {"max_abs_all": float(d.abs().max()), "max_abs_J": float(dj.abs().max()),
           "centred_J_norm2_m": float(cj.pow(2).mean())}
    if ulp_ref is not None:
        out["max_abs_J_in_f32_ulps"] = float(dj.abs().max() / ulp_ref)
    return out


def x_stats(a, b):
    return {"max_abs": float((a - b).abs().max()), "rel_norm": float((a - b).norm() / b.norm().clamp_min(1e-300))}


def f32_ulp(v):
    return float(np.spacing(np.float32(abs(v))))


def diagnose(model, layer, pad, device, targets, mix_pool, J, model64=None, spots=0, pads=(1, 2, 3, 4)):
    rows = []
    for ti, p in enumerate(targets):
        n = len(p)
        res = {"target": ti, "len": n}
        single = padded_batch([p], pad, device)
        l1, x1, _ = run_layout(model, layer, single, 0)
        ulp = f32_ulp(float(l1[J].abs().max()))
        res["max_abs_logit_J"] = float(l1[J].abs().max())
        res["f32_ulp_at_max_logit"] = ulp
        g19 = padded_batch([p] * 19, pad, device)
        l19, x19, _ = run_layout(model, layer, g19, 0)
        s48 = padded_batch([p] * 48, pad, device)
        ls48, xs48, _ = run_layout(model, layer, s48, 0)
        # mix48: target at position 0, then 47 mixed-length old calibration prompts
        mix = [p] + [mix_pool[(ti * 47 + j) % len(mix_pool)] for j in range(47)]
        mb = padded_batch(mix, pad, device)
        lmix, xmix, _ = run_layout(model, layer, mb, 0)
        res["mix48_pad"] = int(max(len(x) for x in mix) - n)
        # H: hook identity in the same layout
        h = {}
        for name, b, l in (("single", single, l1), ("group19", g19, l19), ("mix48", mb, lmix)):
            with branch_mlp(model, layer) as w:  # every row's own branch input in this very layout
                xb = forward(model, w, b)["x_in_last"].detach()
            lh, _, _ = run_layout(model, layer, b, 0, branch=xb)
            h[name] = stats(lh, l, J, ulp)
        res["H_hook_identity_same_layout"] = h
        # X / G: no-hook drift across layouts (reference: single)
        res["X_input_drift_vs_single"] = {"group19": x_stats(x19, x1), "same48": x_stats(xs48, x1),
                                          "mix48": x_stats(xmix, x1)}
        res["G_logit_drift_vs_single"] = {"group19": stats(l19, l1, J, ulp), "same48": stats(ls48, l1, J, ulp),
                                          "mix48": stats(lmix, l1, J, ulp)}
        padk = {}
        for k in pads:
            lk, xk, _ = run_layout(model, layer, padded_batch([p], pad, device, extra_pad=k), 0)
            padk[f"pad{k}"] = {"x": x_stats(xk, x1), "logits": stats(lk, l1, J, ulp)}
        res["padding_only_vs_single"] = padk
        # D: the development control and its decomposition
        dtype = next(model.parameters()).dtype
        lmixed_inj, _, _ = run_layout(model, layer, g19, 0, branch=xmix.to(dtype).to(device).expand(19, -1))
        res["D_dev_control"] = {
            "total_F19_branch_xmix_vs_Fmix": stats(lmixed_inj, lmix, J, ulp),
            "part_injected_input_drift_F19_branch_xmix_vs_F19": stats(lmixed_inj, l19, J, ulp),
            "part_layout_drift_no_hook_F19_vs_Fmix": stats(l19, lmix, J, ulp),
            "decomposition_residual_max_abs": float(((lmixed_inj - l19) + (l19 - lmix) - (lmixed_inj - lmix)).abs().max())}
        # M: lm_head GEMM row-block shape only (same final hidden row)
        with branch_mlp(model, layer) as w:
            o = forward(model, w, single)
        hrow = o["hidden_all"][0, single.last_idx[0]].detach()
        with torch.inference_mode():
            m1 = model.lm_head(hrow[None])[0].cpu().to(F64)
            m19 = model.lm_head(hrow[None].expand(19, -1).contiguous())[0].cpu().to(F64)
            m48 = model.lm_head(hrow[None].expand(48, -1).contiguous())[0].cpu().to(F64)
        res["M_lm_head_gemm_shape_vs_1row"] = {"rows19": stats(m19, m1, J, ulp), "rows48": stats(m48, m1, J, ulp)}
        # C: CPU float64 reference for the first `spots` targets
        if model64 is not None and ti < spots:
            b64 = padded_batch([p], pad, torch.device("cpu"))
            l64, x64, _ = run_layout(model64, layer, b64, 0)
            res["C_f32_vs_cpu64"] = {"single": stats(l1, l64, J, ulp), "group19": stats(l19, l64, J, ulp),
                                     "same48": stats(ls48, l64, J, ulp), "mix48": stats(lmix, l64, J, ulp),
                                     "x_single_vs_64": x_stats(x1, x64), "x_mix48_vs_64": x_stats(xmix, x64)}
        rows.append(res)
    return rows


def summarise(rows):
    def mx(path):
        vals = []
        for r in rows:
            v = r
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None:
                vals.append(v)
        return max(vals) if vals else None
    s = {"n_targets": len(rows),
         "H_hook_identity_same_layout_max_abs_J": {k: mx(["H_hook_identity_same_layout", k, "max_abs_J"])
                                                   for k in ("single", "group19", "mix48")},
         "G_logit_drift_vs_single_max_abs_J": {k: mx(["G_logit_drift_vs_single", k, "max_abs_J"])
                                              for k in ("group19", "same48", "mix48")},
         "X_input_drift_vs_single_max_abs": {k: mx(["X_input_drift_vs_single", k, "max_abs"])
                                             for k in ("group19", "same48", "mix48")},
         "padding_only_logits_max_abs_J": {f"pad{k}": mx(["padding_only_vs_single", f"pad{k}", "logits", "max_abs_J"])
                                           for k in (1, 2, 3, 4)},
         "D_total_max_abs_J": mx(["D_dev_control", "total_F19_branch_xmix_vs_Fmix", "max_abs_J"]),
         "D_injected_input_part_max_abs_J": mx(["D_dev_control", "part_injected_input_drift_F19_branch_xmix_vs_F19", "max_abs_J"]),
         "D_layout_part_max_abs_J": mx(["D_dev_control", "part_layout_drift_no_hook_F19_vs_Fmix", "max_abs_J"]),
         "M_gemm_shape_max_abs_J": {k: mx(["M_lm_head_gemm_shape_vs_1row", k, "max_abs_J"]) for k in ("rows19", "rows48")},
         "C_f32_vs_cpu64_max_abs_J": {k: mx(["C_f32_vs_cpu64", k, "max_abs_J"]) for k in ("single", "group19", "same48", "mix48")},
         "max_abs_logit_J": mx(["max_abs_logit_J"]), "f32_ulp_at_max_logit": mx(["f32_ulp_at_max_logit"]),
         "reading_guide": ("H != 0 in any layout -> hook not an identity (bug). H == 0 with G/X > 0 for padded or mixed "
                           "layouts and 0 for unpadded ones -> float32 layout numerics. M isolates the lm_head GEMM "
                           "shape. C shows how far every f32 layout is from a float64 reference.")}
    return s


def old_cal_prompts(seed):
    pairs = read_csv(ROOT / "manifests" / "pilot_pairs_calibration.csv")
    pool = read_csv(ROOT / "manifests" / "moment_pool_calibration.csv")
    rng = np.random.default_rng(seed)
    recs = [p["rec_ids"] for p in pairs]
    mix = [q["ids"] for q in pool]
    return [recs[i] for i in rng.permutation(len(recs))], [mix[i] for i in rng.permutation(len(mix))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-targets", type=int, default=16)
    ap.add_argument("--cpu64-spots", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261033)
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
    targets, mix = old_cal_prompts(a.seed)
    # targets: alternate the two old-calibration lengths (12, 14) so both padded and unpadded mix48 cases occur
    t12 = [t for t in targets if len(t) == 12]
    t14 = [t for t in targets if len(t) == 14]
    chosen = [x for pair in zip(t12, t14) for x in pair][:a.n_targets]
    rows = diagnose(model, 9, tok.eos_token_id, device, chosen, mix, J, model64, a.cpu64_spots)
    out = Path(a.out) / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_diag"
    out.mkdir(parents=True, exist_ok=False)
    summary = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "device": str(device),
               "torch": torch.__version__, "data": "old Day-1 calibration prompts only (pilot recipients + moment pool)",
               "script_sha256": sha256_file(Path(__file__)), "args": vars(a), **summarise(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_target.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(summary, indent=2))
    print(out)


if __name__ == "__main__":
    main()
