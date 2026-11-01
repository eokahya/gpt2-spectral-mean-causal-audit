"""Norm / pre-activation-energy matching of intervention deltas (PROTOCOL_DAY2 §C). CPU float64."""
import numpy as np
import torch

F64 = torch.float64


def l2_norm(delta):
    return delta.to(F64).norm(dim=-1)


def preact_norm(delta, Wc):
    """||delta_row @ Wc||: the original model's pre-activation change (no bias)."""
    return (delta.to(F64) @ Wc.to(F64)).norm(dim=-1)


def match_scale(native, target, mcfg):
    """Per-row scale = target / native, with the pre-specified degenerate and out-of-support rules."""
    native, target = np.asarray(native, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if not (np.isfinite(native).all() and np.isfinite(target).all()):
        raise ValueError("non-finite norm in matching input")
    degenerate = (native < mcfg["abs_floor"]) | (native < mcfg["rel_floor"] * target) | (target < mcfg["abs_floor"])
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(degenerate, np.nan, target / np.where(degenerate, 1.0, native))
    lo, hi = mcfg["oos_range"]
    oos = ~degenerate & ((scale < lo) | (scale > hi))
    return scale, degenerate, oos


def apply_match(delta, measure, target, mcfg, Wc=None):
    """Rescale delta rows so that measure(delta) == target. Asserts the match in float64 and float32.

    measure: 'l2' or 'preact'. Degenerate rows are returned as zero deltas (and must be dropped downstream).
    Returns (delta_f32_cpu, info dict with per-row arrays)."""
    d64 = delta.to(F64)
    norm = (lambda t: l2_norm(t)) if measure == "l2" else (lambda t: preact_norm(t, Wc))
    native = norm(d64).numpy()
    scale, degenerate, oos = match_scale(native, target, mcfg)
    s = torch.from_numpy(np.nan_to_num(scale, nan=0.0))
    matched = d64 * s[:, None]
    ok = ~degenerate
    got64 = norm(matched).numpy()
    got32 = norm(matched.float()).numpy()
    t = np.asarray(target, dtype=np.float64)
    if ok.any():
        rel64 = np.abs(got64[ok] - t[ok]) / t[ok]
        rel32 = np.abs(got32[ok] - t[ok]) / t[ok]
        if rel64.max() > mcfg["assert_rel_tol_f64"] or rel32.max() > mcfg["assert_rel_tol_f32"]:
            raise AssertionError(f"{measure} matching failed: rel64 {rel64.max():.2e}, rel32 {rel32.max():.2e}")
    else:
        rel64 = rel32 = np.array([0.0])
    if not torch.isfinite(matched).all():
        raise AssertionError("non-finite matched delta")
    return matched.float(), {"scale": scale, "degenerate": degenerate, "oos": oos, "native": native,
                             "target": t, "max_rel_err_f64": float(rel64.max()),
                             "max_rel_err_f32": float(rel32.max())}


def scale_quantiles(scale, degenerate):
    s = np.asarray(scale)[~np.asarray(degenerate)]
    if len(s) == 0:
        return None
    return dict(zip(("q0", "q05", "q50", "q95", "q100"), np.quantile(s, [0, 0.05, 0.5, 0.95, 1.0]).tolist()))
