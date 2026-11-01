"""Fixed-weight cell MSE, paired template-within-frame cluster bootstrap, primary gain G and its decision."""
import itertools

import numpy as np


def cell_keys(cfg, rows):
    dims = cfg["evaluation"]["cell_dims"]
    return [tuple(r[d] for d in dims) for r in rows]


def cell_weights(cfg, rows):
    """Product of per-dimension uniform weights over the design levels present in the split (all cells must be
    present: the design is complete by construction)."""
    dims = cfg["evaluation"]["cell_dims"]
    for d in dims:
        if cfg["evaluation"]["cell_weights"][d] != "uniform":
            raise ValueError(f"only uniform per-dimension weights are registered, got {d}")
    levels = [sorted({r[d] for r in rows}) for d in dims]
    cells = list(itertools.product(*levels))
    present = set(cell_keys(cfg, rows))
    if set(cells) != present:
        raise ValueError("incomplete design: some frame x direction x gap x alpha cell is empty")
    w = 1.0 / len(cells)
    return cells, np.full(len(cells), w)


def weighted_mse(cfg, rows, err2):
    cells, w = cell_weights(cfg, rows)
    keys = cell_keys(cfg, rows)
    err2 = np.asarray(err2, np.float64)
    means = np.array([err2[[k == c for k in keys]].mean() for c in cells])
    return float(means @ w)


def cluster_tables(cfg, rows, err2_list):
    """S[t, c] = sum of err2 of template t in cell c, N[t, c] = count; template frame labels."""
    cells, w = cell_weights(cfg, rows)
    cidx = {c: i for i, c in enumerate(cells)}
    templates = sorted({r["template"] for r in rows})
    tidx = {t: i for i, t in enumerate(templates)}
    N = np.zeros((len(templates), len(cells)))
    S = [np.zeros_like(N) for _ in err2_list]
    for i, (r, k) in enumerate(zip(rows, cell_keys(cfg, rows))):
        t, c = tidx[r["template"]], cidx[k]
        N[t, c] += 1
        for s_, e in zip(S, err2_list):
            s_[t, c] += e[i]
    frames = np.array([t.split("|")[0] for t in templates])
    return templates, frames, N, S, w


def multiplicity_draws(frames, B, seed):
    """[B, T] multiplicities: within each frame, its templates are resampled with replacement (fixed frames)."""
    rng = np.random.default_rng(seed)
    M = np.zeros((B, len(frames)))
    for f in sorted(set(frames)):
        idx = np.flatnonzero(frames == f)
        picks = idx[rng.integers(0, len(idx), (B, len(idx)))]
        for b in range(B):
            np.add.at(M[b], picks[b], 1)
    return M


def _mse_from(M, S, N, w):
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((M @ S) / (M @ N)) @ w


def gain(cfg, rows, err2_A, err2_B, B=None, seed=None):
    """G = 1 - MSE_A / MSE_B with fixed cell weights; paired cluster bootstrap. Never discards a draw: any
    non-finite or <= floor denominator (point or any draw) invalidates the ratio interval."""
    ev = cfg["evaluation"]
    B = ev["bootstrap_B"] if B is None else B
    seed = ev["bootstrap_seed"] if seed is None else seed
    floor, q = ev["mse_floor"], ev["one_sided_conf"]
    err2_A, err2_B = np.asarray(err2_A, np.float64), np.asarray(err2_B, np.float64)
    templates, frames, N, (SA, SB), w = cluster_tables(cfg, rows, [err2_A, err2_B])
    if (N.sum(0) == 0).any():
        raise ValueError("empty cell")
    for f in sorted(set(frames)):
        Nf = N[frames == f]
        if not (Nf[:, Nf.sum(0) > 0] > 0).all():
            raise ValueError(f"frame {f}: not every template has rows in every cell of its frame")
    ones = np.ones((1, len(templates)))
    mA, mB = float(_mse_from(ones, SA, N, w)[0]), float(_mse_from(ones, SB, N, w)[0])
    M = multiplicity_draws(frames, B, seed)
    dA, dB = _mse_from(M, SA, N, w), _mse_from(M, SB, N, w)
    diff = dB - dA
    out = {"mse_A": mA, "mse_B": mB, "mse_B_minus_A": mB - mA,
           "mse_B_minus_A_ci95_two_sided": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]
           if np.isfinite(diff).all() else None,
           "n_templates": len(templates), "n_rows": len(rows), "bootstrap_B": B, "bootstrap_seed": seed,
           "n_draws_denominator_invalid": int((~np.isfinite(dB) | (dB <= floor)).sum()),
           "point_denominator_invalid": bool(not np.isfinite(mB) or mB <= floor)}
    valid = not out["point_denominator_invalid"] and out["n_draws_denominator_invalid"] == 0 \
        and np.isfinite(dA).all()
    out["ratio_valid"] = bool(valid)
    if valid:
        G = 1.0 - dA / dB
        out.update(G=1.0 - mA / mB, G_lower_one_sided=float(np.quantile(G, 1 - q)),
                   G_upper_one_sided=float(np.quantile(G, q)), one_sided_conf=q)
    else:
        out.update(G=(1.0 - mA / mB) if np.isfinite(mB) and mB > floor else None,
                   G_lower_one_sided=None, G_upper_one_sided=None, one_sided_conf=q)
    return out


def decision(cfg, res):
    thr = cfg["evaluation"]["go_threshold"]
    if not res["ratio_valid"]:
        return "EVALUATION_INVALID"
    if res["G_lower_one_sided"] > thr:
        return "GO"
    if res["G_upper_one_sided"] <= thr:
        return "TARGET_GAIN_RULED_OUT"
    return "INCONCLUSIVE"
