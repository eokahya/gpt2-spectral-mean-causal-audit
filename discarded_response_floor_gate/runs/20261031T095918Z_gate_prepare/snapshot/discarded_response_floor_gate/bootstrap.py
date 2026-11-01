"""Template-within-frame percentile bootstrap with fixed frame weights, by exact multinomial counts.

A template's value T_ft already averages its two groups (each group's U averages its 8 donors), so resampling
templates carries both groups and all their donors together."""
import numpy as np


def rng_for(seed, *key):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=tuple(int(k) for k in key)))


def count_matrix(n, B, rng):
    """[B, n] multinomial counts: n templates drawn with replacement, B times."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)


def theta_star(T_list, C_list):
    """T_list[f]: [S, n_f] template values; C_list[f]: [B, n_f] counts. Returns [S, B]: frame-weighted (1/F) mean
    over frames of the resampled frame means."""
    F = len(T_list)
    return sum((T @ C.T) / C.shape[1] for T, C in zip(T_list, C_list)) / F


def percentile_bounds(ts, conf):
    """Separate one-sided bounds: lower = (1-conf) quantile, upper = conf quantile (linear interpolation)."""
    return (np.quantile(ts, 1 - conf, axis=-1, method="linear"), np.quantile(ts, conf, axis=-1, method="linear"))


def template_bootstrap(T, B, seed, conf):
    """T: {frame: {template: value}} (theta = L^2 scale). Point estimate and one-sided bounds on theta and L."""
    frames = sorted(T)
    vals = [np.array([T[f][t] for t in sorted(T[f])])[None, :] for f in frames]
    C = [count_matrix(v.shape[1], B, rng_for(seed, 7, i)) for i, v in enumerate(vals)]
    ts = theta_star(vals, C)[0]
    if not np.isfinite(ts).all():
        raise ValueError("non-finite bootstrap draw")
    point = float(np.mean([v.mean() for v in vals]))
    lo, hi = percentile_bounds(ts, conf)
    return {"theta": point, "theta_lower": float(lo), "theta_upper": float(hi), "L": point ** 0.5,
            "L_lower": float(lo) ** 0.5, "L_upper": float(hi) ** 0.5, "B": B, "seed": seed, "conf_each": conf,
            "note": "lower and upper are separate one-sided bounds, not a two-sided interval"}


def confirm_decision(res, L0, valid=True):
    if not valid:
        return "EVALUATION_INVALID"
    if res["L_lower"] > L0:
        return "SUPPORTED_FLOOR"
    if res["L_upper"] <= L0:
        return "TARGET_FLOOR_RULED_OUT"
    return "INCONCLUSIVE"
