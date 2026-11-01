"""Template-within-frame percentile bootstrap with fixed frame weights.

One resampler (`resample_theta`) is used both by the future confirmatory analysis (`template_bootstrap`) and by
every simulated dataset of the power plan, so the simulated and the real procedure are the same. A template's
value T_ft already averages its two groups (each group's U averages its 8 donors), so resampling templates carries
both groups and all their donors together."""
import numpy as np


def rng_for(seed, *key):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=tuple(int(k) for k in key)))


def resample_theta(T_list, B, rng):
    """T_list[f]: 1-D template values of frame f. Returns [B] bootstrap draws of theta* = (1/F) sum_f mean of n_f
    templates drawn with replacement within frame f. Frames are drawn in list order from one generator."""
    acc = np.zeros(B)
    for T in T_list:
        T = np.asarray(T, dtype=np.float64)
        n = len(T)
        idx = rng.integers(0, n, size=(B, n), dtype=np.int16 if n < 2 ** 15 else np.int64)
        acc += T[idx].mean(-1)
    return acc / len(T_list)


def percentile_bounds(ts, conf):
    """Separate one-sided bounds: lower = (1-conf) quantile, upper = conf quantile (exact, linear interpolation)."""
    return (np.quantile(ts, 1 - conf, axis=-1, method="linear"), np.quantile(ts, conf, axis=-1, method="linear"))


def template_bootstrap(T, B, seed, conf):
    """T: {frame: {template: value}} (theta = L^2 scale). Point estimate and one-sided bounds on theta and L."""
    frames = sorted(T)
    vals = [np.array([T[f][t] for t in sorted(T[f])]) for f in frames]
    ts = resample_theta(vals, B, rng_for(seed, 7))
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
