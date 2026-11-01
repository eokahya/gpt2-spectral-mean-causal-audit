"""Pre-specified Gamma power planning (GPT Pro addendum §4-5), bounded memory and a hard compute cap."""
import time

import numpy as np
from scipy.stats import beta

from .bootstrap import count_matrix, percentile_bounds, rng_for, theta_star

SCENARIOS = [(1, "kappa_v"), (4, "kappa_v"), (1, "kappa_r2_v"), (4, "kappa_r2_v")]


def scenario_name(kappa, mode):
    return f"kappa{kappa}_{'v' if mode == 'kappa_v' else 'r2v'}"


def guard(m, v, noise_T, g):
    """Deterministic moment guard. 'invalid' = non-finite or negative (measurement problem); 'near_zero_floor' = a
    valid but numerically indistinguishable-from-zero frame mean; 'zero_variance' = degenerate spread. Either of the
    last two makes the Gamma family not estimable (POWER_NOT_ESTABLISHED), not an invalid measurement."""
    out = {}
    floor = max(g["moment_abs_floor"], g["moment_over_noise_min"] * noise_T)
    for f in sorted(m):
        if not (np.isfinite(m[f]) and np.isfinite(v[f])) or m[f] < 0 or v[f] < 0:
            out[f] = "invalid"
        elif m[f] <= floor:
            out[f] = "near_zero_floor"
        elif v[f] <= g["variance_rel_floor"] * m[f] ** 2:
            out[f] = "zero_variance"
        else:
            out[f] = "ok"
    return {"per_frame": out, "floor": floor, "ok": all(s == "ok" for s in out.values()),
            "invalid": any(s == "invalid" for s in out.values())}


def scenario_moments(m, v, mbar, theta, kappa, mode):
    r = theta / mbar
    mu = {f: r * m[f] for f in m}
    V = {f: kappa * (v[f] if mode == "kappa_v" else r * r * v[f]) for f in v}
    return mu, V, r


def gamma_params(mu, V):
    return mu * mu / V, V / mu  # shape, scale


def clopper_pearson(x, n, alpha):
    lo = 0.0 if x == 0 else float(beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - alpha / 2, x + 1, n - x))
    return lo, hi


def simulate_one(frames, mu, V, n_t, C, outer, chunk, theta0, conf, rngs, deadline):
    """Returns (successes, completed sims). success = one-sided lower bound on theta > theta0. One generator per
    frame, so the results do not depend on the memory chunk size."""
    shapes = [gamma_params(mu[f], V[f]) for f in frames]
    succ = done = 0
    while done < outer:
        if deadline is not None and time.monotonic() > deadline:
            break
        S = min(chunk, outer - done)
        T = [rng.gamma(sh, sc, size=(S, n_t)) for rng, (sh, sc) in zip(rngs, shapes)]
        lo, _ = percentile_bounds(theta_star(T, C), conf)
        succ += int((lo > theta0).sum())
        done += S
    return succ, done


def run_power(cfg, m, v, mbar, noise_T, cap_s=None, outer=None, B=None):
    p = cfg["power"]
    outer = p["outer_sims"] if outer is None else outer
    B = p["bootstrap_B"] if B is None else B
    cap_s = cfg["budget"]["stats_cap_s"] if cap_s is None else cap_s
    t0 = time.monotonic()
    deadline = t0 + cap_s
    frames = sorted(m)
    g = guard(m, v, noise_T, p["guards"])
    res = {"guard": g, "outer_sims": outer, "bootstrap_B": B, "seed": p["seed"], "rows": [],
           "prescribed": outer == p["outer_sims"] and B == p["bootstrap_B"]}
    if not g["ok"]:
        res.update(decision="POWER_NOT_ESTABLISHED", reason=f"moment guard: {g['per_frame']}", complete=False,
                   stats_runtime_s=time.monotonic() - t0)
        return res
    n_comp = len(p["candidates"]) * len(SCENARIOS) * 2
    alpha = p["cp_family_alpha"] / n_comp
    complete = True
    for iN, N in enumerate(p["candidates"]):
        n_t = N // (p["groups_per_template"] * p["n_frames"])
        C = [count_matrix(n_t, B, rng_for(p["seed"], 1, iN, i)) for i in range(len(frames))]
        for iS, (kappa, mode) in enumerate(SCENARIOS):
            row = {"N": N, "templates_per_frame": n_t, "scenario": scenario_name(kappa, mode)}
            for iH, (hyp, theta) in enumerate((("null", p["theta0"]), ("alt", p["theta1"]))):
                mu, V, r = scenario_moments(m, v, mbar, theta, kappa, mode)
                x, n = simulate_one(frames, mu, V, n_t, C, outer, p["chunk"], p["theta0"], p["one_sided_conf"],
                                    [rng_for(p["seed"], 2, iN, iS, iH, i) for i in range(len(frames))], deadline)
                lo, hi = clopper_pearson(x, n, alpha) if n else (None, None)
                row[hyp] = {"theta": theta, "r": r, "successes": x, "n": n, "rate": x / n if n else None,
                            "cp_lo": lo, "cp_hi": hi}
                complete &= n == outer
            res["rows"].append(row)
            if not complete:
                break
        if not complete:
            break
    res["cp_alpha_per_comparison"] = alpha
    res["n_comparisons"] = n_comp
    res["complete"] = complete
    res["stats_runtime_s"] = time.monotonic() - t0
    res.update(decide(cfg, res["rows"], complete and res["prescribed"]))
    return res


def decide(cfg, rows, complete):
    p = cfg["power"]
    if not complete:
        return {"decision": "POWER_NOT_ESTABLISHED",
                "reason": "planning compute cap or non-prescribed simulation size; partial outputs are not the prescribed power result"}
    by_N = {}
    for r in rows:
        by_N.setdefault(r["N"], []).append(r)
    for N in sorted(by_N):
        if all(r["alt"]["cp_lo"] >= p["power_target"] and r["null"]["cp_hi"] <= p["null_fp_alarm"] for r in by_N[N]):
            return {"decision": "POWER_PLAN_CANDIDATE", "candidate_N": N, "reason": "smallest N meeting both bounds in all scenarios"}
    if all(any(r["alt"]["cp_hi"] < p["power_target"] for r in by_N[N]) for N in by_N):
        return {"decision": "NO_GO_BUDGET", "reason": "every N has a scenario whose power upper bound is below the target"}
    return {"decision": "POWER_NOT_ESTABLISHED", "reason": "no N meets both bounds in all scenarios, and not every N is clearly underpowered"}
