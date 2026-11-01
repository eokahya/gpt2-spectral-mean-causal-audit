"""Pre-specified Gamma power planning (GPT Pro addendum §4-5), bounded memory and a hard compute cap.

Every simulated dataset j of every (N, scenario, hypothesis) has its own deterministic data stream
(2, iN, iS, iH, j) and its own independent bootstrap stream (3, iN, iS, iH, j); the bootstrap is the same resampler
as the confirmatory analysis (B = 10,000, exact linear percentiles). Memory per dataset is one [B, n_t] index array
per frame."""
import time

import numpy as np
from scipy.stats import beta

from .bootstrap import percentile_bounds, resample_theta, rng_for

SCENARIOS = [(1, "kappa_v"), (4, "kappa_v"), (1, "kappa_r2_v"), (4, "kappa_r2_v")]
HYPOTHESES = (("null", "theta0"), ("alt", "theta1"))


def scenario_name(kappa, mode):
    return f"kappa{kappa}_{'v' if mode == 'kappa_v' else 'r2v'}"


def guard(m, v, noise_T, g):
    """Deterministic moment guard. 'invalid' = non-finite or negative moments (a measurement problem ->
    EVALUATION_INVALID); 'near_zero_floor' / 'zero_variance' = valid measurements for which the Gamma family is not
    estimable (-> POWER_NOT_ESTABLISHED)."""
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


def simulate_dataset(frames, shapes, n_t, B, theta0, conf, seed, key):
    """One simulated confirmatory dataset: Gamma template values, then the confirmatory bootstrap with its own
    stream. Returns (success, lower bound)."""
    drng = rng_for(seed, 2, *key)
    T = [drng.gamma(sh, sc, size=n_t) for sh, sc in shapes]
    lo, _ = percentile_bounds(resample_theta(T, B, rng_for(seed, 3, *key)), conf)
    return bool(lo > theta0), float(lo)


def simulate(frames, mu, V, n_t, outer, B, theta0, conf, seed, key, deadline, start=0):
    """Datasets start..outer-1. Returns (successes, completed)."""
    shapes = [gamma_params(mu[f], V[f]) for f in frames]
    succ = done = 0
    for j in range(start, outer):
        if deadline is not None and time.monotonic() > deadline:
            break
        s, _ = simulate_dataset(frames, shapes, n_t, B, theta0, conf, seed, (*key, j))
        succ += s
        done += 1
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
    if g["invalid"]:
        res.update(decision="EVALUATION_INVALID", reason=f"invalid measured moments: {g['per_frame']}",
                   complete=False, stats_runtime_s=time.monotonic() - t0)
        return res
    if not g["ok"]:
        res.update(decision="POWER_NOT_ESTABLISHED", reason=f"moment guard (valid but not estimable): {g['per_frame']}",
                   complete=False, stats_runtime_s=time.monotonic() - t0)
        return res
    n_comp = len(p["candidates"]) * len(SCENARIOS) * len(HYPOTHESES)
    alpha = p["cp_family_alpha"] / n_comp
    stopped = False
    for iN, N in enumerate(p["candidates"]):
        n_t = N // (p["groups_per_template"] * p["n_frames"])
        for iS, (kappa, mode) in enumerate(SCENARIOS):
            row = {"N": N, "templates_per_frame": n_t, "scenario": scenario_name(kappa, mode)}
            for iH, (hyp, key) in enumerate(HYPOTHESES):
                theta = p[key]
                mu, V, r = scenario_moments(m, v, mbar, theta, kappa, mode)
                x, n = simulate(frames, mu, V, n_t, outer, B, p["theta0"], p["one_sided_conf"], p["seed"],
                                (iN, iS, iH), deadline)
                lo, hi = clopper_pearson(x, n, alpha) if n else (None, None)
                row[hyp] = {"theta": theta, "r": r, "successes": x, "n": n, "rate": x / n if n else None,
                            "cp_lo": lo, "cp_hi": hi}
                stopped |= n < outer
            res["rows"].append(row)
            if stopped:
                break
        if stopped:
            break
    res.update(cp_alpha_per_comparison=alpha, n_comparisons=n_comp, complete=not stopped,
               stats_runtime_s=time.monotonic() - t0)
    if stopped:
        res.update(decision="POWER_NOT_ESTABLISHED",
                   reason="planning compute cap reached; partial outputs are not the prescribed power result")
    else:
        res.update(decide(cfg, res["rows"], outer, res["prescribed"]))
    return res


def validate_rows(cfg, rows, outer):
    """All 5 candidates x 4 scenarios x 2 hypotheses present exactly once, each with n == outer."""
    p = cfg["power"]
    want = {(N, scenario_name(k, md)) for N in p["candidates"] for k, md in SCENARIOS}
    keys = [(r.get("N"), r.get("scenario")) for r in rows]
    problems = []
    if len(keys) != len(set(keys)):
        problems.append("duplicate (N, scenario) rows")
    if set(keys) != want:
        problems.append(f"missing or unexpected rows: {sorted(want ^ set(keys), key=str)}")
    for r in rows:
        for h, _ in HYPOTHESES:
            d = r.get(h)
            if not d or d.get("n") != outer or d.get("cp_lo") is None or d.get("cp_hi") is None:
                problems.append(f"incomplete {h} rates for {r.get('N')}/{r.get('scenario')}")
    return problems


def decide(cfg, rows, outer, prescribed=True):
    p = cfg["power"]
    if not prescribed:
        return {"decision": "POWER_NOT_ESTABLISHED", "reason": "non-prescribed simulation size"}
    problems = validate_rows(cfg, rows, outer)
    if problems:
        return {"decision": "POWER_NOT_ESTABLISHED", "reason": f"power table incomplete: {problems}"}
    by_N = {}
    for r in rows:
        by_N.setdefault(r["N"], []).append(r)
    for N in sorted(by_N):
        if all(r["alt"]["cp_lo"] >= p["power_target"] and r["null"]["cp_hi"] <= p["null_fp_alarm"] for r in by_N[N]):
            return {"decision": "POWER_PLAN_CANDIDATE", "candidate_N": N,
                    "reason": "smallest N meeting both bounds in all scenarios"}
    if all(any(r["alt"]["cp_hi"] < p["power_target"] for r in by_N[N]) for N in by_N):
        return {"decision": "NO_GO_BUDGET", "reason": "every N has a scenario whose power upper bound is below the target"}
    return {"decision": "POWER_NOT_ESTABLISHED",
            "reason": "no N meets both bounds in all scenarios, and not every N is clearly underpowered"}
