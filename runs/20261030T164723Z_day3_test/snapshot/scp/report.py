"""Markdown tables from a pilot run's results.json:  .venv/bin/python -m scp.report runs/<id>"""
import json
import sys
from pathlib import Path


def f(x, nd=3):
    return f"{x:.{nd}g}" if isinstance(x, float) else str(x)


def ci(t, nd=3):
    return f"{t[0]:.{nd}g} [{t[1]:.{nd}g}, {t[2]:.{nd}g}]"


def tables(res):
    out = []
    pc = res["positive_control"]
    out += ["### Positive control (original model, calibration split, n=192)", "",
            f"- mean PD: {ci(pc['pd_mean_ci'])}; fraction PD>0: {pc['pd_frac_positive']:.3f}",
            f"- T_RB natural: {ci(pc['T_RB_natural_mean_ci'])} (template-cluster CI, 18 clusters: "
            f"{ci(pc['T_RB_natural_cluster_ci'])})",
            f"- T_RB same_year: {ci(pc['T_RB_same_year_mean_ci'])}; wrong_donor: {ci(pc['T_RB_wrong_donor_mean_ci'])}",
            f"- valid: {pc['valid']}", ""]
    out += ["### Checks", "", "```", json.dumps(res["checks"], indent=1), "```", "",
            "Retained Frobenius energy of W_in: " + ", ".join(f"k={k}: {v:.3f}" for k, v in res["retained_energy"].items()), ""]

    out += ["### Unintervened quality (recipients, final token)", "",
            "| k | method | scope | KL(orig‖M) mean [95% CI] | KL median | PD_M − PD_orig [95% CI] | top-1 agree |",
            "|---|---|---|---|---|---|---|"]
    for u in res["unintervened"]:
        out.append(f"| {u['k']} | {u['method']} | {u['scope']} | {ci(u['KL_mean_ci'])} | {f(u['KL_median'])} | "
                   f"{ci(u['PD_diff_mean_ci'])} | {u['top1_agree']:.3f} |")
    out.append("")

    out += ["### Original-model interaction I = Δ_RB − Δ_R − Δ_B (diagnostic; ≡0 in any reduced model)", "",
            "| k | donor | T_R | T_B | T_RB | I_T mean [CI] | RMS I_T | RMS‖I_local‖ / RMS‖Δ_RB local‖ |",
            "|---|---|---|---|---|---|---|---|"]
    for i in res["interaction_original"]:
        out.append(f"| {i['k']} | {i['donor']} | {f(i['T_R_mean'])} | {f(i['T_B_mean'])} | {f(i['T_RB_mean'])} | "
                   f"{ci(i['I_T_mean_ci'])} | {f(i['I_T_rms'])} | {f(i['I_local_norm_rms'] / i['RB_local_norm_rms'])} |")
    out.append("")

    agg = {(a["model"], a["scope"], a["k"], a["donor"], a["S"]): a for a in res["scalar_aggregates"]}
    vec = {(a["method"], a["scope"], a["k"], a["donor"], a["S"]): a for a in res["vector_aggregates"]}
    ks = sorted({a["k"] for a in res["scalar_aggregates"]})
    for donor in ("natural", "same_year", "wrong_donor"):
        for S in ("RB", "R"):
            out += [f"### Response error vs original — donor={donor}, S={S}", "",
                    "| k | method | scope | T_M mean | T_orig mean | mean e_T [CI] | RMS e_T [pair CI] | RMS e_T [cluster CI] | "
                    "local RMS err / RMS orig resp | ‖mean local err‖ | logit RMS err / RMS orig resp |",
                    "|---|---|---|---|---|---|---|---|---|---|---|"]
            for k in ks:
                for method in ("zero", "bias", "gauss", "gauss_wo"):
                    for scope in ("last_token", "full_layer"):
                        a = agg.get((method, scope, k, donor, S))
                        v = vec.get((method, scope, k, donor, S))
                        if a is None:
                            continue
                        out.append(
                            f"| {k} | {method} | {scope} | {f(a['T_mean_ci'][0])} | {f(a['T_orig_mean'])} | "
                            f"{ci(a['e_T_mean_ci'])} | {ci(a['e_T_rms_ci'])} | {ci(a['e_T_rms_cluster_ci'])} | "
                            f"{f(v['local_rms_error_norm'] / v['local_rms_orig_response_norm'])} | "
                            f"{f(v['local_mean_error_vec_norm'])} | "
                            f"{f(v['logit_rms_error_norm'] / v['logit_rms_orig_response_norm'])} |")
            out.append("")

    out += ["### Control effect norm vs pre-activation energy (original model, k-specific for R/B)", "",
            "| donor | S | k | T mean | RMS‖Δ local‖ | RMS‖Δ logits‖ | RMS‖δ‖ | RMS‖W_in δ‖ | RMS‖W_in P_R δ‖ |",
            "|---|---|---|---|---|---|---|---|---|"]
    for donor in ("natural", "same_year", "wrong_donor"):
        for S in ("R", "B", "RB"):
            for k in ks:
                a = agg[("orig", "none", k, donor, S)]
                out.append(f"| {donor} | {S} | {k} | {ci(a['T_mean_ci'])} | {f(a['local_resp_norm_rms'])} | "
                           f"{f(a['logit_resp_norm_rms'])} | {f(a['delta_norm_rms'])} | "
                           f"{f(a['preact_energy_orig_rms'])} | {f(a['preact_energy_reduced_rms'])} |")
    out.append("")

    out += ["### Candidate primary comparison — EXPLORATORY (calibration split; ε not frozen)", "",
            "gauss vs bias, natural donor, S=RB, statistic RMS_i(e_T).", "",
            "| scope | k | RMS e_T gauss | RMS e_T bias | RMS gauss − bias [pair CI] | [cluster CI] | mean e_T gauss | mean e_T bias |",
            "|---|---|---|---|---|---|---|---|"]
    for p in res["primary_candidate_exploratory"]:
        out.append(f"| {p['scope']}{' (candidate)' if p['primary_scope_candidate'] else ''} | {p['k']} | "
                   f"{ci(p['rms_e_T_gauss_ci'])} | {ci(p['rms_e_T_bias_ci'])} | {ci(p['rms_gauss_minus_bias_ci'])} | "
                   f"{ci(p['rms_gauss_minus_bias_cluster_ci'])} | {f(p['mean_e_T_gauss'])} | {f(p['mean_e_T_bias'])} |")
    e = res["epsilon_reference_scales"]
    out += ["", f"ε reference scales (not frozen): T_orig natural RB mean {f(e['T_orig_natural_RB_mean'])}, "
                f"SD {f(e['T_orig_natural_RB_sd'])}, RMS {f(e['T_orig_natural_RB_rms'])}.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    d = Path(sys.argv[1])
    md = tables(json.loads((d / "results.json").read_text()))
    (d / "summary_tables.md").write_text(md)
    print(md)
