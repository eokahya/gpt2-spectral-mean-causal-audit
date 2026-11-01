"""Markdown tables from a Day-2 run:  .venv/bin/python -m scp.report_day2 runs/<id>"""
import json
import sys
from pathlib import Path


def f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))


def ci(t, nd=3):
    return "—" if not t else f"{t[0]:.{nd}g} [{t[1]:.{nd}g}, {t[2]:.{nd}g}]"


ORDER = ["svd", "pca", "coord", "wsvd", "random_s101", "random_s202", "random_s303", "random_s404", "random_s505"]


def bkey(b):
    return ORDER.index(b) if b in ORDER else 99


def tables(res):
    out = []
    pc = res["positive_control"]
    out += [f"Positive control (gate): valid={pc['valid']}, T_RB natural {ci(pc['T_RB_natural_mean_ci'])}.", ""]
    ms = res["matching_summary"]
    out += [f"Matching: max rel err f64 {ms['max_rel_err_f64']:.2e}, f32 {ms['max_rel_err_f32']:.2e}; "
            f"degenerate rows {ms['n_degenerate_total']}, out-of-support rows {ms['n_oos_total']} "
            f"(row counts across all cells, not N).", ""]

    out += ["### Basis diagnostics (k=96/192/384)", "", "| basis | k | diagnostics |", "|---|---|---|"]
    for key, m in sorted(res["bases"].items(), key=lambda kv: (bkey(kv[0].split("|")[0]), int(kv[0].split("|")[1]))):
        b, k = key.split("|")
        keep = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in m.items()
                if kk in ("explained_var_frac", "m2_frac", "lambda", "n_eig_cov_below_lambda", "cond_C", "cond_L",
                          "asym_rel", "idempotence_err")}
        out.append(f"| {b} | {k} | {keep} |")
    out.append("")

    ctl = {(a["basis"], a["k"], a["donor"], a["S"], a["regime"]): a for a in res["controls"]}
    out += ["### Original model, natural donor, NATIVE deltas: T by basis (pair-iid CI; cluster CI)", "",
            "| basis | k | T_R | T_R cluster CI | T_B | T_B cluster CI | RMS‖δ_R‖ | RMS‖δ_B‖ | RMS‖W_in δ_R‖ | RMS‖W_in δ_B‖ |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for (b, k, d, S, rg), a in sorted(ctl.items(), key=lambda kv: (bkey(kv[0][0]), kv[0][1])):
        if d == "natural" and rg == "native" and S == "R" and b != "any":
            bb = ctl[(b, k, "natural", "B", "native")]
            out.append(f"| {b} | {k} | {ci(a['T_mean_ci'])} | {ci(a['T_mean_cluster_ci'])} | {ci(bb['T_mean_ci'])} | "
                       f"{ci(bb['T_mean_cluster_ci'])} | {f(a['delta_l2_rms'])} | {f(bb['delta_l2_rms'])} | "
                       f"{f(a['delta_preact_rms'])} | {f(bb['delta_preact_rms'])} |")
    rb = ctl[("any", 0, "natural", "RB", "native")]
    out += ["", f"T_RB natural (basis-independent): {ci(rb['T_mean_ci'])}.", ""]

    out += ["### Original-model interaction I = Δ_RB − Δ_R − Δ_B (native, natural)", "",
            "| basis | k | T_R | T_B | T_RB | I_T mean [CI] | I_T cluster CI |", "|---|---|---|---|---|---|---|"]
    for i in sorted(res["interaction_original"], key=lambda x: (bkey(x["basis"]), x["k"])):
        out.append(f"| {i['basis']} | {i['k']} | {f(i['T_R_mean'])} | {f(i['T_B_mean'])} | {f(i['T_RB_mean'])} | "
                   f"{ci(i['I_T_mean_ci'])} | {ci(i.get('I_T_mean_cluster_ci'))} |")
    out.append("")

    for title, kind in (("Cross-basis contrast: T[basis natural] − T[svd natural native] (paired, common mask)",
                         "cross_basis"),
                        ("Donor-control contrast: T[basis natural native] − T[basis control] (paired, common mask)",
                         "donor_control")):
        out += [f"### {title}", "",
                "| basis | k | S | donor | regime | rhs | n | diff mean [pair CI] | [cluster CI] | lhs mean | rhs mean | "
                "n excl. oos | diff excl. oos |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = [c for c in res["contrasts"] if c["contrast"] == kind]
        for c in sorted(rows, key=lambda x: (x["S"], x.get("donor", ""), x["regime"], bkey(x["basis"]), x["k"])):
            a, e = c["all"], c["excl_oos"]
            out.append(f"| {c['basis']} | {c['k']} | {c['S']} | {c.get('donor', 'natural')} | {c['regime']} | {c['rhs']} | "
                       f"{a['n']} | {ci(a.get('diff_mean_ci'))} | {ci(a.get('diff_mean_cluster_ci'))} | "
                       f"{f(a.get('lhs_mean'))} | {f(a.get('rhs_mean'))} | {e['n']} | {ci(e.get('diff_mean_ci'))} |")
        out.append("")

    out += ["### Matched-regime scale quantiles and flags (original controls)", "",
            "| basis | k | donor | S | regime | target | scale q0/q05/q50/q95/q100 | n_degenerate | n_oos | v-alignment |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for (b, k, d, S, rg), a in sorted(ctl.items(), key=lambda kv: (bkey(kv[0][0]), kv[0][1], kv[0][2], kv[0][3])):
        if rg != "native":
            q = a["scale_quantiles"]
            qs = "/".join(f"{q[x]:.3g}" for x in ("q0", "q05", "q50", "q95", "q100")) if q else "—"
            out.append(f"| {b} | {k} | {d} | {S} | {rg} | {a['match_target']} | {qs} | {a['n_degenerate']} | "
                       f"{a['n_oos']} | {f(a['v_alignment_mean'])} |")
    out.append("")

    out += ["### Random-subspace seed spread (original controls; per-seed mean T, NOT pooled into N)", "",
            "| k | donor | S | regime | mean over seeds | SD | min | max | per seed |", "|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(res["controls_random_seed_spread"], key=lambda x: (x["donor"], x["S"], x["regime"], x["k"])):
        out.append(f"| {s['k']} | {s['donor']} | {s['S']} | {s['regime']} | {f(s['mean'])} | {f(s['sd'])} | "
                   f"{f(s['min'])} | {f(s['max'])} | {', '.join(f'{v:.3g}' for v in s['per_seed'].values())} |")
    out.append("")

    unp = {(u["basis"], u["k"], u["method"], u["scope"]): u for u in res["unintervened"]}
    red = {(a["basis"], a["k"], a["method"], a["scope"], a["S"]): a for a in res["reduced"]}
    dec = {(d["basis"], d["k"], d["method"], d["scope"]): d for d in res["decomposition"]}
    for scope in ("last_token", "full_layer"):
        out += [f"### Reduced models, scope={scope}: unperturbed quality and natural-intervention fidelity", "",
                "| basis | k | method | KL mean [CI] | PD_M−PD_orig | top1 | mean e_R [pair; cluster] | RMS e_R | "
                "mean e_RB [pair; cluster] | RMS e_RB [pair; cluster] | local err ratio R | local err ratio RB |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (b, k, m, sc), u in sorted(unp.items(), key=lambda kv: (bkey(kv[0][0]), kv[0][1], kv[0][2])):
            if sc != scope:
                continue
            aR, aRB, d = red[(b, k, m, sc, "R")], red[(b, k, m, sc, "RB")], dec[(b, k, m, sc)]
            out.append(f"| {b} | {k} | {m} | {ci(u['kl_mean_ci'])} | {ci(u['pd_diff_mean_ci'])} | {u['top1_agree']:.3f} | "
                       f"{ci(aR['e_T_mean_ci'])}; {ci(aR['e_T_mean_cluster_ci'])} | {ci(aR['e_T_rms_ci'])} | "
                       f"{ci(aRB['e_T_mean_ci'])}; {ci(aRB['e_T_mean_cluster_ci'])} | {ci(aRB['e_T_rms_ci'])}; "
                       f"{ci(aRB['e_T_rms_cluster_ci'])} | {f(aR['local_err_ratio'])} | {f(aRB['local_err_ratio'])} |")
        out.append("")

    out += ["### Signed decomposition e_RB = e_R − C, C = T_B^orig + I^orig (per reduced cell)", "",
            "| basis | k | method | scope | C mean | MSE_R | −2E[C·e_R] | E[C²] | MSE_RB | identity residual |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for (b, k, m, sc), d in sorted(dec.items(), key=lambda kv: (kv[0][3], bkey(kv[0][0]), kv[0][1], kv[0][2])):
        out.append(f"| {b} | {k} | {m} | {sc} | {f(d['C_mean'])} | {f(d['MSE_R'])} | {f(d['cross_minus2_E_C_eR'])} | "
                   f"{f(d['E_C2'])} | {f(d['MSE_RB'])} | {d['identity_max_residual']:.1e} |")
    out.append("")

    out += ["### Random-subspace seed spread (reduced; per-seed RMS e_T)", "",
            "| k | method | scope | S | mean over seeds | SD | min | max |", "|---|---|---|---|---|---|---|---|"]
    for s in sorted(res["reduced_random_seed_spread"], key=lambda x: (x["scope"], x["S"], x["method"], x["k"])):
        out.append(f"| {s['k']} | {s['method']} | {s['scope']} | {s['S']} | {f(s['mean'])} | {f(s['sd'])} | "
                   f"{f(s['min'])} | {f(s['max'])} |")
    return "\n".join(out)


if __name__ == "__main__":
    d = Path(sys.argv[1])
    md = tables(json.loads((d / "results.json").read_text()))
    (d / "summary_tables.md").write_text(md)
    print(f"wrote {d / 'summary_tables.md'} ({len(md.splitlines())} lines)")
