"""Post-hoc, descriptive arithmetic (requested by GPT Pro after the registered result). Reads only the raw CSVs
of the locked predict-test and evaluate-test runs and the frozen s; writes nothing outside notes/.

Delta_i = (Y_i - predB_i)^2 - (Y_i - predA_i)^2 per test row (recipient x alpha). The registered MSE gives each of
the 72 frame x direction x gap x alpha cells weight 1/72, and each cell has 8 rows, so it equals the plain mean
over the 576 rows. Hence G = mean(Delta) / MSE_B, and a group g contributes sum_g(Delta) / (N * MSE_B) to G.
No CI, threshold, leave-out or causal explanation."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
P = ROOT / "followup/runs/20261030T192301Z_followup_predict_test"
E = ROOT / "followup/runs/20261030T192318Z_followup_evaluate_test"
s = json.loads((ROOT / "followup/registration/FOLLOWUP_FROZEN.json").read_text())["s"]
res = json.loads((E / "results.json").read_text())

feat = {(r["rid"], float(r["alpha"])): r for r in csv.DictReader(open(P / "test_features.csv"))}
pred = {(r["rid"], float(r["alpha"])): r for r in csv.DictReader(open(P / "test_predictions.csv"))}
lab = {(r["rid"], float(r["alpha"])): float(r["label_raw"]) / s for r in csv.DictReader(open(E / "test_labels.csv"))}
assert set(feat) == set(pred) == set(lab) and len(lab) == 576

rows = []
for k in sorted(lab):
    f, p, y = feat[k], pred[k], lab[k]
    eA, eB = (y - float(p["pred_A"])) ** 2, (y - float(p["pred_B"])) ** 2
    rows.append({"rid": k[0], "alpha": k[1], "template": f["template"], "frame": f["frame"],
                 "direction": f["direction"], "diff_bin": f["diff_bin"], "Y": y, "pred_A": float(p["pred_A"]),
                 "pred_B": float(p["pred_B"]), "sqerr_A": eA, "sqerr_B": eB, "delta": eB - eA})
N = len(rows)
cells = defaultdict(list)
for r in rows:
    cells[(r["frame"], r["direction"], r["diff_bin"], r["alpha"])].append(r)
assert len(cells) == 72 and all(len(v) == 8 for v in cells.values())
mse_A = np.mean([r["sqerr_A"] for r in rows])
mse_B = np.mean([r["sqerr_B"] for r in rows])
tot = sum(r["delta"] for r in rows)
G = tot / N / mse_B
assert abs(mse_A - res["primary"]["mse_A"]) < 1e-15 and abs(mse_B - res["primary"]["mse_B"]) < 1e-15
assert abs(G - res["primary"]["G"]) < 1e-12


def summarise(level, key, rs):
    d = np.array([r["delta"] for r in rs])
    mB, mA = np.mean([r["sqerr_B"] for r in rs]), np.mean([r["sqerr_A"] for r in rs])
    return {"level": level, "key": key, "n": len(rs), "mse_B": mB, "mse_A": mA, "mean_delta": d.mean(),
            "sum_delta": d.sum(), "group_G_point": 1 - mA / mB, "B_mse_weight": len(rs) * mB / (N * mse_B),
            "contribution_to_G": d.sum() / (N * mse_B), "net_share_of_total_gain": d.sum() / tot,
            "n_rows_delta_pos": int((d > 0).sum()), "n_rows_delta_neg": int((d < 0).sum())}


groups = {"template_x_alpha": defaultdict(list), "template": defaultdict(list), "alpha": defaultdict(list)}
for r in rows:
    groups["template_x_alpha"][f"{r['template']}|a={r['alpha']}"].append(r)
    groups["template"][r["template"]].append(r)
    groups["alpha"][f"a={r['alpha']}"].append(r)
out = [summarise(lv, k, v) for lv in groups for k, v in sorted(groups[lv].items())]
out.append(summarise("total", "all", rows))
for lv in groups:
    c = sum(o["contribution_to_G"] for o in out if o["level"] == lv)
    assert abs(c - G) < 1e-12, lv
assert all(o["n"] == 6 for o in out if o["level"] == "template_x_alpha")
assert all(o["n"] == 18 for o in out if o["level"] == "template")

with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
with open(sys.argv[3], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(json.dumps({"G": G, "mse_A": mse_A, "mse_B": mse_B, "sum_delta": tot, "N": N}))
