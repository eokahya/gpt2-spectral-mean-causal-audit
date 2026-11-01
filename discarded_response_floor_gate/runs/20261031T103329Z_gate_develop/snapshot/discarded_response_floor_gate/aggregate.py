"""Separate aggregation implementation (numpy only; does not import estimators.py or pipeline.py).

Recomputes R, U (two ways), s^2, Lhat^2, template T and frame moments from the frozen raw files of a development
run: raw_J_logits.pt (19 rows x 100 J logits per group, row order in `row_kinds`) and the groups.csv of the manifest
directory recorded in the run's manifest (v2: discarded_response_floor_gate/manifests_v2/), whose hash must equal the
hash the run was bound to.
This is a second implementation by the same author, not an independent researcher audit.

    python -m discarded_response_floor_gate.aggregate discarded_response_floor_gate/runs/<dev_run>
"""
import csv
import json
import sys
from pathlib import Path

import hashlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def aggregate(run_dir):
    run_dir = Path(run_dir)
    raw = torch.load(run_dir / "raw_J_logits.pt", map_location="cpu", weights_only=False)
    L = raw["raw_J_logits"].numpy().astype(np.float64)  # [G, 19, m]
    kinds = raw["row_kinds"]
    idx = {k: i for i, k in enumerate(kinds)}
    m = L.shape[-1]
    F = L - L.mean(axis=-1, keepdims=True)
    minus = F[:, [idx[f"minus{k}"] for k in range(8)]]
    plus = F[:, [idx[f"plus{k}"] for k in range(8)]]
    R = plus - minus  # [G, 8, m]
    G = R.shape[0]
    U_var = np.array([((R[g] - R[g].mean(0)) ** 2).sum() / m / 8 for g in range(G)])
    U_pair = np.zeros(G)
    for g in range(G):
        acc = 0.0
        for i in range(8):
            for j in range(i + 1, 8):
                acc += ((R[g, i] - R[g, j]) ** 2).sum() / m
        U_pair[g] = acc / 64
    nat = F[:, idx["nat_a"]] - F[:, idx["nat_r"]]
    s2 = float(((nat ** 2).sum(-1) / m).mean())
    man = json.loads((run_dir / "run_manifest.json").read_text())
    groups_rel = f"{man['manifest_dir']}/groups.csv"
    groups_csv = ROOT / groups_rel
    if hashlib.sha256(groups_csv.read_bytes()).hexdigest() != man["bound"]["manifests"][groups_rel]:
        raise ValueError(f"{groups_rel} differs from the manifest the run was bound to")
    with open(groups_csv) as f:
        meta = {r["group_id"]: r for r in csv.DictReader(f)}
    gids = raw["group_ids"]
    T = {}
    for g, gid in enumerate(gids):
        r = meta[gid]
        T.setdefault(r["frame"], {}).setdefault(r["template"], []).append(U_var[g])
    T = {f: {t: float(np.mean(v)) / s2 for t, v in d.items()} for f, d in T.items()}
    fm = {f: float(np.mean(list(d.values()))) for f, d in T.items()}
    fv = {f: float(np.var(list(d.values()), ddof=1)) for f, d in T.items()}
    with open(run_dir / "groups_U.csv") as f:
        saved = {r["group_id"]: float(r["U"]) for r in csv.DictReader(f)}
    return {"n_groups": G, "m": m, "s2": s2, "Lhat2": float(U_var.mean() / s2), "Lhat": float((U_var.mean() / s2) ** 0.5),
            "U_two_way_max_rel": float(np.max(np.abs(U_var - U_pair) / np.maximum(U_var, 1e-300))),
            "U_max_abs_diff_vs_groups_U_csv": float(max(abs(saved[gid] - U_var[g]) for g, gid in enumerate(gids))),
            "frame_mean_T": fm, "frame_var_T": fv, "mbar": float(np.mean(list(fm.values()))),
            "groups_csv": groups_rel,
            "label": "separate aggregation implementation (same author)"}


if __name__ == "__main__":
    print(json.dumps(aggregate(sys.argv[1]), indent=2))
