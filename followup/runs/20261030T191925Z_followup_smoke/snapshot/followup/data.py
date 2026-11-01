"""Fresh follow-up splits: frames, nouns and YY pools; recipient/donor rows; leakage and structural checks.

Outcome-blind: only the config, the seed and the tokenizer are used (no model forward pass)."""
import csv
import itertools
import json

import numpy as np

from scp.common import sha256_file, utcnow
from scp.data import check_prompt

from . import ROOT
from .common import MANIFEST_DIR, rel

SPLITS = ("fit", "validation", "test")


def prompt(cfg, frame, noun, yy):
    return cfg["task"]["frames"][frame].format(noun=noun, yy=f"{yy:02d}")


def cells(cfg):
    """(direction, 'lo-hi') design cells in fixed order."""
    return [(d, f"{lo}-{hi}") for d in cfg["task"]["directions"] for lo, hi in cfg["task"]["diff_bins"]]


def in_cell(r, d, direction, diff_bin):
    lo, hi = map(int, diff_bin.split("-"))
    gap = d - r if direction == "up" else r - d
    return lo <= gap <= hi


def historical_nouns(cfg):
    return set(json.loads((ROOT / cfg["task"]["historical_noun_source"]).read_text())["task"]["nouns"])


def assign(cfg):
    """Seeded, outcome-blind assignment of frames, (unique) nouns and YY triples to the three splits."""
    rng = np.random.default_rng(cfg["seed"])
    des = cfg["design"]
    frames = sorted(cfg["task"]["frames"])
    nouns = list(cfg["task"]["nouns"])
    if len(set(nouns)) != len(nouns):
        raise ValueError("duplicate nouns in config")
    if set(nouns) & historical_nouns(cfg):
        raise ValueError("follow-up nouns overlap the historical Day-1..3 noun list")
    need_f = sum(des["frames_per_split"][s] for s in SPLITS)
    need_n = sum(des["frames_per_split"][s] * des["templates_per_frame"][s] for s in SPLITS)
    if need_f > len(frames) or need_n > len(nouns):
        raise ValueError(f"need {need_f} frames / {need_n} nouns, have {len(frames)} / {len(nouns)}")
    fperm = [frames[i] for i in rng.permutation(len(frames))]
    nperm = [nouns[i] for i in rng.permutation(len(nouns))]
    out_f, out_t = {}, {}
    fi = ni = 0
    for s in SPLITS:
        nf, nt = des["frames_per_split"][s], des["templates_per_frame"][s]
        out_f[s] = sorted(fperm[fi:fi + nf])
        fi += nf
        out_t[s] = []
        for f in out_f[s]:
            out_t[s] += [(f, n) for n in sorted(nperm[ni:ni + nt])]
            ni += nt
    years = list(range(cfg["task"]["yy_min"], cfg["task"]["yy_max"] + 1))
    out_y = {s: [] for s in SPLITS}
    for start in range(0, len(years), 3):
        chunk = years[start:start + 3]
        for y, si in zip(chunk, rng.permutation(3)):
            out_y[SPLITS[si]].append(y)
    return {"frames": out_f, "templates": out_t, "years": {s: sorted(v) for s, v in out_y.items()},
            "unused_frames": sorted(fperm[fi:]), "unused_nouns": sorted(nperm[ni:])}


def build_split(tok, cfg, split, templates, years, rng):
    """One recipient per template x (direction, gap bin) cell; donor = same template, other year.

    Recipient years are distinct within a template; donor prompts may repeat inside the template cluster.
    Candidates are uniform over the (r, d) pairs whose two prompts pass the tokenizer check."""
    rows, n_excluded = [], 0
    for f, n in templates:
        valid = {}
        for y in years:
            text = prompt(cfg, f, n, y)
            ids = check_prompt(tok, cfg, text, y)
            if ids is None:
                n_excluded += 1
            else:
                valid[y] = (text, ids)
        used_r = set()
        for ci in rng.permutation(len(cells(cfg))):
            direction, gbin = cells(cfg)[ci]
            cands = [(r, d) for r, d in itertools.product(sorted(valid), sorted(valid))
                     if r not in used_r and in_cell(r, d, direction, gbin)]
            if not cands:
                raise ValueError(f"{split}: no valid (r, d) for {f}|{n} {direction} {gbin}")
            r, d = cands[rng.integers(len(cands))]
            used_r.add(r)
            rows.append({"split": split, "frame": f, "noun": n, "template": f"{f}|{n}", "direction": direction,
                         "diff_bin": gbin, "r": r, "d": d, "rec_text": valid[r][0], "rec_ids": valid[r][1],
                         "don_text": valid[d][0], "don_ids": valid[d][1]})
    rows.sort(key=lambda p: (p["template"], cells(cfg).index((p["direction"], p["diff_bin"]))))
    for i, p in enumerate(rows):
        p["rid"] = f"{split[:3]}-{i:04d}"
    return rows, n_excluded


def historical_texts(cfg):
    """Every exact prompt text in the Day-1..3 manifests (moment pool, calibration pairs incl. controls, sealed
    validation/test prompts, Day-3 held-out pairs)."""
    texts, files = set(), []
    for g in cfg["design"]["historical_manifests"]:
        for p in sorted(ROOT.glob(g)):
            files.append(rel(p))
            with open(p) as f:
                for row in csv.DictReader(f):
                    texts |= {v for k, v in row.items() if k == "text" or k.endswith("_text")}
    if not texts:
        raise ValueError("no historical prompts found; leakage check cannot run")
    return texts, files


def leakage(rows, hist):
    return sorted({t for p in rows for t in (p["rec_text"], p["don_text"]) if t in hist})


def structural_checks(cfg, split, rows, asg):
    """Raises on any violation of the follow-up design; returns counts."""
    des = cfg["design"]
    tmpl = asg["templates"][split]
    assert len(rows) == des["expected_recipients"][split], (split, len(rows))
    assert len({p["rid"] for p in rows}) == len(rows)
    by_t = {}
    for p in rows:
        assert p["split"] == split and p["template"] == f"{p['frame']}|{p['noun']}"
        assert (p["frame"], p["noun"]) in tmpl and p["frame"] in asg["frames"][split]
        assert p["r"] in asg["years"][split] and p["d"] in asg["years"][split] and p["r"] != p["d"]
        assert in_cell(p["r"], p["d"], p["direction"], p["diff_bin"])
        assert p["rec_text"] == prompt(cfg, p["frame"], p["noun"], p["r"])
        assert p["don_text"] == prompt(cfg, p["frame"], p["noun"], p["d"])
        by_t.setdefault(p["template"], []).append(p)
    assert len(by_t) == len(tmpl)
    for t, ps in by_t.items():
        assert sorted((p["direction"], p["diff_bin"]) for p in ps) == sorted(cells(cfg)), t
        assert len({p["r"] for p in ps}) == len(ps), f"{t}: recipient years repeat"
    if split == "test":
        assert len(by_t) >= des["min_test_templates"]
    return {"n_recipients": len(rows), "n_templates": len(by_t), "frames": sorted({p["frame"] for p in rows}),
            "per_frame": {f: sum(p["frame"] == f for p in rows) for f in asg["frames"][split]}}


def cross_split_checks(asg, rows_by):
    """Frames, nouns, YY pools and prompts are disjoint across the follow-up splits."""
    for a, b in itertools.combinations(SPLITS, 2):
        assert not set(asg["frames"][a]) & set(asg["frames"][b]), (a, b)
        assert not {n for _, n in asg["templates"][a]} & {n for _, n in asg["templates"][b]}, (a, b)
        assert not set(asg["years"][a]) & set(asg["years"][b]), (a, b)
        ta = {t for p in rows_by[a] for t in (p["rec_text"], p["don_text"])}
        tb = {t for p in rows_by[b] for t in (p["rec_text"], p["don_text"])}
        assert not ta & tb, (a, b)
    nouns = [n for s in SPLITS for _, n in asg["templates"][s]]
    assert len(nouns) == len(set(nouns)), "a noun is shared by two templates"


def write_csv(path, rows, mode="x"):
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()})


def read_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("rec_ids", "don_ids"):
            r[k] = json.loads(r[k])
        for k in ("r", "d"):
            r[k] = int(r[k])
    return rows


def prepare(tok, cfg):
    """Write followup/manifests/{split}_recipients.csv + index.json (write-once)."""
    if MANIFEST_DIR.exists() and any(MANIFEST_DIR.iterdir()):
        raise FileExistsError(f"{MANIFEST_DIR} is not empty; follow-up manifests are never overwritten")
    asg = assign(cfg)
    hist, hist_files = historical_texts(cfg)
    rows_by, excl, checks = {}, {}, {}
    for i, s in enumerate(SPLITS):
        rng = np.random.default_rng(cfg["seed"] + 10 + i)
        rows_by[s], excl[s] = build_split(tok, cfg, s, asg["templates"][s], asg["years"][s], rng)
        checks[s] = structural_checks(cfg, s, rows_by[s], asg)
        leaked = leakage(rows_by[s], hist)
        if leaked:
            raise ValueError(f"{s}: {len(leaked)} prompts equal historical prompts, e.g. {leaked[:3]}")
    cross_split_checks(asg, rows_by)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    index = {"created_utc": utcnow().isoformat(), "seed": cfg["seed"], "assignment": asg,
             "rule": "outcome-blind: config + seed + tokenizer only; exclusions only for tokenization",
             "leakage": {"historical_files": hist_files, "n_historical_texts": len(hist), "n_overlap": 0},
             "splits": {}}
    for s in SPLITS:
        path = MANIFEST_DIR / f"{s}_recipients.csv"
        write_csv(path, rows_by[s])
        index["splits"][s] = {**checks[s], "file": rel(path), "sha256": sha256_file(path),
                              "tokenization_exclusions": excl[s], "build_seed": cfg["seed"] + 10 + SPLITS.index(s)}
    with open(MANIFEST_DIR / "index.json", "x") as f:
        json.dump(index, f, indent=2)
    return index


def load_split(split):
    index = json.loads((MANIFEST_DIR / "index.json").read_text())
    info = index["splits"][split]
    path = ROOT / info["file"]
    if sha256_file(path) != info["sha256"]:
        raise ValueError(f"{path}: hash differs from the manifest index")
    return read_rows(path), index
