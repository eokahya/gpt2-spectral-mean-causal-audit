"""Development prompts: frames, nouns, recipient/retained/discarded-donor groups, manifests, leakage checks.

Outcome-blind: config + seed + tokenizer only (no model forward)."""
import csv
import itertools
import json

import numpy as np

from scp.common import sha256_file, utcnow
from scp.data import check_prompt

from . import ROOT
from .common import MANIFESTS, ids_sha, rel, text_sha


def prompt(cfg, frame, noun, yy):
    return cfg["task"]["frames"][frame].format(noun=noun, yy=f"{yy:02d}")


def cells(cfg):
    return [(d, f"{lo}-{hi}") for d in cfg["task"]["directions"] for lo, hi in cfg["task"]["gap_bins"]]


def in_cell(r, a, direction, gap_bin):
    lo, hi = map(int, gap_bin.split("-"))
    gap = a - r if direction == "up" else r - a
    return lo <= gap <= hi


def historical(cfg):
    nouns, frames = set(), set()
    for src in cfg["task"]["historical_noun_sources"]:
        t = json.loads((ROOT / src).read_text())["task"]
        nouns |= set(t["nouns"])
        frames |= set(t["frames"].values())
    return nouns, frames


def assign(cfg):
    """Seeded, outcome-blind split of frames and nouns into recipient / donor / unused (reserved)."""
    des = cfg["design"]
    rng = np.random.default_rng(cfg["seed"])
    frames, nouns = sorted(cfg["task"]["frames"]), list(cfg["task"]["nouns"])
    h_nouns, h_frames = historical(cfg)
    if len(set(nouns)) != len(nouns):
        raise ValueError("duplicate nouns in config")
    if set(nouns) & h_nouns:
        raise ValueError(f"config nouns overlap historical nouns: {sorted(set(nouns) & h_nouns)}")
    if set(cfg["task"]["frames"].values()) & h_frames:
        raise ValueError("a config frame equals a historical frame")
    n_rec_f, n_don_f = des["recipient_frames"], des["donor_frames"]
    n_rec_n = n_rec_f * des["templates_per_frame"]
    n_groups = n_rec_n * des["groups_per_template"]
    if des["K"] != n_don_f or n_groups != des["n_groups"]:
        raise ValueError("design counts inconsistent (K must equal donor_frames; n_groups = templates x groups)")
    n_don_n = n_groups  # each donor frame uses every donor noun once (Latin-type assignment)
    if n_rec_f + n_don_f > len(frames) or n_rec_n + n_don_n > len(nouns):
        raise ValueError("not enough frames/nouns")
    fp = [frames[i] for i in rng.permutation(len(frames))]
    npm = [nouns[i] for i in rng.permutation(len(nouns))]
    return {"recipient_frames": sorted(fp[:n_rec_f]), "donor_frames": fp[n_rec_f:n_rec_f + n_don_f],
            "unused_frames": sorted(fp[n_rec_f + n_don_f:]), "recipient_nouns": npm[:n_rec_n],
            "donor_nouns": npm[n_rec_n:n_rec_n + n_don_n], "unused_nouns": sorted(npm[n_rec_n + n_don_n:])}


class TokenizationError(ValueError):
    pass


def _ids(tok, cfg, frame, noun, yy):
    text = prompt(cfg, frame, noun, yy)
    ids = check_prompt(tok, cfg, text, yy)
    return text, ids


def build(tok, cfg, asg):
    """48 groups. Raises (before any model use) if a required prompt fails the tokenizer check."""
    des = cfg["design"]
    rng = np.random.default_rng(cfg["seed"] + 1)
    years = list(range(cfg["task"]["yy_min"], cfg["task"]["yy_max"] + 1))
    cl = cells(cfg)
    groups = []
    for fi, f in enumerate(asg["recipient_frames"]):
        pi = rng.permutation(len(cl))
        for ti in range(des["templates_per_frame"]):
            noun = asg["recipient_nouns"][fi * des["templates_per_frame"] + ti]
            valid = {y: t for y in years for t in [_ids(tok, cfg, f, noun, y)] if t[1] is not None}
            used = set()
            for ci in (pi[ti], pi[(ti + 3) % len(cl)]):
                direction, gbin = cl[ci]
                cands = [(r, a) for r, a in itertools.product(sorted(valid), sorted(valid))
                         if r not in used and a not in used and in_cell(r, a, direction, gbin)]
                if not cands:
                    raise ValueError(f"no valid (r, a) for {f}|{noun} {direction} {gbin}")
                r, a = cands[rng.integers(len(cands))]
                used |= {r, a}
                groups.append({"frame": f, "noun": noun, "template": f"{f}|{noun}", "direction": direction,
                               "gap_bin": gbin, "r": int(r), "a": int(a), "rec_text": valid[r][0],
                               "rec_ids": valid[r][1], "ret_text": valid[a][0], "ret_ids": valid[a][1]})
    base = asg["donor_nouns"]
    n = len(groups)
    for g, grp in enumerate(groups):
        grp["group_id"] = f"dev-{g:03d}"
        for k, df in enumerate(asg["donor_frames"]):
            dn = base[(g + 6 * k) % n]
            text, ids = _ids(tok, cfg, df, dn, grp["r"])
            if ids is None:
                raise TokenizationError(f"donor prompt fails tokenization: {text!r}")
            grp[f"don{k}_frame"], grp[f"don{k}_noun"] = df, dn
            grp[f"don{k}_text"], grp[f"don{k}_ids"] = text, ids
    return groups


def long_prompts(cfg, groups):
    out = []
    for g in groups:
        out.append({"prompt_id": f"{g['group_id']}-rec", "group_id": g["group_id"], "role": "recipient", "slot": "",
                    "frame": g["frame"], "noun": g["noun"], "yy": g["r"], "text": g["rec_text"], "ids": g["rec_ids"]})
        out.append({"prompt_id": f"{g['group_id']}-ret", "group_id": g["group_id"], "role": "retained_donor",
                    "slot": "", "frame": g["frame"], "noun": g["noun"], "yy": g["a"], "text": g["ret_text"],
                    "ids": g["ret_ids"]})
        for k in range(cfg["design"]["K"]):
            out.append({"prompt_id": f"{g['group_id']}-d{k}", "group_id": g["group_id"], "role": "discarded_donor",
                        "slot": k, "frame": g[f"don{k}_frame"], "noun": g[f"don{k}_noun"], "yy": g["r"],
                        "text": g[f"don{k}_text"], "ids": g[f"don{k}_ids"]})
    for p in out:
        p["text_sha256"], p["ids_sha256"] = text_sha(p["text"]), ids_sha(p["ids"])
    return out


def historical_texts(cfg):
    texts, files = set(), []
    for g in cfg["design"]["historical_manifests"]:
        for p in sorted(ROOT.glob(g)):
            files.append(rel(p))
            with open(p) as f:
                for row in csv.DictReader(f):
                    texts |= {v for k, v in row.items() if k == "text" or k.endswith("_text")}
    if not texts:
        raise ValueError("no historical prompts found")
    return texts, files


def structural_checks(cfg, asg, groups, prompts, hist):
    """Raises on any violation; returns counts."""
    des = cfg["design"]
    K = des["K"]
    assert len(groups) == des["n_groups"]
    assert len({g["group_id"] for g in groups}) == len(groups)
    templates = {g["template"] for g in groups}
    assert len(templates) == des["recipient_frames"] * des["templates_per_frame"]
    for f in asg["recipient_frames"]:
        gf = [g for g in groups if g["frame"] == f]
        assert len(gf) == des["templates_per_frame"] * des["groups_per_template"]
        cnt = {}
        for g in gf:
            cnt[(g["direction"], g["gap_bin"])] = cnt.get((g["direction"], g["gap_bin"]), 0) + 1
        assert set(cnt) == set(cells(cfg)) and set(cnt.values()) == {len(gf) // len(cells(cfg))}, (f, cnt)
    for t in templates:
        gt = [g for g in groups if g["template"] == t]
        assert len(gt) == des["groups_per_template"]
        yrs = [y for g in gt for y in (g["r"], g["a"])]
        assert len(set(yrs)) == len(yrs), t
    contexts = set()
    rec_nouns = {g["noun"] for g in groups}
    for g in groups:
        assert in_cell(g["r"], g["a"], g["direction"], g["gap_bin"]) and g["r"] != g["a"]
        assert g["frame"] in asg["recipient_frames"]
        nouns_g = set()
        for k in range(K):
            df, dn = g[f"don{k}_frame"], g[f"don{k}_noun"]
            assert df == asg["donor_frames"][k] and df not in asg["recipient_frames"]
            assert dn not in rec_nouns
            assert g[f"don{k}_text"] == prompt(cfg, df, dn, g["r"])  # same year as the recipient
            assert (df, dn) not in contexts, "donor context reused"
            contexts.add((df, dn))
            nouns_g.add(dn)
        assert len(nouns_g) == K
    texts = [p["text"] for p in prompts]
    ids = [tuple(p["ids"]) for p in prompts]
    assert len(set(texts)) == len(texts), "duplicate prompt text"
    assert len(set(ids)) == len(ids), "duplicate token sequence"
    leaked = sorted(set(texts) & hist)
    assert not leaked, f"historical prompt reused: {leaked[:3]}"
    for p in prompts:
        assert check_prompt_ok(cfg, p)
    return {"n_groups": len(groups), "n_templates": len(templates), "n_prompts": len(prompts),
            "n_donor_contexts": len(contexts), "n_historical_texts": len(hist)}


def check_prompt_ok(cfg, p):
    return p["text"] == prompt(cfg, p["frame"], p["noun"], p["yy"]) and len(p["ids"]) > 2


def write_csv(path, rows, mode="x"):
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()})


def read_groups(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list(r):
            if k.endswith("_ids"):
                r[k] = json.loads(r[k])
        r["r"], r["a"] = int(r["r"]), int(r["a"])
    return rows


def prepare(tok, cfg):
    if MANIFESTS.exists() and any(MANIFESTS.iterdir()):
        raise FileExistsError(f"{MANIFESTS} is not empty; manifests are never overwritten")
    asg = assign(cfg)
    groups = build(tok, cfg, asg)
    prompts = long_prompts(cfg, groups)
    hist, files = historical_texts(cfg)
    for p in prompts:  # re-check every stored id list against the tokenizer
        if check_prompt(tok, cfg, p["text"], p["yy"]) != p["ids"]:
            raise TokenizationError(p["text"])
    info = structural_checks(cfg, asg, groups, prompts, hist)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    write_csv(MANIFESTS / "groups.csv", groups)
    write_csv(MANIFESTS / "prompts.csv", prompts)
    index = {"created_utc": utcnow().isoformat(), "seed": cfg["seed"], "build_seed": cfg["seed"] + 1,
             "assignment": asg, "checks": info, "historical_files": files,
             "rule": "outcome-blind: config + seed + tokenizer only; no exclusions (a tokenization failure aborts)",
             "files": {n: sha256_file(MANIFESTS / n) for n in ("groups.csv", "prompts.csv")}}
    with open(MANIFESTS / "index.json", "x") as f:
        json.dump(index, f, indent=2)
    return index


def load_groups():
    index = json.loads((MANIFESTS / "index.json").read_text())
    for n, sha in index["files"].items():
        if sha256_file(MANIFESTS / n) != sha:
            raise ValueError(f"{n}: hash differs from the manifest index")
    return read_groups(MANIFESTS / "groups.csv"), index
