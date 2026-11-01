"""Greater-than prompts: tokenizer validation, seeded splits, pilot pairs, manifests.

Nothing here looks at model outputs; prompts are excluded only for tokenization reasons.
"""
import csv
import itertools
import json
from dataclasses import dataclass

import numpy as np
import torch

from . import ROOT

SPLITS = ("calibration", "validation", "test")
MANIFEST_DIR = ROOT / "manifests"


def year_token_ids(tok):
    """J: ids of the 100 two-digit tokens '00'..'99' (no leading space). Raises if any is not one token."""
    ids = []
    for y in range(100):
        enc = tok.encode(f"{y:02d}")
        if len(enc) != 1:
            raise ValueError(f"two-digit year {y:02d} is not a single GPT-2 token: {enc}")
        ids.append(enc[0])
    if len(set(ids)) != 100:
        raise ValueError("year tokens are not distinct")
    return ids


def make_prompt(cfg, frame, noun, yy):
    return cfg["task"]["frames"][frame].format(noun=noun, yy=f"{yy:02d}")


def check_prompt(tok, cfg, text, yy):
    """Return token ids if the start year is ' CC' + 'YY' single tokens and the prompt ends in ' CC', else None."""
    cc = f" {cfg['task']['century']}"
    ids = tok.encode(text)
    toks = [tok.decode([i]) for i in ids]
    slots = [p for p in range(len(toks)) if toks[p] == cc]
    if len(slots) != 2 or slots[-1] != len(toks) - 1:
        return None
    p = slots[0]
    if toks[p + 1] != f"{yy:02d}":
        return None
    return ids


def make_splits(cfg):
    """Seeded assignment of nouns (4 groups) and years (stratified triples) to splits."""
    rng = np.random.default_rng(cfg["seed"])
    nouns = list(cfg["task"]["nouns"])
    groups = cfg["splits"]["noun_groups"]
    if len(nouns) % len(groups):
        raise ValueError("noun count must divide evenly into groups")
    perm = rng.permutation(len(nouns))
    size = len(nouns) // len(groups)
    noun_split = {g: sorted(nouns[i] for i in perm[j * size:(j + 1) * size]) for j, g in enumerate(groups)}

    years = list(range(cfg["task"]["yy_min"], cfg["task"]["yy_max"] + 1))
    year_split = {s: [] for s in SPLITS}
    for start in range(0, len(years), 3):
        chunk = years[start:start + 3]
        order = rng.permutation(3)
        for y, si in zip(chunk, order):
            year_split[SPLITS[si]].append(y)
    year_split = {s: sorted(v) for s, v in year_split.items()}
    return {"frames": cfg["splits"]["frames"], "nouns": noun_split, "years": year_split}


def valid_prompts(tok, cfg, frames, nouns, years):
    """All tokenizer-valid prompts for the given factors, plus the excluded combinations."""
    ok, excluded = {}, []
    for frame, noun, yy in itertools.product(frames, nouns, years):
        text = make_prompt(cfg, frame, noun, yy)
        ids = check_prompt(tok, cfg, text, yy)
        if ids is None:
            excluded.append((frame, noun, yy, text))
        else:
            ok[(frame, noun, yy)] = (text, ids)
    return ok, excluded


def build_pilot(tok, cfg, splits):
    """Year-balanced calibration pairs with control donors, and the disjoint GT-moment pool."""
    rng = np.random.default_rng(cfg["seed"] + 1)
    t = cfg["task"]
    cal_frames = splits["frames"]["calibration"]
    cal_nouns = splits["nouns"]["calibration"]
    cal_years = splits["years"]["calibration"]
    pool, excl_cal = valid_prompts(tok, cfg, cal_frames, cal_nouns, cal_years)
    templates = [(f, n) for f in cal_frames for n in cal_nouns]

    pairs = []
    n_per = cfg["budget"]["pilot_pairs_per_stratum"]
    strata = [(direction, tuple(b)) for direction in ("up", "down") for b in t["diff_bins"]]
    for direction, (lo, hi) in strata:
        cands = [(r, d) for r in cal_years for d in cal_years
                 if lo <= (d - r if direction == "up" else r - d) <= hi]
        if len(cands) < n_per:
            raise ValueError(f"stratum {direction} {lo}-{hi} has only {len(cands)} candidate year pairs")
        chosen = [cands[i] for i in rng.choice(len(cands), size=n_per, replace=False)]
        tmpl_order = [templates[i] for i in rng.permutation(len(templates))]
        for j, (r, d) in enumerate(chosen):
            # Cycle through templates for balance; skip templates where either prompt failed tokenization.
            for off in range(len(templates)):
                f, n = tmpl_order[(j + off) % len(templates)]
                if (f, n, r) in pool and (f, n, d) in pool:
                    break
            else:
                raise ValueError(f"no valid template for years {r},{d}")
            others = [(f2, n2) for (f2, n2) in templates if (f2, n2) != (f, n)]
            sy = [o for o in others if (*o, r) in pool]
            sy_f, sy_n = sy[rng.integers(len(sy))]
            wd_cands = [(f2, n2, y) for (f2, n2) in others for y in cal_years
                        if y not in (r, d) and (f2, n2, y) in pool]
            wd_f, wd_n, wd_y = wd_cands[rng.integers(len(wd_cands))]
            pairs.append(dict(direction=direction, diff_bin=f"{lo}-{hi}", r=r, d=d,
                              rec=(f, n, r), don=(f, n, d), same_year=(sy_f, sy_n, r),
                              wrong_donor=(wd_f, wd_n, wd_y)))
    out = []
    for i, p in enumerate(pairs):
        pid = f"cal-{i:04d}"
        row = {"pair_id": pid, "direction": p["direction"], "diff_bin": p["diff_bin"], "r": p["r"], "d": p["d"],
               "template": f"{p['rec'][0]}|{p['rec'][1]}", "frame": p["rec"][0], "noun": p["rec"][1]}
        for role in ("rec", "don", "same_year", "wrong_donor"):
            key = p[role]
            text, ids = pool[key]
            row[f"{role}_template"] = f"{key[0]}|{key[1]}"
            row[f"{role}_yy"] = key[2]
            row[f"{role}_text"] = text
            row[f"{role}_ids"] = ids
        out.append(row)

    mom_pool, excl_mom = valid_prompts(tok, cfg, cal_frames, splits["nouns"]["moment_pool"], cal_years)
    moment = [{"prompt_id": f"mom-{i:04d}", "frame": f, "noun": n, "yy": y, "text": text, "ids": ids}
              for i, ((f, n, y), (text, ids)) in enumerate(sorted(mom_pool.items()))]
    return out, moment, excl_cal + excl_mom


def sealed_prompts(tok, cfg, splits, split):
    """Validation/test prompts (generated and hashed, not run in the pilot)."""
    pool, excl = valid_prompts(tok, cfg, splits["frames"][split], splits["nouns"][split], splits["years"][split])
    return [{"frame": f, "noun": n, "yy": y, "text": text, "ids": ids}
            for (f, n, y), (text, ids) in sorted(pool.items())], excl


def write_manifests(tok, cfg):
    MANIFEST_DIR.mkdir(exist_ok=True)
    splits = make_splits(cfg)
    j_ids = year_token_ids(tok)
    pairs, moment, excluded = build_pilot(tok, cfg, splits)
    sealed = {s: sealed_prompts(tok, cfg, splits, s) for s in ("validation", "test")}
    for s, (_, ex) in sealed.items():
        excluded += ex
    meta = {
        "protocol_version": cfg["protocol_version"],
        "seed": cfg["seed"],
        "tokenizer": cfg["model"]["name"],
        "tokenizer_revision": cfg["model"]["revision"],
        "J_token_ids": j_ids,
        "J_tokens": [f"{y:02d}" for y in range(100)],
        "century_token_id": tok.encode(f" {cfg['task']['century']}"),
        "splits": splits,
        "n_pilot_pairs": len(pairs),
        "n_moment_pool": len(moment),
        "n_validation_prompts": len(sealed["validation"][0]),
        "n_test_prompts": len(sealed["test"][0]),
        "excluded_by_tokenization": [list(e) for e in excluded],
        "rule": "excluded only for tokenization; never by model output",
    }
    with open(MANIFEST_DIR / "split_manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    _write_csv(MANIFEST_DIR / "pilot_pairs_calibration.csv", pairs)
    _write_csv(MANIFEST_DIR / "moment_pool_calibration.csv", moment)
    for s, (rows, _) in sealed.items():
        _write_csv(MANIFEST_DIR / f"sealed_prompts_{s}.csv", rows)
    return meta


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()})


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k.endswith("ids"):
                r[k] = json.loads(v)
            elif k in ("r", "d", "yy") or k.endswith("_yy"):
                r[k] = int(v)
    return rows


def load_pilot():
    meta = json.loads((MANIFEST_DIR / "split_manifest.json").read_text())
    pairs = read_csv(MANIFEST_DIR / "pilot_pairs_calibration.csv")
    moment = read_csv(MANIFEST_DIR / "moment_pool_calibration.csv")
    return meta, pairs, moment


@dataclass
class Batch:
    ids: torch.Tensor
    mask: torch.Tensor
    last_idx: torch.Tensor

    def to(self, device):
        return Batch(self.ids.to(device), self.mask.to(device), self.last_idx.to(device))

    def slice(self, sl):
        return Batch(self.ids[sl], self.mask[sl], self.last_idx[sl])


def make_batch(id_lists, pad_id, device):
    """Right-padded batch. Positions are 0..L-1 for real tokens, so padding never affects them (causal)."""
    L = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), L), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    last = torch.tensor([len(x) - 1 for x in id_lists], dtype=torch.long)
    return Batch(ids, mask, last).to(device)
