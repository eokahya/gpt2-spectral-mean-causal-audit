"""Development prompts: frames, nouns, recipient/retained/discarded-donor groups, manifests, leakage checks.

Outcome-blind: config + seed + tokenizer only (no model forward)."""
import csv
import itertools
import json

import numpy as np

from scp.common import sha256_file, utcnow
from scp.data import check_prompt

from . import ROOT
from .common import GROOT, ids_sha, manifest_dir, rel, text_sha


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
    D = donor_offsets(cfg)
    for g, grp in enumerate(groups):
        grp["group_id"] = f"dev-{g:03d}"
        for k, df in enumerate(asg["donor_frames"]):
            dn = base[(g + D[k]) % n]
            text, ids = _ids(tok, cfg, df, dn, grp["r"])
            if ids is None:
                raise TokenizationError(f"donor prompt fails tokenization: {text!r}")
            grp[f"don{k}_frame"], grp[f"don{k}_noun"] = df, dn
            grp[f"don{k}_text"], grp[f"don{k}_ids"] = text, ids
    return groups


def donor_offsets(cfg):
    """Slot k of group g uses donor noun base[(g + D[k]) mod n]. v1 (no offsets in the config): D = 6k."""
    K = cfg["design"]["K"]
    D = cfg["design"].get("donor_offsets", [6 * k for k in range(K)])
    if len(D) != K or len(set(d % cfg["design"]["n_groups"] for d in D)) != K:
        raise ValueError(f"donor offsets must be {K} distinct residues: {D}")
    return list(D)


def offset_profile(D, n):
    """Positive differences, modular ordered-difference multiplicities, and the group-pair noun-overlap profile."""
    pos = [D[j] - D[i] for i in range(len(D)) for j in range(i + 1, len(D))]
    mult = {}
    for a in D:
        for b in D:
            if a != b:
                mult[(a - b) % n] = mult.get((a - b) % n, 0) + 1
    return {"offsets": list(D), "positive_differences": sorted(pos),
            "positive_differences_distinct": len(set(pos)) == len(pos),
            "max_modular_multiplicity": max(mult.values()), "multiplicities": dict(sorted(mult.items()))}


def noun_overlaps(groups, K):
    """Group-level [G, G] and template-level [T, T] donor-noun overlap matrices.
    Template level: the number of shared donor nouns summed over the 2 x 2 group pairs of the two templates
    (diagonal: the overlap between the template's own two groups)."""
    nouns = [{g[f"don{k}_noun"] for k in range(K)} for g in groups]
    G = len(groups)
    M = np.array([[len(nouns[a] & nouns[b]) for b in range(G)] for a in range(G)])
    tmpl = sorted({g["template"] for g in groups})
    members = {t: [i for i, g in enumerate(groups) if g["template"] == t] for t in tmpl}
    TM = np.zeros((len(tmpl), len(tmpl)), dtype=int)
    for i, a in enumerate(tmpl):
        for j, b in enumerate(tmpl):
            if a == b:
                x, y = members[a]
                TM[i, j] = M[x, y]
            else:
                TM[i, j] = sum(M[x, y] for x in members[a] for y in members[b])
    return M, tmpl, TM


def acceptance_checks(cfg, groups, prompts, asg):
    """Exhaustive outcome-blind acceptance for configs with an `acceptance` block (v2). Raises on any failure."""
    acc = cfg["design"].get("acceptance")
    if acc is None:
        return None
    K, n = cfg["design"]["K"], cfg["design"]["n_groups"]
    prof = offset_profile(donor_offsets(cfg), n)
    assert prof["positive_differences_distinct"], "positive offset differences are not distinct"
    assert prof["max_modular_multiplicity"] <= acc["max_modular_multiplicity"], prof
    M, tmpl, TM = noun_overlaps(groups, K)
    off = M[~np.eye(len(groups), dtype=bool)]
    n_pairs = len(groups) * (len(groups) - 1) // 2
    assert int(off.max()) <= acc["max_pair_noun_overlap"], "a group pair shares too many donor nouns"
    assert int(M.diagonal().min()) == int(M.diagonal().max()) == K, "a group does not have K distinct donor nouns"
    total = int(off.sum()) // 2
    assert total == n * K * (K - 1) // 2, "total shared-noun incidences differ from 48 x C(8, 2)"
    uses = {}
    for g in groups:
        for k in range(K):
            uses[g[f"don{k}_noun"]] = uses.get(g[f"don{k}_noun"], 0) + 1
    assert set(uses.values()) == {K} and len(uses) == n
    # lexical separation of recipients and donors; grammatical/format compatibility (not semantic naturalness)
    rec_nouns = {g["noun"] for g in groups}
    assert not rec_nouns & set(uses) and len(rec_nouns) == len(groups) // cfg["design"]["groups_per_template"]
    assert not set(asg["recipient_frames"]) & set(asg["donor_frames"])
    for f in asg["recipient_frames"] + asg["donor_frames"]:
        tmpl_s = cfg["task"]["frames"][f]
        assert tmpl_s.count("{noun}") == 1 and tmpl_s.count("{yy}") == 1 and tmpl_s.startswith("The {noun} ")
        assert tmpl_s.endswith(" the year 17") and "the year 17{yy}" in tmpl_s
    for w in rec_nouns | set(uses):
        assert w.isalpha() and w.islower(), w
    for p in prompts:
        assert p["text"] == prompt(cfg, p["frame"], p["noun"], p["yy"])
    hist = sorted((ov, cnt) for ov, cnt in zip(*np.unique(off, return_counts=True)))
    return {"offset_profile": prof, "n_group_pairs": n_pairs,
            "pair_overlap_counts": {int(k): int(v) // 2 for k, v in hist},
            "mean_pair_overlap": float(off.mean()), "total_shared_noun_incidences": total,
            "max_template_pair_incidences": int(TM[~np.eye(len(tmpl), dtype=bool)].max()),
            "same_template_group_overlaps": sorted({int(TM[i, i]) for i in range(len(tmpl))}),
            "semantic_naturalness": "not checked (format, template agreement and tokenization only)"}


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
    """v3: no rebuild. The v2 manifests are copied byte-identically into the v3 manifest directory and re-verified
    (hash equality with the v2 registration, index hashes, structure, acceptance and tokenization), then a
    write-once v3 index records the provenance."""
    from .common import verify_v2_preservation
    import shutil
    d = manifest_dir(cfg)
    pres = cfg["v2_preservation"]
    src = ROOT / pres["v2_manifest_dir"]
    if (d / "manifest_copy_v3.json").exists():
        raise FileExistsError("v3 manifest copy is already recorded; never overwritten")
    d.mkdir(parents=True, exist_ok=True)
    for name in pres["manifest_copy"]:
        target = d / name
        if target.exists():
            if sha256_file(target) != sha256_file(src / name):
                raise FileExistsError(f"{target} exists and differs from v2; never overwritten")
        else:
            shutil.copyfile(src / name, target)
    vp = verify_v2_preservation(cfg)
    if not vp["ok"]:
        raise ValueError(f"v2 preservation failed: {vp['problems']}")
    groups, index = load_groups(cfg, check_index=True)
    asg = index["assignment"]
    prompts = long_prompts(cfg, groups)
    hist, files = historical_texts(cfg)
    for p in prompts:
        if check_prompt(tok, cfg, p["text"], p["yy"]) != p["ids"]:
            raise TokenizationError(p["text"])
    info = structural_checks(cfg, asg, groups, prompts, hist)
    accept = acceptance_checks(cfg, groups, prompts, asg)
    rec = {"created_utc": utcnow().isoformat(), "protocol_version": cfg["protocol_version"],
           "rule": "byte-identical copy of the registered v2 manifests; no resampling, no new nouns/rank/model",
           "source_dir": pres["v2_manifest_dir"],
           "files": {n: sha256_file(d / n) for n in pres["manifest_copy"]},
           "v2_registration_sha256": pres["files"]["discarded_response_floor_gate/registration/DEV_REGISTRATION_v2.json"],
           "checks": info, "acceptance_recheck": accept, "historical_files": files}
    with open(d / "manifest_copy_v3.json", "x") as f:
        json.dump(rec, f, indent=2)
    return rec


def load_groups(cfg, check_index=True):
    """Groups from the (copied, immutable) manifest; the v2 index hashes and the v3 copy record must both match."""
    d = manifest_dir(cfg)
    index = json.loads((d / "index.json").read_text())
    for n, sha in index["files"].items():
        if sha256_file(d / n) != sha:
            raise ValueError(f"{n}: hash differs from the manifest index")
    rec = d / "manifest_copy_v3.json"
    if rec.exists():
        for n, sha in json.loads(rec.read_text())["files"].items():
            if sha256_file(d / n) != sha:
                raise ValueError(f"{n}: hash differs from the v3 copy record")
    return read_groups(d / "groups.csv"), index
