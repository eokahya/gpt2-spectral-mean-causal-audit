"""Tokenization, split disjointness / leakage, pair design, score definition."""
import collections

import numpy as np
import pytest
import torch

from scp.data import build_pilot, check_prompt, make_splits, year_token_ids
from scp.metrics import t_score, v_rd


def test_year_tokens_single_and_distinct(cpu_model, pilot):
    _, tok = cpu_model
    meta, _, _ = pilot
    ids = year_token_ids(tok)
    assert ids == meta["J_token_ids"] and len(set(ids)) == 100
    assert [tok.decode([i]) for i in ids] == [f"{y:02d}" for y in range(100)]


def test_all_prompts_tokenize_as_specified(cpu_model, cfg, pilot):
    _, tok = cpu_model
    meta, pairs, moment = pilot
    cc = tok.encode(" 17")[0]
    for p in pairs:
        for role in ("rec", "don", "same_year", "wrong_donor"):
            ids = p[f"{role}_ids"]
            assert ids == tok.encode(p[f"{role}_text"]) == check_prompt(tok, cfg, p[f"{role}_text"], p[f"{role}_yy"])
            assert ids[-1] == cc
            assert ids[ids.index(cc) + 1] == meta["J_token_ids"][p[f"{role}_yy"]]
    for m in moment:
        assert m["ids"] == check_prompt(tok, cfg, m["text"], m["yy"])


def test_split_disjointness_and_leakage(pilot):
    meta, pairs, moment = pilot
    sp = meta["splits"]
    fr = sp["frames"]
    assert not set(fr["calibration"]) & set(fr["validation"]) and not set(fr["calibration"]) & set(fr["test"])
    assert not set(fr["validation"]) & set(fr["test"])
    groups = list(sp["nouns"].values())
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert not set(groups[i]) & set(groups[j])
    ys = sp["years"]
    assert not set(ys["calibration"]) & set(ys["validation"]) and not set(ys["calibration"]) & set(ys["test"])
    assert not set(ys["validation"]) & set(ys["test"])
    cal_n, cal_f, cal_y = set(sp["nouns"]["calibration"]), set(fr["calibration"]), set(ys["calibration"])
    for p in pairs:
        for role in ("rec", "don", "same_year", "wrong_donor"):
            f, n = p[f"{role}_template"].split("|")
            assert f in cal_f and n in cal_n and p[f"{role}_yy"] in cal_y
    # GT-moment pool shares no template or prompt with any recipient/donor
    pair_templates = {p[f"{r}_template"] for p in pairs for r in ("rec", "don", "same_year", "wrong_donor")}
    pair_texts = {p[f"{r}_text"] for p in pairs for r in ("rec", "don", "same_year", "wrong_donor")}
    assert not {f"{m['frame']}|{m['noun']}" for m in moment} & pair_templates
    assert not {m["text"] for m in moment} & pair_texts
    assert {m["noun"] for m in moment} <= set(sp["nouns"]["moment_pool"])
    assert len({p["pair_id"] for p in pairs}) == len(pairs)


def test_pair_design(cfg, pilot):
    _, pairs, _ = pilot
    assert len(pairs) == cfg["budget"]["pilot_pairs_total"]
    strata = collections.Counter((p["direction"], p["diff_bin"]) for p in pairs)
    assert set(strata.values()) == {cfg["budget"]["pilot_pairs_per_stratum"]} and len(strata) == 6
    for p in pairs:
        diff = p["d"] - p["r"]
        assert 10 <= abs(diff) <= 40 and (diff > 0) == (p["direction"] == "up")
        assert p["rec_template"] == p["don_template"] == p["template"]
        assert p["rec_yy"] == p["r"] and p["don_yy"] == p["d"]
        assert p["same_year_yy"] == p["r"] and p["same_year_template"] != p["template"]
        assert p["wrong_donor_template"] != p["template"] and p["wrong_donor_yy"] not in (p["r"], p["d"])


def test_splits_are_seed_deterministic(cpu_model, cfg, pilot):
    _, tok = cpu_model
    meta, pairs, moment = pilot
    sp = make_splits(cfg)
    assert sp["nouns"] == meta["splits"]["nouns"] and sp["years"] == meta["splits"]["years"]
    pairs2, moment2, _ = build_pilot(tok, cfg, sp)
    assert [p["rec_text"] for p in pairs2] == [p["rec_text"] for p in pairs]
    assert [p["wrong_donor_text"] for p in pairs2] == [p["wrong_donor_text"] for p in pairs]
    assert len(moment2) == len(moment)


def test_score_definition(pilot):
    _, pairs, _ = pilot
    r = np.array([p["r"] for p in pairs])
    d = np.array([p["d"] for p in pairs])
    v = v_rd(r, d)
    assert (v.pow(2).sum(-1) > 0).all()
    assert torch.allclose(v.sum(-1), torch.zeros(len(pairs), dtype=torch.float64), atol=1e-12)
    # a logit shift of exactly c * v (+ any constant) gives T = c
    c = torch.linspace(-2, 2, len(pairs), dtype=torch.float64)
    dl = c[:, None] * v + 3.7
    assert torch.allclose(t_score(dl, v), c)
    # recipient -> donor indicator change, uncentred, also gives T = 1
    j = torch.arange(100, dtype=torch.float64)
    ind = (j > torch.tensor(d)[:, None]).double() - (j > torch.tensor(r)[:, None]).double()
    assert torch.allclose(t_score(ind, v), torch.ones(len(pairs), dtype=torch.float64))
    with pytest.raises(ValueError):
        v_rd([30], [30])
