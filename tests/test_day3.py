"""Day-3: held-out datasets, frozen calibration artifact, stratified inference, KL rule, epsilon, freeze guards.

No test here runs a validation/test model forward pass, and none writes the real registration/freeze files."""
import copy
import csv
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from scp import ROOT, day3
from scp.bases import svd_family
from scp.common import load_config, sha256_file
from scp.data import load_pilot, read_csv

F64 = torch.float64
CFG = ROOT / "config" / "day3_v1.json"


@pytest.fixture(scope="module")
def cfg3():
    return load_config(CFG)


@pytest.fixture(scope="module")
def splits(cfg3):
    d = ROOT / cfg3["datasets"]["dir"]
    return {s: read_csv(d / f"{s}_pairs.csv") for s in ("validation", "test")}


# ------------------------------------------------------------------ datasets

def test_heldout_datasets_valid_and_hashed(cfg3, splits):
    idx = json.loads((ROOT / cfg3["datasets"]["dir"] / "datasets_index.json").read_text())
    for s, rows in splits.items():
        pool = day3.load_sealed(ROOT / cfg3["datasets"]["source"][s])
        info = day3.dataset_checks(rows, pool, cfg3["datasets"]["pairs_per_stratum"])
        assert info["n_pairs"] == 192 and info["design"]["n_cells"] == 12
        assert all(v == 9 for v in info["design"]["templates_per_frame"].values())
        assert idx["splits"][s]["tokenization_failures"] == 0
        assert idx["splits"][s]["sha256"] == sha256_file(ROOT / idx["splits"][s]["file"])
        # deterministic, output-independent regeneration
        again = day3.build_pairs(pool, s, cfg3["datasets"], cfg3["datasets"]["seeds"][s])
        assert [p["rec_text"] for p in again] == [p["rec_text"] for p in rows]
        assert [p["wrong_donor_text"] for p in again] == [p["wrong_donor_text"] for p in rows]


def test_heldout_no_overlap(splits):
    meta, cal_pairs, moment = load_pilot()
    texts = {}
    for s, rows in splits.items():
        texts[s] = {p[f"{r}_text"] for p in rows for r in ("rec", "don", "same_year", "wrong_donor")}
    cal = {p[f"{r}_text"] for p in cal_pairs for r in ("rec", "don", "same_year", "wrong_donor")} | {
        m["text"] for m in moment}
    assert not texts["validation"] & texts["test"] and not (texts["validation"] | texts["test"]) & cal
    fr = {s: {p["frame"] for p in rows} for s, rows in splits.items()}
    assert fr == {"validation": {"F2", "F3"}, "test": {"F4", "F5"}}
    nouns = {s: {p["noun"] for p in rows} for s, rows in splits.items()}
    assert not nouns["validation"] & nouns["test"]
    years = {s: {p[f"{r}_yy"] for p in rows for r in ("rec", "don", "same_year", "wrong_donor")}
             for s, rows in splits.items()}
    assert not years["validation"] & years["test"]
    assert years["validation"] <= set(meta["splits"]["years"]["validation"])


def test_design_check_rejects_missing_template(splits):
    rows = copy.deepcopy(splits["validation"])
    victim = rows[0]
    cell = day3.cell_key(victim)
    same = [p for p in rows if day3.cell_key(p) == cell and p["template"] == victim["template"]]
    for p in same:
        rows.remove(p)
    with pytest.raises(AssertionError):
        day3.design_check(rows)


def test_dataset_checks_reject_text_mismatch(cfg3, splits):
    rows = copy.deepcopy(splits["validation"])
    pool = day3.load_sealed(ROOT / cfg3["datasets"]["source"]["validation"])
    rows[3]["don_ids"] = rows[3]["don_ids"][:-1] + [0]
    with pytest.raises(AssertionError):
        day3.dataset_checks(rows, pool, 32)


# ------------------------------------------------------------------ artifact

def test_artifact_loads_and_matches_runtime(cfg3, spectrum):
    from scp.model import model_file_info
    msha = model_file_info(cfg3)["safetensors_sha256"]
    bases, moments, idx = day3.load_artifact(cfg3, model_sha=msha)
    assert len(bases) == 8 * 3 and set(moments) == set(bases)
    for k in cfg3["ranks"]:
        assert torch.allclose(bases[("svd", k)].P, spectrum.projectors(k)[0], atol=1e-12)
    assert bases[("wsvd", 192)].P.dtype == F64
    with pytest.raises(ValueError):
        day3.load_artifact(cfg3, model_sha="0" * 64)
    with pytest.raises(ValueError):
        day3.load_artifact(cfg3, expected_file_sha="0" * 64)
    bad = copy.deepcopy(cfg3)
    bad["bases"]["ridge_rel"] = 0.02
    with pytest.raises(ValueError):
        day3.load_artifact(bad)
    bad = copy.deepcopy(cfg3)
    bad["bases"]["random_seeds"] = [101, 202, 303]
    with pytest.raises(ValueError):
        day3.load_artifact(bad)


def _synthetic_artifact(tmp_path, cfg3, spectrum, sigma_value=0.1, drop=None):
    cfg = copy.deepcopy(cfg3)
    cfg["bases"]["families"] = ["svd"]
    cfg["ranks"] = [96]
    cfg["calibration_artifact"] = {"dir": str(tmp_path), "file": "a.pt", "index": "i.json"}
    b = svd_family(spectrum)(96)
    t = {"P|svd|96": b.P.contiguous()}
    for pool in day3.POOLS:
        t[f"mu|svd|96|{pool}"] = torch.zeros(3072, dtype=F64)
        t[f"sigma|svd|96|{pool}"] = torch.full((3072,), sigma_value, dtype=F64)
    if drop:
        t.pop(drop)
    torch.save({"tensors": t}, tmp_path / "a.pt")
    idx = {"file_sha256": sha256_file(tmp_path / "a.pt"),
           "tensors": {n: {"shape": list(x.shape), "dtype": str(x.dtype), "sha256": day3._tsha(x)} for n, x in t.items()},
           "fit_pool": {"moment_pool_manifest_sha256": sha256_file(ROOT / "manifests" / "moment_pool_calibration.csv")},
           "fit_settings": day3.fit_settings(cfg), "families": ["svd"], "random_seeds": cfg["bases"]["random_seeds"],
           "ranks": [96], "model_safetensors_sha256": "m", "calib_B_energy_rms": {}}
    (tmp_path / "i.json").write_text(json.dumps(idx))
    return cfg


def test_artifact_guards_synthetic(tmp_path, cfg3, spectrum):
    cfg = _synthetic_artifact(tmp_path, cfg3, spectrum)
    bases, moments, _ = day3.load_artifact(cfg)  # valid synthetic loads
    assert set(bases) == {("svd", 96)}
    for sub in ("neg", "drop", "bytes"):
        d = tmp_path / sub
        d.mkdir()
        if sub == "neg":
            c = _synthetic_artifact(d, cfg3, spectrum, sigma_value=-0.1)
        elif sub == "drop":
            c = _synthetic_artifact(d, cfg3, spectrum, drop="sigma|svd|96|all_token")
        else:
            c = _synthetic_artifact(d, cfg3, spectrum)
            with open(d / "a.pt", "ab") as f:
                f.write(b"x")
        with pytest.raises(ValueError):
            day3.load_artifact(c)


def test_heldout_context_cannot_refit():
    with pytest.raises(RuntimeError, match="forbidden"):
        day3.HeldoutContext.moment_activations(None, None)


# ------------------------------------------------------------------ stratified inference

def _synthetic_pairs():
    pairs = []
    for f in ("F2", "F3"):
        for direction in ("up", "down"):
            for b in ("10-19", "20-29", "30-40"):
                for t in range(9):
                    for rep in range(2 if t < 3 else 1):
                        pairs.append({"pair_id": f"p{len(pairs)}", "template": f"{f}|n{t}", "direction": direction,
                                      "diff_bin": b})
    return pairs


def test_strat_bootstrap_point_and_invariants():
    pairs = _synthetic_pairs()
    rng = np.random.default_rng(0)
    e = rng.normal(size=len(pairs))
    for mode in ("cluster", "iid"):
        (m,), d = day3.strat_mse_draws([e], pairs, 500, 1, mode)
        assert np.isclose(m, np.mean(e ** 2)) and np.isfinite(d).all()
        (m,), d = day3.strat_mse_draws([np.full(len(pairs), 0.3)], pairs, 200, 1, mode)
        assert np.allclose(d, 0.09)
    # frames are fixed: errors 0 in F2 and 1 in F3 -> every cluster draw equals the F3 weight
    e2 = np.array([0.0 if p["template"].startswith("F2") else 1.0 for p in pairs])
    (_,), d = day3.strat_mse_draws([e2], pairs, 300, 2, "cluster")
    assert np.allclose(d, np.mean(e2))


def _red_rows(pairs, spec):
    rows = []
    for (basis, method, k, scope, S), errs in spec.items():
        rows += [{"pair_id": p["pair_id"], "basis": basis, "method": method, "k": k, "scope": scope, "S": S,
                  "e_T": float(x)} for p, x in zip(pairs, errs)]
    return rows


def test_analyze_synthetic_decisions(cfg3):
    pairs = _synthetic_pairs()
    n = len(pairs)
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1.0, n)
    a = copy.deepcopy(cfg3["analysis"])
    a["bootstrap_B"] = 1000
    rows = _red_rows(pairs, {("svd", "gauss", 192, "last_token", "RB"): base * 1.5,
                             ("svd", "bias", 192, "last_token", "RB"): base,
                             ("pca", "bias", 192, "last_token", "RB"): base * 0.01})
    out = day3.analyze(pairs, rows, a, 0.0897, 1.79, 1.7, 0)
    assert out["P1"]["direction"] == "lhs_higher" and "equivalent_within_eps" not in out["P1"]
    assert out["S1"]["supported_lhs_lower"] and out["S2"]["preserved_upper_lt_eps"]
    assert out["secondary_upper_level"] == 0.975 and out["bootstrap_B"] == 1000
    assert set(out["per_frame_descriptive"]) == {"F2", "F3"}
    rows = _red_rows(pairs, {("svd", "gauss", 192, "last_token", "RB"): base,
                             ("svd", "bias", 192, "last_token", "RB"): base,
                             ("pca", "bias", 192, "last_token", "RB"): base})
    out = day3.analyze(pairs, rows, a, 0.0897, 1.79, 1.7, 0)
    assert out["P1"]["direction"] == "not_distinguished"
    assert not out["S1"]["supported_lhs_lower"] and not out["S2"]["preserved_upper_lt_eps"]


# ------------------------------------------------------------------ KL rule / epsilon

def test_kl_gap_rule_and_tiebreak(cfg3):
    assert day3.kl_gap(0.0, 0.0) == 0.0 and np.isclose(day3.kl_gap(1.0, 1.1), 2 * 0.1 / 2.1)
    k = copy.deepcopy(cfg3["analysis"]["kl_matching"])
    k["families"] = [{"id": "Q1", "lhs": ["svd", "bias"], "rhs": ["pca", "bias"]}]
    k["scopes"] = ["last_token"]

    def rows(vals):
        return [{"basis": b, "method": "bias", "scope": "last_token", "k": kk, "kl_orig_to_M": v}
                for (b, kk), v in vals.items()]
    vals = {("svd", 96): 0.30, ("svd", 192): 0.15, ("svd", 384): 0.05,
            ("pca", 96): 0.15, ("pca", 192): 0.05, ("pca", 384): 0.30}
    r = day3.kl_match(rows(vals), k, [96, 192, 384])["families"]["Q1|last_token"]
    # exact gap-0 ties: (96,384),(192,96),(384,192) -> min max-rank: (192,96)
    assert r["selected"]["lhs_k"] == 192 and r["selected"]["rhs_k"] == 96 and len(r["matched"]) == 3
    vals = {("svd", kk): 0.2 for kk in (96, 192, 384)} | {("pca", kk): 0.2 * 1.106 for kk in (96, 192, 384)}
    r = day3.kl_match(rows(vals), k, [96, 192, 384])["families"]["Q1|last_token"]
    assert r["selected"] is None and r["claim"] == "no_iso_kl_claim"  # g ~ 0.1007 > 0.10, strict


def test_epsilon_from_day1_independent(cfg3):
    T = []
    with open(ROOT / "runs/20261030T153707Z_pilot/per_pair_rows.csv") as f:
        for r in csv.DictReader(f):
            if r["model"] == "orig" and r["donor"] == "natural" and r["S"] == "RB" and r["k"] == "96":
                T.append(float(r["T"]))
    e = day3.epsilon_from_day1(cfg3)
    assert len(T) == 192 and abs(e["reference_rms"] - 1.7947415640957833) < 1e-14
    assert abs(e["epsilon"] - 0.08973707820478917) < 1e-15 and abs(e["epsilon"] - 0.05 * np.sqrt(np.mean(np.square(T)))) < 1e-15


# ------------------------------------------------------------------ registration / freeze guards (tmp paths only)

@pytest.fixture()
def tmp_reg(tmp_path, monkeypatch):
    monkeypatch.setattr(day3, "REG", tmp_path / "REG.json")
    monkeypatch.setattr(day3, "FROZEN_SEL", tmp_path / "SEL.json")
    monkeypatch.setattr(day3, "FROZEN_EPS", tmp_path / "EPS.json")
    monkeypatch.setattr(day3, "AMEND", tmp_path / "AMEND.json")
    return tmp_path


def _fake_val_run(tmp, cfg3, split="validation", gate_valid=True, source=None):
    d = tmp / f"run_{split}_{gate_valid}_{source is None}"
    d.mkdir()
    now = day3.current_bindings(cfg3)
    (d / "run_manifest.json").write_text(json.dumps({
        "status": "ok", "split": split, "source": {"tree_sha256": source or now["source_tree"]},
        "config_sha256": now["config_day3"], "day3_protocol_sha256": now["protocol_day3"],
        "calibration_artifact_sha256": now["calibration_artifact"],
        "day3_registration": {"sha256": sha256_file(day3.REG)}}))
    (d / "results.json").write_text(json.dumps({"split": split, "positive_control": {"valid": gate_valid}}))
    for f in ("control_rows.csv", "reduced_rows.csv"):
        (d / f).write_text("x\n1\n")
    with open(d / "unperturbed_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["basis", "method", "scope", "k", "kl_orig_to_M"])
        w.writeheader()
        for b in ("svd", "pca"):
            for m in ("bias", "gauss"):
                for s in ("last_token", "full_layer"):
                    for k in (96, 192, 384):
                        w.writerow({"basis": b, "method": m, "scope": s, "k": k,
                                    "kl_orig_to_M": 0.01 * k / 96 * (1 if b == "svd" else 1.02)})
    return d


def test_registration_freeze_verify_and_tamper(tmp_reg, cfg3, monkeypatch):
    day3.write_registration(cfg3)
    day3.verify_registration(cfg3)
    with pytest.raises(FileExistsError):
        day3.write_registration(cfg3)
    vr = _fake_val_run(tmp_reg, cfg3)
    s, e = day3.freeze(cfg3, vr)
    assert e["epsilon"] == 0.08973707820478917 and e["reference_csv_sha256"]
    day3.verify_frozen(cfg3)
    with pytest.raises(FileExistsError):
        day3.freeze(cfg3, vr)
    # tampered validation CSV after freeze
    orig = (vr / "unperturbed_rows.csv").read_bytes()
    (vr / "unperturbed_rows.csv").write_bytes(orig + b"\n")
    with pytest.raises(RuntimeError):
        day3.verify_frozen(cfg3)
    (vr / "unperturbed_rows.csv").write_bytes(orig)
    day3.verify_frozen(cfg3)
    # content tamper with a valid checksum -> recomputation mismatch
    sel = json.loads(day3.FROZEN_SEL.read_text())
    sel["kl_matching"]["families"]["Q1|last_token"]["selected"] = None
    sel["checksum"] = day3.canonical_sha({k: v for k, v in sel.items() if k != "checksum"})
    good = day3.FROZEN_SEL.read_text()
    day3.FROZEN_SEL.write_text(json.dumps(sel))
    with pytest.raises(RuntimeError):
        day3.verify_frozen(cfg3)
    # dummy / touch file
    day3.FROZEN_SEL.write_text("{}")
    with pytest.raises(RuntimeError):
        day3.verify_frozen(cfg3)
    day3.FROZEN_SEL.write_text(good)
    day3.verify_frozen(cfg3)
    # epsilon object tamper (reference hash) with a valid checksum
    eps = json.loads(day3.FROZEN_EPS.read_text())
    eps["reference_csv_sha256"] = "0" * 64
    eps["checksum"] = day3.canonical_sha({k: v for k, v in eps.items() if k != "checksum"})
    good_e = day3.FROZEN_EPS.read_text()
    day3.FROZEN_EPS.write_text(json.dumps(eps))
    with pytest.raises(RuntimeError):
        day3.verify_frozen(cfg3)
    day3.FROZEN_EPS.write_text(good_e)
    # source change after freeze is never bypassed, even with an amendment entry
    real = day3.source_hashes
    monkeypatch.setattr(day3, "source_hashes", lambda: {"tree_sha256": "f" * 64})
    arch = tmp_reg / "arch"
    arch.mkdir()
    day3.AMEND.write_text(json.dumps([{"tree_sha256": "f" * 64, "utc": "x", "archive": str(arch)}]))
    with pytest.raises(RuntimeError):
        day3.verify_frozen(cfg3)
    monkeypatch.setattr(day3, "source_hashes", real)
    # registration tamper
    reg = json.loads(day3.REG.read_text())
    reg["epsilon"]["epsilon"] = 0.2
    day3.REG.write_text(json.dumps(reg))
    with pytest.raises(RuntimeError):
        day3.verify_registration(cfg3)


def test_freeze_refuses_bad_validation_runs(tmp_reg, cfg3):
    day3.write_registration(cfg3)
    for kw in ({"split": "test"}, {"gate_valid": False}, {"source": "e" * 64}):
        vr = _fake_val_run(tmp_reg, cfg3, **kw)
        with pytest.raises(RuntimeError):
            day3.freeze(cfg3, vr)
        assert not day3.FROZEN_SEL.exists()


def test_verify_frozen_requires_files(tmp_reg, cfg3):
    with pytest.raises(RuntimeError):
        day3.verify_frozen(cfg3)
    day3.FROZEN_SEL.touch()
    day3.FROZEN_EPS.touch()
    with pytest.raises(Exception):
        day3.verify_frozen(cfg3)


# ------------------------------------------------------------------ CLI guards (never launch held-out runs)

def _cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "run_day3.py"), *args], capture_output=True, text=True, cwd=ROOT)


def test_cli_rejects_split_flag():
    p = _cli("smoke", "--split", "test")
    assert p.returncode != 0 and "--split is not accepted" in p.stderr


@pytest.mark.skipif((ROOT / "registration" / "DAY3_REGISTRATION.json").exists(),
                    reason="registration exists: the guard test would launch a real validation run")
def test_cli_validate_requires_registration():
    p = _cli("validate")
    assert p.returncode != 0 and "refusing validate" in p.stderr


@pytest.mark.skipif((ROOT / "registration" / "DAY3_FROZEN_SELECTION.json").exists(),
                    reason="freeze exists: the guard test could launch a real test run")
def test_cli_test_requires_freeze():
    p = _cli("test")
    assert p.returncode != 0 and "refusing test" in p.stderr
