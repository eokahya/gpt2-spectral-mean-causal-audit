"""CPU-only tests of the diagnostic logic (v2) on a tiny random GPT-2 (no GPT-2 weights, no MPS, no scientific data).

    .venv/bin/python -m pytest notes/gate_numerical_diagnostic/test_diagnose.py -q -p no:cacheprovider
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

import diagnose  # noqa: E402

LAYER = 1


def tiny(dtype):
    c = GPT2Config(vocab_size=61, n_positions=40, n_embd=24, n_layer=3, n_head=4, attn_implementation="eager",
                   bos_token_id=0, eos_token_id=0)
    torch.manual_seed(0)
    m = GPT2LMHeadModel(c).to(dtype).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def prompts(n, lengths, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, 61, (lengths[i % len(lengths)],), generator=g).tolist() for i in range(n)]


@pytest.fixture(scope="module")
def results():
    m32, m64 = tiny(torch.float32), tiny(torch.float64)
    pool = prompts(300, [12, 14], 2)  # the old-calibration length profile
    targets = [pool[0], pool[1], pool[2]]  # lengths 12, 14, 12
    J = list(range(20, 40))
    return diagnose.diagnose(m32, LAYER, 0, torch.device("cpu"), targets, pool, J, m64, spots=1,
                             pads=(0, 1, 3), rows=(0, 24, 47), seed=5)


def test_cases_cover_padded_and_unpadded_rows_and_natural_padding(results):
    s = diagnose.summarise(results)
    f = s["float32"]
    assert f["n_padded"] > 0 and f["n_unpadded"] > 0
    cases = [c for r in results for c in r["float32"]["mixed"]]
    assert {c["row"] for c in cases} == {0, 24, 47}
    assert {c["pad"] for c in cases} == {0, 1, 2, 3}  # pad 2 only from the natural variant of length-12 targets
    assert any(c["case"].startswith("natural") and c["pad"] == 2 for c in cases)
    assert not any(c["case"].startswith("natural") for c in results[1]["float32"]["mixed"])  # 14 is the longest


def test_hook_identity_exact_in_every_layout_both_precisions(results):
    for r in results:
        for key in ("float32", "float64_cpu"):
            if key not in r:
                continue
            for s in r[key]["H_hook_identity_same_layout"].values():
                assert s["max_abs_all"] == 0.0
            for c in r[key]["mixed"]:
                assert c["H_mix_all_rows_max_abs"] == 0.0
                assert c["D_decomposition_residual"] < 1e-9


def test_cpu64_same_and_cross_layout_controls_present(results):
    assert "float64_cpu" in results[0] and "float64_cpu" not in results[1]
    r64 = results[0]["float64_cpu"]
    assert r64["mixed"] and "group19" in r64["G_drift_vs_single"]
    for c in r64["mixed"]:  # float64 cross-layout effects are tiny
        assert c["D_total_F19_branch_xmix_vs_Fmix"]["max_abs_all"] < 1e-9
    assert set(results[0]["f32_vs_cpu64_single_ref"]) == {"single", "group19", "same48"}
    s = diagnose.summarise(results)
    assert "float64_cpu" in s and s["float64_cpu"]["H_mix_all_rows_max_abs"] == 0.0
    json.dumps(results)  # serialisable


def test_controlled_padding_is_exact():
    g = np.random.default_rng(0)
    pool = prompts(200, [12, 14], 3)
    t = pool[0]
    for label, P, r, ids, extra in diagnose.mixed_cases(t, pool, (0, 2, 4), (5,), g):
        b = diagnose.padded_batch(ids, 0, torch.device("cpu"), extra)
        assert b.ids.shape[1] - len(t) == P and int(b.last_idx[r]) == len(t) - 1 and ids[r] == t


def test_v1_script_preserved_for_first_run():
    run = HERE / "runs" / "20261031T103857Z_diag" / "summary.json"
    if not run.exists():
        pytest.skip("first diagnostic run not present")
    want = json.loads(run.read_text())["script_sha256"]
    assert hashlib.sha256((HERE / "diagnose_v1.py").read_bytes()).hexdigest() == want
