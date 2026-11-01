# spectral_causal_pilot

Exploratory 3-day pilot: *When Can Spectral Components Be Replaced by Their Average? A Causal Audit of
Transformer MLPs.* GPT-2 small, block-9 MLP `W_in` truncated by SVD; the discarded pre-activation is
compensated by nothing / its calibrated mean / a calibrated Gaussian expectation / a weights-only Gaussian,
and the reduced model's causal responses to greater-than year-swap interventions are compared with the
original model's. Read **PROTOCOL.md** and **PROTOCOL_AMENDMENT_v1.1.md** first; results and status are in **SUMMARY.md**.
The reduced model is a *functional rank-k audit* (dense `P_R` + original `Wc`), not a compression method.
Day-1 reviewer outcome (conditional GO, limits, Day 2/3 priorities): **notes/three_model_review_day1.md**.
Day 2 (geometric controls: `PROTOCOL_DAY2.md`, `run_day2.py`): results in **DAY2_SUMMARY.md**; reviewer outcome
(conditional GO for bounded Day-3 validation) in **notes/three_model_review_day2.md**.
Day 3 (held-out validation F2/F3 and test F4/F5: `PROTOCOL_DAY3.md`, `config/day3_v1.json`, `run_day3.py`): results
in **DAY3_SUMMARY.md** (final); final review **notes/three_model_review_day3.md**; post-hoc diagnostic
`notes/day3_posthoc_diagnostic.md`. Pre-validation review:
`notes/three_model_review_day3_prevalidation.md`; audits: `notes/codex_day3_*_independent_audit.json`.

## Layout

| path | content |
|---|---|
| `PROTOCOL.md` | registered protocol (pilot-v1) |
| `config/pilot_v1.json` | all parameters (seed, model revision, ranks, methods, budget, quadrature nodes) |
| `manifests/` | split manifest, 192 calibration pairs + control donors, GT-moment pool, sealed val/test prompts |
| `registration/REGISTRATION.json` | SHA-256 of protocol/config/manifests/source + UTC time, written before measurement |
| `scp/` | code: `data` (prompts/splits), `spectral` (SVD, projectors, moments, Gauss–Hermite), `model` (MLP wrapper: patch/reduce/capture), `metrics` (T, PD, KL, bootstrap), `experiment` (stages) |
| `run.py` | CLI |
| `tests/` | scientific unit tests (pytest) |
| `runs/<utc>_<stage>/` | one directory per run: `run_manifest.json`, `log.txt`, results, `snapshot/` of the exact code/config/protocol |
| `env/hardware_and_versions.txt`, `requirements*.txt` | environment |
| `logs/` | pytest and pilot stdout logs |

## Reproduce

```bash
cd /Users/emre/Documents/Codex/ML_Projects/spectral_causal_pilot
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m venv .venv
PIP_CACHE_DIR=$PWD/.cache/pip .venv/bin/pip install -r requirements.lock.txt
HF_HOME=$PWD/.cache/hf .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('openai-community/gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e')"
.venv/bin/python -m pytest -q
.venv/bin/python run.py splits      # regenerates identical manifests (seeded; hashes must match registration)
.venv/bin/python run.py quadcheck   # Gauss-Hermite node convergence on calibration moments
.venv/bin/python run.py smoke
.venv/bin/python run.py bench
.venv/bin/python run.py pilot
```

`run.py register` refuses to overwrite an existing registration; `pilot/smoke/bench` refuse to run if the
protocol, config or manifests changed since registration. `--split test` is refused until
`registration/FROZEN_EPSILON.json` exists. Only `--layer 10` (fixed backup) is accepted besides the primary 9.

## Conventions worth knowing

* HF `Conv1D` `c_fc.weight` is `[768, 3072]` and computes `x @ Wc`; `W_in = Wc.T`. Projectors act on the
  768-d input `x` (= `ln_2` output).
* `last_token` scope = reduced `W_in` only at the final prompt position (local diagnosis);
  `full_layer` = at every position (actual replacement). They use different calibrated moment pools
  (final-token vs all-token); every result row has a `moment_pool` column.
* `S = RB` patch uses the full difference, i.e. the recipient's MLP input becomes the donor's (positive control).
* MPS float32 for the model; SVD and moments on CPU float64. No silent device fallback.
