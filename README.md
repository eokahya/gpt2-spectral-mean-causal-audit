# Spectral component mean causal audit (GPT-2 small, block-9 MLP)

[Türkçe sürüm / Turkish version](README_TR.md)

A pre-registered causal audit of what happens to a transformer's behaviour when the discarded part of a low-rank
weight truncation is replaced by its calibrated average (or by a calibrated Gaussian expectation).

Working title in the project's protocol: *When Can Spectral Components Be Replaced by Their Average? A Causal Audit
of Transformer MLPs.*

## Bottom line

**The main hypothesis was not supported.** In the locked, held-out test of the main experiment (GPT-2 small, block 9,
MLP input matrix truncated by SVD to rank 192, last token only), replacing the discarded part with a calibrated
**Gaussian expectation** made the model's response to a year-swap intervention *worse*, not better, than replacing
it with a plain calibrated **mean**:

* registered primary statistic P1 = RMS error (Gaussian) − RMS error (mean) = **+0.0268, 95% CI [0.0235, 0.0302]**,
  about 2.1% of the mean-replacement error;
* both SVD variants had a large total error (RMS about 1.27–1.30, against an original response RMS of 1.758).

The project's final joint decision was **NO-GO** for the Gaussian/SVD-specific thesis.

A secondary, post-calibration choice did better. With a **PCA** basis (fitted on calibration activations) and plain
mean replacement at the same rank, the scalar response error was small (RMS 0.0684). Its one-sided upper bound
(0.0780) fell under the pre-registered tolerance ε = 0.0897. This is a narrow, scalar result. About 14–15% of the
underlying response *vectors* (MLP output and logits) were still not reproduced, and PCA was chosen only after the
calibration data had been seen.

Two later follow-up studies did not change this picture. One ended **INCONCLUSIVE**; the other was a development-only
measurement whose branch was closed.

This does **not** prove that mean or Gaussian replacement of spectral components can never work. It is a bounded
result: one model, one layer, one rank, one synthetic task, two prompt frames per held-out split, and one kind of
intervention.

## The problem in everyday language

Large weight matrices are often compressed by keeping only their strongest directions (a low-rank, "spectral"
truncation, typically via SVD). Whatever falls outside the kept directions is thrown away. A common repair is to
replace the thrown-away part with its **average** value, measured on some calibration text, so the layer is not
systematically shifted.

**Toy example.** Take one hidden neuron of an MLP. Its input is a sum `a = z + q`:

* `z` comes through the directions we keep;
* `q` comes through the directions we discard.

The neuron outputs `gelu(a)`. Suppose that, over calibration prompts, `q` for this neuron averages 0.3 with a spread
(standard deviation) of 1.0. The candidate replacements are:

| name | what the reduced neuron computes | idea |
|---|---|---|
| `zero` | `gelu(z)` | plain truncation; forget `q` |
| `bias` ("mean") | `gelu(z + 0.3)` | put back the average of `q` |
| `gauss` | average of `gelu(z + 0.3 + 1.0·ε)` over standard normal `ε` | put back the average *output* if `q` were Gaussian noise |
| `gauss_wo` | like `gauss`, but mean and spread estimated from the weights only | no calibration data |

Because `gelu` is curved, the Gaussian version is not the same as the mean version. The hope was that it matches the
real neuron better on average.

**What a causal audit checks.** Matching the model's outputs on ordinary inputs (for example, low KL divergence)
does not show that the reduced model *reacts to changes* the way the original does. The audit uses a task where the
original model's mechanism is known to respond: GPT-2's "greater-than" behaviour on prompts such as

> The drought continued from the year 1724 through the year 17

The original model puts most probability on two-digit years after 24. Now take a *donor* prompt with a different
start year ("… from the year 1737 through the year 17"). Copy the donor's MLP input into the recipient at the last
token, and the original model's year preferences shift. The scalar `T` measures that shift, projected onto the
donor-minus-recipient "greater than" pattern.

The audit applies **the same physical change** to the original model and to each reduced model, then compares the
two shifts. A good replacement should reproduce the original's response, not just its unperturbed output.

## The hypothesis tested

The registered primary question (Day 1 protocol, unchanged through Day 3): at SVD rank k = 192, block 9, last-token
scope, does the calibrated Gaussian replacement reproduce the original model's natural year-swap response with a
**different RMS error** than the calibrated mean replacement? The direction was not assumed. The underlying thesis
was that a Gaussian treatment of the discarded SVD component would be the better, mechanism-preserving
compensation.

## What the method does

**Plain words.** Cut the block-9 MLP's input weight matrix down to its top 192 directions. Replace what was cut with
a fixed correction learned once from calibration prompts. Then check, on fresh prompts that were sealed until the
end, whether the cut-down model reacts to a year swap as the full model does.

**Precise version** (see `PROTOCOL.md`, `PROTOCOL_DAY2.md` and `PROTOCOL_DAY3.md`):

* **Model and site:** `openai-community/gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, zero-based
  block 9, MLP input projection `W_in = Wc^T` (HF `Conv1D` weight `[768, 3072]`). `W_out` is never modified.
* **Split:** input subspace projectors `P_R` (kept, rank k) and `P_B = I − P_R` (discarded). Pre-activation
  `a = z + q` with `z = W_in P_R x + b` and `q = W_in P_B x`.
* **Compensations:** `zero`, `bias` (`g(z+μ)`), `gauss` (`E g(z+μ+σε)`, probabilists' Gauss–Hermite with 64 nodes on
  Day 1, 128 nodes from Day 2 on), `gauss_wo` (weights-only moments). Moments are fixed after calibration.
* **Bases:** SVD of `W_in` (main); PCA of calibration activations (main from Day 3, chosen after calibration);
  whitened SVD (sensitivity); 5 fixed random bases (dimension control).
* **Scopes:** `last_token` (reduced weights only at the final position; primary) and `full_layer` (all positions).
* **Intervention:** at the final token, `x' = x_rec + P_S(x_don − x_rec)` for `S ∈ {R, B, RB}`. The residual bypass
  and all other positions are untouched.
* **Score:** `T = <Δlogits_J, v_rd> / ||v_rd||²` over the 100 two-digit year tokens J, with
  `v_rd = centre(1[j > d] − 1[j > r])`. Error `e_T = T_reduced − T_original` on the same physical change.
* **Reduced model:** a dense functional rank-k audit (dense `P_R` plus the original weights). It is **not** a
  compression method, and no speed or memory claim is made.

## Baselines, controls and decision rules

* **Baselines:** `zero` (plain truncation) and `bias` (calibrated mean) against `gauss` and `gauss_wo`. Random bases of
  equal dimension, coordinate masks (Day 2 only), whitened SVD and PCA serve as geometric controls.
* **Donor controls:** `same_example` (identity, exact zero), `same_year` (context change without a year change) and
  `wrong_donor` (a different year). On Day 2, norm-matched and pre-activation-matched versions were added.
* **Positive-control gate** (per split, before any reduced-model claim): mean PD > 0, the CI lower bound of mean
  `T_RB` (natural) > 0, and the CI lower bound of `T_natural − T_same_year` > 0.
* **Pre-registered Day-3 rules** (`PROTOCOL_DAY3.md`, frozen in `registration/DAY3_REGISTRATION.json` and
  `registration/DAY3_FROZEN_*.json` before test):
  * **P1 (primary):** `RMS(e_T^gauss) − RMS(e_T^bias)`, SVD, k = 192, last token, natural donor, S = RB. Two-sided
    95% stratified template-cluster bootstrap (B = 10,000). The result is a direction flag only: CI > 0 means
    Gaussian worse, CI < 0 means Gaussian better. There is no equivalence claim.
  * **S1 (secondary):** `RMS(e^{pca,bias}) − RMS(e^{svd,bias})`; supported if the one-sided 97.5% upper bound < 0.
  * **S2 (secondary):** `RMS(e^{pca,bias})`; within tolerance if the one-sided 97.5% upper bound < ε.
    ε = 0.05 × calibration original RB RMS (1.7947) = **0.08973707820478917**. ε was proposed after the calibration
    errors had been seen.
  * Bonferroni applies within {S1, S2} only. There is no global error-rate guarantee including P1.
  * KL-matched contrasts (Q1–Q3) were selected on validation and are descriptive on test.

## Results

### Main experiment: Day-3 held-out test (frozen, pre-registered)

Test split: frames F4/F5, 192 pairs, 18 template clusters. The validation split (F2/F3) was used only for the frozen
KL selections and is shown for comparison. Each test run directory includes a `snapshot/` of the exact code,
config and protocol used.

| quantity | validation (F2/F3) | **test (F4/F5)** | source field (`results.json`) |
|---|---|---|---|
| positive gate | valid; PD 0.904; T_RB 1.776 [1.692, 1.866] | valid; PD 0.905; T_RB 1.706 [1.647, 1.765] | `positive_control.pd_mean_ci`, `.T_RB_natural_mean_ci` |
| original RB RMS on this split | 1.886 | 1.758 | `analysis.orig_RB_rms_this_split` |
| **P1** RMS gauss − RMS bias (SVD k192 LT) | +0.0416 [0.0376, 0.0457] | **+0.0268 [0.0235, 0.0302]** → Gaussian worse | `analysis.P1.diff_cluster_ci` |
| RMS e_T gauss / bias (SVD k192 LT) | 1.312 / 1.271 | 1.301 / 1.275 | `analysis.P1.rms_gauss`, `.rms_bias` |
| **S1** PCA-bias − SVD-bias (upper 97.5%) | −1.197 (−1.123) | **−1.206 (−1.136)** → supported | `analysis.S1.diff_cluster_upper` |
| **S2** RMS PCA-bias (upper 97.5%) vs ε 0.0897 | 0.0740 (0.0790) | **0.0684 (0.0780)** → within tolerance | `analysis.S2.rms_cluster_upper` |

Sources: `runs/20261030T164723Z_day3_test/results.json` (test) and
`runs/20261030T164225Z_day3_validate/results.json` (validation).

Descriptive context on test (k = 192, last token; not decision criteria):

| quantity | value | source |
|---|---|---|
| unperturbed KL(orig‖reduced): SVD bias / SVD gauss / PCA bias | 0.142 / 0.132 / 0.0022 | test `results.json` → `unintervened[].kl_mean` |
| original response split T_R / T_B: SVD | 0.396 / 1.263 | test `results.json` → `controls[]` (natural, native) |
| original response split T_R / T_B: PCA | 1.683 / 0.016 | same |
| PCA-bias vector error ratio: local MLP-out / centred logits | 0.138 / 0.153 | `notes/codex_day3_vector_ratio_audit.json` → `results.test.last_token` |

How to read this:

* The Gaussian had *lower* unperturbed KL than the mean (0.132 vs 0.142), yet a *higher* response error. A better
  unperturbed fit did not mean a better causal response.
* For SVD, most of the natural response runs through the discarded subspace (T_B 1.263 of about 1.7). Any constant
  compensation structurally cannot carry it: the reduced model's response to a pure-B patch is exactly zero. This
  dominates the large SVD error.
* For PCA, almost all of the response runs through the kept subspace (T_B 0.016), which is why its error is small.
* S2 passing is **not** whole-vector preservation, circuit preservation or a per-pair guarantee. In the last-token
  scope, about 14% of the local and 15% of the logit response vector is not reproduced. Per frame (descriptive), the
  validation frame F3 exceeded ε and the test frame F4 passed with a narrow margin (`DAY3_SUMMARY.md` §2).

The same direction of P1 was seen in the exploratory calibration measurement on Day 1: +0.035 [0.031, 0.038]
(`runs/20261030T153707Z_pilot/results.json` → `primary_candidate_exploratory`, last_token, k = 192). That is
calibration data, not the main result.

### Exploratory precursors (calibration split only, not confirmatory)

* **Day 1** (`SUMMARY.md`, `runs/20261030T153707Z_pilot/`): the first measurements, all on calibration pairs. The
  discarded SVD subspace carries a large part of the year response.
* **Day 2** (`DAY2_SUMMARY.md`, `runs/20261030T160653Z_day2_run/`): geometric controls. At k = 192 the SVD split
  gives T_R 0.429 / T_B 1.240, while PCA gives T_R 1.724 / T_B 0.0005 (`results.json` → `controls[]`). This is why
  PCA was added as a main basis for Day 3, after calibration.

### Follow-up 1: limited propagation audit (separate protocol; labelled follow-up)

Question: for the Day-3 PCA192 + mean model, does weighting the local error by downstream sensitivities (feature
`D`) help *predict the size* of the logit-level error better than a strong 14-feature baseline `B`? This concerns
predicting the error norm, not reducing the model's error. Fresh frames N0–N9 and new nouns were used. Test set:
192 recipients × 3 intervention strengths α = 576 rows, 32 template clusters, 4 frames.

Pre-registered rule: GO if the one-sided 95% lower bound of `G = 1 − MSE(A)/MSE(B)` > 0.10; target ruled out if the
one-sided 95% upper bound ≤ 0.10; otherwise INCONCLUSIVE.

| quantity | value | source field |
|---|---|---|
| MSE(B), baseline | 3.4355e-4 | `primary.mse_B` |
| MSE(A) = B + D_exact | 2.8779e-4 | `primary.mse_A` |
| **G** | **0.1623** | `primary.G` |
| one-sided 95% lower / upper bound | 0.00082 / 0.3280 | `primary.G_lower_one_sided`, `.G_upper_one_sided` |
| **decision** | **INCONCLUSIVE** | `primary.decision` |
| secondary G_Taylor (descriptive) | 0.1319 (lower −0.0457, upper 0.3122) | `secondary_descriptive` |

Source: `followup/runs/20261030T192318Z_followup_evaluate_test/results.json`. Summary: `FOLLOWUP_SUMMARY.md`.

The two bounds are separate one-sided bounds, not a two-sided interval. The point gains per α were spread from
−0.26 (α = 0.5) to 0.27 (α = 1.5), descriptive only. The 10% target gain was neither confirmed nor ruled out.

### Follow-up 2: discarded-state response floor gate (development only; labelled follow-up)

Question: with the recipient context, the kept state and the kept intervention held fixed, how much does the
response vary when only the discarded component is swapped among 8 same-year donors? The answer is a normalised
error floor `L` for any predictor that cannot see the discarded state. The predeclared target was L0 = 0.05. This
was a **development** measurement on 48 groups, not a confirmatory test.

| quantity | value | source |
|---|---|---|
| s² (response scale) | 1.0934 | `discarded_response_floor_gate_v3/runs/20261031T105820Z_gate_v3_develop/development.json` → `s2` |
| **L** | **0.0396** (L² = 0.0015673) | same → `Lhat`, `Lhat2` |
| descriptive bootstrap lower / upper (one-sided each) | 0.0382 / 0.0409 | same → `dev_bootstrap_descriptive` |
| planning output (not acted on) | POWER_PLAN_CANDIDATE, N = 192 | `discarded_response_floor_gate_v3/runs/20261031T105841Z_gate_v3_power/power.json` |

The floor in this finite design was below the target. The branch was closed without a confirmatory run
(`notes/discarded_response_floor_gate_v3_results_2026-10-31.txt`, in Turkish). This experiment did not test
renormalisation-group (RG) or WeightWatcher ideas.

### Superseded or invalid runs (kept as records, not results)

* `discarded_response_floor_gate/` **v1**: design superseded before any outcome, because of a donor-noun aliasing
  issue.
* `discarded_response_floor_gate/` **v2** develop run `20261031T103329Z_gate_develop`: **EVALUATION_INVALID**. 15 of
  48 identity controls exceeded the fixed 1e-4 threshold (max 2.136e-4), traced to MPS float32 padding/layout
  sensitivity. v3 is its technical remeasurement.
* `runs/20261030T153332Z_quadcheck` (failed: MPS float64 cast) and `runs/20261030T153344Z_quadcheck` (32 nodes
  not converged).
* `followup/runs/20261030T192217Z_followup_fit` (failed before model load: MPS unavailable in a sandbox) and the
  earlier follow-up smoke `20261030T191925Z_followup_smoke`.

## Why the project was not continued

* The registered primary result went against the thesis on held-out data. The same direction appeared on
  calibration, validation and test.
* The positive PCA result is narrow (one task, two frames per split, scalar metric, basis chosen after calibration),
  and its explanation is mostly geometric: the PCA subspace simply carries the response.
* Follow-up 1 was INCONCLUSIVE. The reviewers and coordinator closed it rather than enlarging it.
  `FOLLOWUP_SUMMARY.md` states that stopping "is a research-resource judgment, not a statistical impossibility".
* Follow-up 2 found a floor below its target in the development design. The branch was closed. Any reopening would
  need a new, justified hypothesis and a fresh registration.

## What kind of evidence is what

**(a) Scientific experiments** (results above): the Day-1 pilot and Day-2 run (exploratory, calibration only); the
Day-3 validation and test (pre-registered; the test is the main result); follow-up 1 (pre-registered, separate
splits); follow-up 2 v3 (development only).

**(b) Engineering checks.** These verify that the code does what it says. They do not test the hypothesis.

* Unit tests: `tests/` (66 passed, `logs/pytest_day3_20261030T164003Z.log`), `tests_followup/` (27 passed,
  `followup/logs/pytest_followup_20261030T192123Z.log`) and `discarded_response_floor_gate_v3/tests/` (48 passed,
  `discarded_response_floor_gate_v3/logs/pytest_gate_v3_20261031T105522Z.log`). The first test run had 5 failures
  (`logs/pytest_20261030T153120Z.log`), kept as a record.
* Smoke, quadrature and benchmark runs: the `*_smoke`, `*_quadcheck`, `*_bench` and `*_calibrate` directories.
* Numerical diagnostics: `notes/gate_numerical_diagnostic/`.
* Registration and freeze files with SHA-256 bindings: `registration/`, `followup/registration/` and the gate
  registration folders.

**(c) Literature and judgement commentary.** At each stage, AI models (GPT Pro, Antigravity/Gemini, Claude/Fable)
reviewed the results and Codex audited them. These records are judgement, not measurement. Where Codex recomputed
numbers independently from the raw files, those audits are in `notes/codex_*.json`. Included review and decision
notes: `notes/three_model_review_day1.md`, `notes/claude_*.md`, `notes/day3_posthoc_diagnostic.md` (post-hoc,
descriptive) and `notes/followup_posthoc_loss_contributions.md` (post-hoc, descriptive). Some review records were
**not** included because they contain private chat links (see "Provenance").

## Limitations

* One model (GPT-2 small), one layer (block 9), one matrix (`W_in`), one anchor rank (192) and one synthetic task
  (greater-than years).
* Only 2 frames per held-out split, so the cluster structure is thin. Intervals are nominal, approximate and
  conditional on the fitted bases and moments.
* ε was chosen with knowledge of calibration errors. PCA was chosen after calibration.
* In `full_layer`, scope and moment pool are confounded (different calibration pools).
* The intervention patches only the MLP input at the final token. Endpoint reconstruction is a strong explanation
  candidate for the PCA result, not a proven mechanism.
* There was no general-text or out-of-task evaluation, and no compression or speed measurement.
* Follow-up 1 depends on 8 fixed random probes whose uncertainty is not in the bootstrap. α = 1.5 is an
  extrapolation.
* Follow-up 2 used synthetic hybrid inputs that need not lie on the natural activation manifold. It was not blind
  (v2 outputs had been seen).

## Repository map

| path | content |
|---|---|
| `PROTOCOL.md`, `PROTOCOL_AMENDMENT_v1.1.md` | Day-1 registered protocol and amendment |
| `PROTOCOL_DAY2.md`, `PROTOCOL_DAY3.md` | Day-2 (geometric controls) and Day-3 (held-out) protocols |
| `SUMMARY.md`, `DAY2_SUMMARY.md`, `DAY3_SUMMARY.md` | stage reports; `DAY3_SUMMARY.md` is the final main report |
| `FOLLOWUP_DESIGN.md`, `FOLLOWUP_SUMMARY.md` | follow-up 1 design record and final report |
| `docs/PROJECT_NOTES.md` | the project's original README (unchanged) |
| `config/` | all parameters (`pilot_v1.json`, `day2_v1.json`, `day3_v1.json`) |
| `registration/` | SHA-256 registrations and Day-3 freeze files |
| `manifests/`, `manifests_day3/`, `metadata/` | prompt splits, pairs, sealed held-out prompts, continuation mask |
| `calibration_day3/calib_index.json` | index and hashes of the frozen calibration artifact (artifact itself not included) |
| `scp/`, `run.py`, `run_day2.py`, `run_day3.py` | main code and CLIs |
| `tests/`, `pytest.ini` | main unit tests |
| `runs/<utc>_<stage>/` | one directory per run: `run_manifest.json`, `log.txt`, results, `snapshot/` |
| `followup/`, `run_followup.py`, `tests_followup/` | follow-up 1 code, protocol, config, manifests, registration, runs |
| `discarded_response_floor_gate/` | follow-up 2 v1/v2 (superseded / invalid) |
| `discarded_response_floor_gate_v3/` | follow-up 2 v3 code, protocol, runs |
| `notes/` | audits, post-hoc diagnostics, included review notes |
| `logs/` | pytest and stdout logs |
| `env/`, `requirements.txt`, `requirements.lock.txt` | recorded environment |

## Reproducing

**Recorded environment** (`env/hardware_and_versions.txt`, `requirements.txt`, `requirements.lock.txt`): Apple M2
Max with 32 GB and macOS 26.6.2; Python 3.13.0; torch 2.15.0 (MPS); transformers 5.22.1; numpy 2.5.4; scipy 1.18.1;
tokenizers 0.23.3; safetensors 0.8.0; huggingface_hub 2.4.0; pytest 9.2.0. The full pinned set is in
`requirements.lock.txt`.

**Model download (the only external dependency):** `openai-community/gpt2`, revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` (weights and tokenizer). `scp/__init__.py` sets
`HF_HOME=<repo>/.cache/hf` and `HF_HUB_OFFLINE=1`, so the model must be downloaded once beforehand. The code expects
MPS and refuses to fall back silently to another device.

**Commands, as recorded in the project docs** (`docs/PROJECT_NOTES.md`, `SUMMARY.md` §7, `DAY3_SUMMARY.md` §7, the
`run_followup.py` docstring and `DEV_PROTOCOL_v3.md` §V3). Run them from the repository root:

```bash
python3.13 -m venv .venv
PIP_CACHE_DIR=$PWD/.cache/pip .venv/bin/pip install -r requirements.lock.txt
HF_HOME=$PWD/.cache/hf .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('openai-community/gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e')"
.venv/bin/python -m pytest -v
.venv/bin/python run.py splits      # regenerates identical manifests (seeded; hashes must match registration)
.venv/bin/python run.py quadcheck
.venv/bin/python run.py smoke
.venv/bin/python run.py bench
.venv/bin/python run.py pilot
.venv/bin/python -m scp.report runs/<pilot_run_id>
.venv/bin/python run_day3.py smoke
.venv/bin/python -c "import scp; from scp.common import load_config; from scp import day3; day3.verify_frozen(load_config('config/day3_v1.json')); print('ok')"
.venv/bin/python run_followup.py <stage>   # prepare -> smoke -> register -> fit -> validate -> freeze -> predict-test -> evaluate-test -> report
.venv/bin/python -m pytest discarded_response_floor_gate_v3/tests -v -p no:cacheprovider
```

**Paths.** The code resolves files relative to the repository root, so a clone can live anywhere. The project docs
begin with `cd /Users/emre/Documents/Codex/ML_Projects/spectral_causal_pilot` and use
`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13`; replace both with your own clone path and Python
3.13. The frozen evidence files (`runs/*/run_manifest.json`, `runs/*/log.txt`, the follow-up and gate run manifests,
several logs, notes and summaries) contain the original machine's absolute paths (`/Users/emre/...`). They are kept
**unmodified** as evidence, because they are hash-bound to registrations. Do not edit them; new runs write new
directories.

**Guards you will meet.** Stages refuse to overwrite registrations, freezes, manifests or write-once markers. The
test stages refuse if any hash-bound file changed. Day-3 validation and test, follow-up 1 and follow-up 2 v3 need
`calibration_day3/calib_artifact.pt` (117 MB), which is **not included**. `run_day3.py calibrate` rebuilds it from
the calibration pool, but later stages verify its SHA-256 against `calibration_day3/calib_index.json` and the
registrations. A rebuild on other hardware or software may not be bit-identical, and those stages will then refuse.

**What can be recomputed from the included files alone (no model, no re-run):**

| reported number | recompute from |
|---|---|
| Day-3 test P1, S1, S2 point values; RMS e_T per method; descriptive decomposition by S | `runs/20261030T164723Z_day3_test/reduced_rows.csv` (columns `basis, method, scope, k, S, template, e_T, T_orig`). The P1/S2 point values above were re-derived from this file and match `results.json` exactly. The bootstrap CIs need the stratified template bootstrap of `PROTOCOL_DAY3.md` §5 (implemented in `scp/day3.py`). |
| Day-3 test unperturbed KL and PD; KL gaps of the frozen pairs on test | `runs/20261030T164723Z_day3_test/unperturbed_rows.csv` |
| Day-3 positive gate (test and validation) | `runs/*_day3_{test,validate}/gate_rows.csv` |
| Day-1 positive control (PD, T for natural / same-year / wrong donor) | `runs/20261030T153643Z_smoke/smoke_per_pair.npz` and `runs/20261030T154234Z_smoke/smoke_per_pair.npz` (four float64 arrays of 192 values each) |
| Follow-up 1 MSE(A), MSE(B), G and per-α / per-frame values | `followup/runs/20261030T192301Z_followup_predict_test/test_predictions.csv` + `followup/runs/20261030T192318Z_followup_evaluate_test/test_labels.csv` + frozen `s` in `followup/registration/FOLLOWUP_FROZEN.json` + `followup/manifests/test_recipients.csv`. G = 0.16230704 was re-derived from these. |
| Follow-up 1 post-hoc contribution shares | `notes/followup_posthoc_loss_contributions_rows.csv`, `.csv`, `.py` |
| Follow-up 2 v3 L, L², s² | `.../20261031T105820Z_gate_v3_develop/groups_U.csv` (L = 0.039589 re-derived), and the raw J-logits `raw_J_logits.pt` (a torch tensor file of model outputs, not weights; it needs torch to open) |
| Follow-up 2 v3 power table | `.../20261031T105841Z_gate_v3_power/power_rows.csv`, `power.json` |

**Not recomputable from included files:**

* Day-1 and Day-2 per-pair numbers: their per-row CSVs (13–24 MB each) were left out for size. Their `results.json`
  and `summary_tables.md` are included.
* Day-3 validation per-pair rows and the Day-3 test `control_rows.csv` (needed for the geometric-control tables and
  part of `notes/day3_posthoc_diagnostic.md`): left out for size. Aggregates are in `results.json`.
* Follow-up 1 probe VJP tensors and follow-up 2 activation tensors: left out. They are not needed for the reported
  numbers above.

## References

Cited in the project's own protocol (`PROTOCOL.md`, `PROTOCOL_AMENDMENT_v1.1.md`), as given there:

* Greater-than circuit, Hanna et al. 2023: arXiv:2305.00586. The task format and PD definition are taken from it;
  no replication is claimed.
* LASER: arXiv:2312.13558
* Beyond Components: arXiv:2511.20273 (the protocol notes that its App. B.2 already performs class-mean swaps)
* SVD-LLM: arXiv:2403.07378
* Optimal Ablation: arXiv:2409.09951

The protocols also mention WeightWatcher (spectral cutoff `k*`, listed as not done) and Wilsonian RG (explicitly not
tested), without a citation.

## Provenance and license

* Copied from a local project folder (`spectral_causal_pilot`, dated 2026-10-30 to 2026-10-31). The folder was not
  a git repository. Code and tests are the frozen versions used for the recorded runs.
* The project docs record that the follow-up code and tests were written by an AI coding agent (Claude Code, Opus
  5.5 High), that Codex acted as coordinator and independent auditor, and that results were reviewed by AI models
  (GPT Pro, Antigravity/Gemini, Claude/Fable).
* Left out of this public copy:
  * the Python virtual environment, Hugging Face and pip caches, `__pycache__`;
  * torch tensor files (`*.pt`) except one small output-logit file;
  * the large per-row CSVs listed above;
  * review and coordination notes that contain private chat-session links or are verbatim chat prompts or
    transcripts: `notes/three_model_review_day2.md`, `notes/three_model_review_day3.md`,
    `notes/three_model_review_day3_prevalidation.md`, `notes/followup_debate.md`,
    `notes/research_direction_rg_2026-10-31.md`, `notes/gpt_pro_*`, `notes/discarded_gate_coordination_20261031.json`,
    `notes/gate_v3_actual_review_record_20261031.json` and
    `notes/gate_v3_technical_remeasurement_instruction_20261031.txt`.

  Some included documents still reference these paths. Their decisions are summarised in `DAY3_SUMMARY.md`,
  `FOLLOWUP_SUMMARY.md`, `discarded_response_floor_gate_v3/DEV_PROTOCOL_v3.md` and
  `notes/discarded_response_floor_gate_v3_results_2026-10-31.txt`. Because of these exclusions, the hash-verification
  commands that cover the whole protected file set (for example `verify-protected` in follow-up 2, whose
  `protected_manifest.json` lists `calib_artifact.pt`, the `*.pt` tensors and the per-row CSVs) will report missing
  files in this copy.
* No license file is included; all rights reserved by the repository owner unless stated otherwise. Contact the
  owner before reuse.
