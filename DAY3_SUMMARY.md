# DAY3_SUMMARY — held-out validation and test (day3-v1), 2026-10-30

**Status: final.** The 3-day pilot is complete. The joint conclusion of Codex, GPT Pro, Antigravity and Claude is
in §4; the full review record is `notes/three_model_review_day3.md`. Codex's independent audits are complete. No
source, protocol or config changed after registration.

## 1. What was pre-registered and how it was run

* Protocol `PROTOCOL_DAY3.md` and config `config/day3_v1.json` (`day3-v1`, final after review). The
  pre-validation review is recorded in `notes/three_model_review_day3_prevalidation.md`.
* Scope:
  * GPT-2 small, block 9, `W_in`; k ∈ {96, 192, 384}; anchor and confirmatory rank **k = 192**, fixed in advance;
  * main bases SVD and PCA (PCA was chosen after the calibration results); wSVD as secondary sensitivity; 5 fixed
    random seeds as the dimension control;
  * `last_token` primary, `full_layer` secondary.
* Frozen calibration artifact `calibration_day3/calib_artifact.pt` (168 CPU-float64 tensors; fitted once on the
  Day-2 moment pool). It was never refitted; its hash was checked after every run.
* Held-out pairs `manifests_day3/` (validation F2/F3, test F4/F5): 192 pairs each, 12 cells, each cell containing all 9 templates of its own frame,
  every cell, 0 tokenization failures. Hashed before measurement.
* Chain:
  1. 66/66 tests (`logs/pytest_day3_20261030T164003Z.log`).
  2. Calibration-only smoke `runs/20261030T164041Z_day3_smoke`: artifact = refit, Δ = 0; Day-2 replay Δ = 0
     over 40 checks.
  3. Registration `registration/DAY3_REGISTRATION.json` (16:42:18Z; ε = 0.08973707820478917 incl. reference
     CSV SHA).
  4. Validation `runs/20261030T164225Z_day3_validate` (ok, 240 s).
  5. Codex validation audit.
  6. Freeze `registration/DAY3_FROZEN_{SELECTION,EPSILON}.json`; `verify_frozen` OK.
  7. Test `runs/20261030T164723Z_day3_test` (ok, 207 s).
  8. Codex test audit.
* Inference:
  * stratified bootstrap over 12 frame × direction × diff-bin cells with fixed weights; templates resampled
    within frame (primary), pairs within cell (supporting); B = 10,000;
  * P1 two-sided 95%; S1/S2 one-sided 97.5% upper bounds (Bonferroni within the secondary family only);
  * intervals are nominal, approximate and conditional, with no global FWER.

## 2. Results

| | validation (F2/F3) | test (F4/F5) |
|---|---|---|
| gate | valid; PD 0.904; T_RB 1.776 [1.692, 1.866] | valid; PD 0.905; T_RB 1.706 [1.647, 1.765] |
| per-frame T_RB | F2 1.37, F3 2.14 | F4 1.707, F5 1.705 |
| original RB RMS (ε / it) | 1.886 (4.8%) | 1.758 (5.1%) |
| **P1** SVD k192 LT: RMS gauss − RMS bias | +0.0416 [0.0376, 0.0457] | **+0.0268 [0.0235, 0.0302]** → gauss higher |
| **S1** RMS PCA-bias − RMS SVD-bias (upper 97.5%) | −1.197 (upper −1.123) | **−1.206 (upper −1.1365)** → supported |
| **S2** RMS PCA-bias (upper 97.5%) vs ε 0.0897 | 0.074 (upper 0.079) | **0.068 (upper 0.078)** → within tolerance |
| S2 per-frame upper (descriptive; does not change the criterion) | F2 0.050, F3 0.100 (> ε) | F4 0.08907559 (ε 0.08973708), F5 0.073 |
| S2 NRMSE (calibration / split denominator) | 0.041 / 0.039 | 0.038 / 0.039 |

The calibration value of P1 (Day 1) was +0.035.

**Frozen KL-matched contrasts** (selected on validation; descriptive on test; test gap recomputed, all within the
rule; all same-rank comparisons):

| family | scope | pair | test gap | test RMS e_RB difference [95% cluster CI] |
|---|---|---|---|---|
| Q1 SVD-bias vs PCA-bias | LT, FL | none matched | — | no iso-KL claim |
| Q2 SVD gauss vs bias | LT | 384/384 | 0.003 | +0.029 [0.025, 0.033] (gauss higher) |
| Q2 SVD gauss vs bias | FL | 192/192 | 0.050 | +0.007 [0.004, 0.009] (gauss higher) |
| Q3 PCA gauss vs bias | LT | 384/384 | 1.5164e-5 | −1.53e-7 [−1.32e-6, 1.01e-6]: not distinguished, numerical scale, not evidence of equivalence |
| Q3 PCA gauss vs bias | FL | 192/192 | 0.045 | +0.008 [0.002, 0.014] (gauss higher) |

**Descriptive context (test):**
* LT k192 KL:
  * SVD zero / bias / gauss: 0.140 / 0.142 / 0.132;
  * PCA bias 0.0022 (calibration 3.4e-4);
  * wSVD bias 0.0026.
* SVD R-only RMS: gauss 0.110 < bias 0.128.
* PCA B on test is **not** null: RMS‖δ_B‖ at k192 is 0.577 (calibration 0.14), and T_B is 0.016
  [0.009, 0.025].
* Random bias RB RMS at k192 (5 seeds): 1.252–1.274, vs SVD 1.275.
* T_R / T_B at k192: SVD 0.40 / 1.26, random_s101 0.41 / 1.24, PCA 1.68 / 0.016.

Full tables are in each run's `results.json`; raw per-pair CSVs sit next to them.

### Metric boundary: scalar T vs whole response vectors (test, PCA-bias k192, natural RB; descriptive)

| scope | scalar T NRMSE (S2 metric) | local MLP-out vector, RMS relative error | centred full-vocab logit vector, RMS relative error |
|---|---|---|---|
| last_token | 0.0389 (test denominator) | **0.1384 (~13.8%)** | **0.1534 (~15.3%)** |
| full_layer | — | 0.4093 (~40.9%) | 0.4646 (~46.5%) |

* These are point descriptives from the existing data; no new measurement and no new criterion.
* The vector ratios divide by the RMS of the original **vector** response, so they are not the same metric as the
  scalar T NRMSE.
* Codex recomputed them independently from the raw scalar norms (`notes/codex_day3_vector_ratio_audit.json`;
  max difference 6.3e-8, float32).
* **S2 passing on the scalar T response is not whole-vector preservation:** about 14–15% of the
  local/logit response vectors is not reproduced in `last_token`, and about 41–46% in `full_layer`.

## 3. Independent audits (Codex)

* Index: `notes/codex_day3_independent_audit.json`.
  * `verify_frozen` still passes after test.
  * The Day-2 vs Day-3 snapshots match in 35 files. Only `scp/common.py` differs, in its two glob lists (needed
    for the Day-3 snapshot); the shared numerics are unchanged.

* Validation, `notes/codex_day3_validation_independent_audit.json`:
  * row counts, and every group has 192 unique pairs;
  * every control mean, reduced RMS and KL mean differs by 0;
  * P1/S1/S2 and both frames re-bootstrapped (10,000) with a different sufficient-statistic implementation:
    max 2.22e-16;
  * KL selections reproduced; hashes unchanged.
* Test, `notes/codex_day3_test_independent_audit.json`:
  * raw error / T_orig / gate alignment;
  * every control mean, reduced RMS and KL mean differs by 0;
  * main and per-frame cluster CIs ≤ 2.22e-16;
  * matched-norm relative error max 1.18e-8;
  * all hash/freeze bindings unchanged.

## 4. Final joint conclusion (Codex, GPT Pro, Antigravity, Claude)

* **NO-GO** for the original Gaussian/SVD-specific mechanism thesis and for an MI-novelty claim in that framing.
* **P1 (registered primary), test:** at SVD k192 `last_token`, calibrated Gaussian compensation *increased* the
  natural RB response error relative to calibrated bias: **+0.0268 [0.0235, 0.0302]**, ~2.1% of the bias error.
  * This is not "no result": the data support the error-increasing direction in this setting.
  * The same direction appeared on validation (+0.0416) and calibration (+0.035).
  * Both SVD models have large total error (RMS ~1.27–1.30).
* **S1/S2 (secondary; PCA chosen after calibration), test:**
  * PCA + bias at the same rank had much smaller scalar response error than SVD + bias (S1 −1.206, upper
    −1.1365).
  * It met the pre-registered **aggregate scalar operational tolerance** (S2 0.0684, upper 0.0780 < ε 0.0897).
  * That pass stands and is not cancelled afterwards by a stricter criterion.
  * It is **not** equal quality, whole-vector preservation (LT local/logit vector errors ~13.8% / ~15.3%), "96%
    mechanism preservation", circuit preservation, per-pair guarantee, all-frame generalisation, or anything
    beyond this task.
  * Per frame (descriptive): validation F3 > ε; test F4 has a narrow margin.
* **KL matching:**
  * Q1 found no iso-KL pair; that "none" matters.
  * Q2 shows that, within same-rank KL-matched pairs, the responses can still differ. This is a narrow
    descriptive finding, not statistical equivalence.
  * Q3-LT is not distinguished.
* The RB intervention puts the donor's real activation only into the patched MLP input at the final token. It
  keeps the residual bypass and the rest of the network in the recipient context. Endpoint reconstruction is a
  strong explanatory candidate, not proof of the whole explanation.
* All intervals are nominal, approximate and conditional; 2 frames per split.

Joint sentence (paraphrase of GPT Pro, accepted by all): *In the locked held-out evaluation, the calibrated
Gaussian increased the primary natural-response error relative to calibrated bias for SVD. At the same rank, PCA +
bias had smaller error and met the aggregate scalar tolerance on the test mixture. This is response fidelity
limited to these frames and this intervention, not equal-quality superiority, circuit preservation or an RG
mechanism.*

## 5. Post-hoc diagnostic from existing records (not confirmatory)

GPT Pro asked whether PCA + bias also holds under separate R / B / RB interventions, or whether the RB result is an
endpoint/cancellation effect. This was answered **only from the existing per-pair rows**
(`notes/day3_posthoc_diagnostic.md`): no new run, no new criterion, no pass claim, and S2 unchanged. GPT Pro had
not seen these numbers when it wrote its final review.

PCA-bias k192 LT RMS e_T on test:
* R 0.0474;
* B 0.0584, which equals the original T_B that the reduced model structurally cannot carry;
* RB 0.0684.

Validation: 0.0464 / 0.0589 / 0.0740.

Test decomposition `MSE_RB = MSE_R − 2E[C·e_R] + E[C²]`:

| model | MSE_R | −2E[C·e_R] | E[C²] | MSE_RB |
|---|---|---|---|---|
| SVD bias | 0.0165 | −0.2369 | 1.8450 | 1.6246 |
| SVD gauss | 0.0120 | −0.1633 | 1.8450 | 1.6938 |
| PCA bias | 0.0022 | −0.0006 | 0.0031 | 0.0047 |

Identity residuals are ≤ 1.3e-5.

Reading (arithmetic only):
* For SVD, the B + interaction term dominates, and bias's larger negative cross term explains P1.
* For PCA, the low RB error is not mainly cancellation: the R error and the B + interaction response are both
  small.
* Still scalar-T only, and not a network mechanism.

**Record distinctions:**
* The R/B/RB errors and the local/logit vector ratios were already part of the **pre-planned descriptive grid**
  (`reduced_rows.csv`, `results.json`).
* Only the focus on them after GPT Pro's review, and any `max{…} ≤ ε` criterion, are **post-hoc**.
* GPT Pro did not see these extra numbers. Its approval of the joint sentence must not be read as covering them.
* Reduced-model Δ_B = 0 is structural and is **not** evidence of success. The original B RMS is small but nonzero,
  and there is no claim that all B-direction information is preserved.

**Closing:** Antigravity read GPT Pro's synthesis and this diagnostic package and approved closing the pilot with
this consensus (no blocking objection). Codex accepted. The result is not called "flawless".

## 5b. Next research (separate; not started)

These need a new protocol and fresh splits:
* separate-R/B/RB fidelity criteria;
* general-text or out-of-task pools;
* iso-KL with new ranks or interpolation;
* other layers, models or tasks;
* resample ablation.

## 6. Limitations

* One model, layer, task and anchor rank.
* 2 frames per split, so the cluster structure is thin.
* The scope/moment-pool confound in `full_layer`.
* Nominal, conditional intervals.
* ε chosen with calibration knowledge.
* PCA selected after calibration.
* The positive gate is not a weak-signal guarantee.
* No general-text or out-of-task evaluation.
* Dense functional audit, not compression.

## 7. Reproduction

```bash
cd /Users/emre/Documents/Codex/ML_Projects/spectral_causal_pilot
.venv/bin/python -m pytest -v                                        # 66 tests (some CLI guards skip once registered/frozen)
.venv/bin/python run_day3.py smoke                                   # calibration-only checks
.venv/bin/python -c "import scp; from scp.common import load_config; from scp import day3; day3.verify_frozen(load_config('config/day3_v1.json')); print('ok')"
```

`datasets`, `calibrate`, `register` and `freeze` refuse to overwrite. Re-running `validate` or `test` creates new
run directories under the same frozen rules.

## 8. Artifacts

| what | path |
|---|---|
| protocol / config | `PROTOCOL_DAY3.md`, `config/day3_v1.json` |
| registration / freeze | `registration/DAY3_REGISTRATION.json`, `registration/DAY3_FROZEN_SELECTION.json`, `registration/DAY3_FROZEN_EPSILON.json` |
| held-out pairs | `manifests_day3/{validation_pairs.csv, test_pairs.csv, datasets_index.json}` |
| calibration artifact | `calibration_day3/{calib_artifact.pt, calib_index.json}`; built by `runs/20261030T163737Z_day3_calibrate` |
| smoke | `runs/20261030T164041Z_day3_smoke/smoke.json` |
| validation | `runs/20261030T164225Z_day3_validate/` (`results.json`, `gate_rows.csv`, `control_rows.csv`, `reduced_rows.csv`, `unperturbed_rows.csv`, `snapshot/`), stdout `logs/day3_validate_stdout_20261030T164224Z.log` |
| test | `runs/20261030T164723Z_day3_test/` (same files), stdout `logs/day3_test_stdout_20261030T164722Z.log` |
| code / tests | `scp/day3.py`, `run_day3.py`, `tests/test_day3.py` |
| reviews / audits | `notes/three_model_review_day3.md` (final), `notes/day3_posthoc_diagnostic.md`, `notes/three_model_review_day3_prevalidation.md`, `notes/codex_day3_independent_audit.json` (index), `notes/codex_day3_vector_ratio_audit.json`, `notes/codex_day3_validation_independent_audit.json`, `notes/codex_day3_test_independent_audit.json`, `notes/claude_day3_test_reading.md` |
