# DAY2_SUMMARY — geometric controls (day2-v1), 2026-10-30

**Label: EXPLORATORY, calibration split only.** Day 2 of 3. Validation and test were not opened. No ε,
confirmatory rank, best seed or best basis was chosen. RB stays the primary *candidate*; R is secondary. Day-1
files (`PROTOCOL.md`, amendment, `config/pilot_v1.json`, `manifests/`, Day-1 runs) are unchanged, hash-checked
by `run_day2.py` on every call.

**Review outcome (final for Day 2):** Codex, GPT Pro and Antigravity give a conditional GO to a *bounded* Day-3
validation. Details, provenance, the consensus wording and what is not established are in
**`notes/three_model_review_day2.md`**.

## 1. What was built and checked

* `PROTOCOL_DAY2.md`, `config/day2_v1.json`. Registration `registration/DAY2_REGISTRATION.json` written
  2026-10-30T16:06:26Z, **before** the Day-2 measurement run; it hashes the protocol, config, Day-1 manifests,
  `metadata/` and the source+tests tree (`90a9bc2c…`). The measurement run's source tree hash is identical.
* Code: `scp/bases.py` (svd, pca, coord, Haar random, whitened SVD), `scp/matching.py` (l2 / pre-activation
  matching with degenerate/oos rules and asserts), `scp/day2.py` (quadcheck, smoke, run, aggregation,
  vectorised bootstrap), `scp/report_day2.py`, `run_day2.py`.
* Codex review items applied **before measurement**:
  * the two paired contrasts (cross-basis vs SVD-native; donor-control vs basis-native) share a common pair mask
    and have an oos-filtered variant;
  * `gauss_wo` wording;
  * whitened mean restoration is exact only at the pre-activation level;
  * budget checks and checkpoints in the control loop;
  * per-pair `unperturbed_rows.csv`;
  * smoke as a failure gate;
  * pair-iid and template-cluster CIs for every reported mean/RMS of T, T error, KL and PD (local/logit norm RMS
    values are point summaries without CIs);
  * the original-model "B effect must be large" precondition removed from the tests (replaced by
    nonzero-projection and independent branch checks).
* **Tests: 49/49 pass** (`logs/pytest_day2_20261030T160558Z.log`). The 30 Day-1 tests are unchanged; the 19 new
  ones cover:
  * PCA/coord deterministic selection (incl. tie-break) and Haar seed repeat/orthogonality/nesting;
  * ridge rule, and whitened oblique algebra: `P²=P`, `P+Q=I`, `P≠Pᵀ`, `W_in P = (W_in L)_k L⁻¹`, residual
    identity, full rank, row/column orientation;
  * row-map moments incl. weights-only for the oblique basis;
  * full-rank = original for all bases;
  * branch patch and reduced B-null for all bases;
  * no moment leak from the current x;
  * matching rules incl. zero and out-of-support cases, common-mask contrasts;
  * whitened MPS/CPU parity (< 2e-3) and the CLI guards (validation/test refused, layer 10 refused).
* **Quadcheck** (`runs/20261030T160333Z_day2_quadcheck`): max σ over all bases, ranks and pools is 1.774
  (pca k=96, all-token). By the rule the bound is 2.5 and the node count 128 (grid error 3.1e-10, real samples
  3.5e-14; 64 nodes gave 2.7e-7 on the grid). Both were written to the config before registration. Day-1 settings
  (64 / 2.0) are untouched.
* **Smoke gate passed** (`runs/20261030T160634Z_day2_smoke`):
  * Day-1 replay at SVD k=192: original T_R/T_B/T_RB and bias/zero reduced T: max |ΔT| = 0 over the compared values;
    gauss differs by ≤ 2.0e-5 (64 → 128 nodes);
  * whitened reduced B-null 1.8e-4 logits; RB patch = donor 8.6e-6;
  * matching rel. error ~5e-9.

## 2. The run

`runs/20261030T160653Z_day2_run`: status **ok**, **257 s** of the 900 s budget.
* Original controls took 49 s; each basis took ~21 s (3 ranks × 4 methods × 2 scopes); aggregation 8 s.
* Positive-control gate valid (T_RB 1.72 [1.66, 1.79]).
* Raw rows: `control_rows.csv` (94,656), `reduced_rows.csv` (124,416), `unperturbed_rows.csv` (41,472). These are
  row counts; **N = 192 pairs** (18 templates, 2 frames).
* Matching max rel. error 1.3e-15 (f64), 1.7e-8 (f32); 0 degenerate rows; 6,890 rows flagged out of support.
* Max |T_B| over reduced rows 8e-7 (structural). Decomposition identity residual ≤ 1.9e-5 in every reduced cell.

Full tables (1,501 lines) are in `runs/20261030T160653Z_day2_run/summary_tables.md`. Key numbers follow; random
bases are given as mean over 5 seeds [min, max] of per-seed means, never pooled into N.

## 3. Results (exploratory)

### 3a. Native R/B split of the natural year response (original model)

| k | basis | T_R | T_B | RMS‖δ_R‖ | RMS‖δ_B‖ |
|---|---|---|---|---|---|
| 96 | svd | 0.185 [0.165, 0.203] | 1.52 | 0.78 | 3.28 |
| 96 | coord | 0.200 | 1.49 | 1.17 | 3.16 |
| 96 | random ×5 | 0.200 [0.154, 0.244] | 1.48 [1.43, 1.53] | 1.2 | 3.15 |
| 96 | pca / wsvd | 1.72 / 1.72 | 0.001 / 0.002 | 3.36 / 3.35 | 0.20 / 0.22 |
| 192 | svd | 0.429 | 1.24 | 1.27 | 3.12 |
| 192 | coord | 0.440 | 1.21 | 1.67 | 2.93 |
| 192 | random ×5 | 0.429 [0.401, 0.467] | 1.21 [1.18, 1.24] | 1.7 | 2.9 |
| 192 | pca / wsvd | 1.72 / 1.72 | 0.001 / 0.002 | 3.37 / 3.36 | 0.14 / 0.18 |
| 384 | svd | 0.862 | 0.768 | 2.01 | 2.71 |
| 384 | coord | 0.822 | 0.803 | 2.39 | 2.38 |
| 384 | random ×5 | 0.856 [0.806, 0.893] | 0.767 [0.729, 0.826] | 2.4 | 2.4 |
| 384 | pca / wsvd | 1.72 / 1.72 | 0.001 / 0.001 | 3.37 / 3.36 | 0.09 / 0.14 |

T_RB = 1.72 for every basis.

Interaction `I_T`:
* svd: 0.016, 0.056, 0.095;
* coord: 0.037, 0.072, 0.099;
* random_s101: 0.039, 0.080, 0.101;
* pca/wsvd: mean ≈ 0.

**PCA/wsvd B is small on average, not inert per pair** (Codex CSV diagnostic, k=192):
* PCA: T_B mean 0.000506, RMS 0.011653, max |T_B| 0.035121;
* wSVD: T_B mean 0.002054, RMS 0.013235, max |T_B| 0.042060.

T_R ≈ T_RB does not by itself show that the whole physical δ lies in R; the δ norms are reported separately.

### 3b. Cross-basis contrasts, natural donor (paired, common mask; k=192 shown)

`T[basis] − T[svd native]`:

| basis | S | native | l2-matched to SVD | preact-matched to SVD |
|---|---|---|---|---|
| random_s101 | R | −0.017 [−0.047, 0.015] | −0.132 [−0.160, −0.104] | +0.016 [−0.014, 0.048] |
| random_s101 | B | −0.008 [−0.038, 0.022] | +0.093 [0.062, 0.123] | −0.174 [−0.202, −0.147] |
| coord | R | +0.011 [−0.019, 0.043] | −0.105 [−0.133, −0.076] | +0.052 [0.021, 0.083] |
| coord | B | −0.027 [−0.057, 0.003] | +0.063 [0.031, 0.093] | −0.199 [−0.228, −0.171] |
| pca | R | +1.30 [1.24, 1.35] | +0.159 [0.131, 0.187] | +0.593 [0.555, 0.633] |
| wsvd | R | +1.29 [1.24, 1.35] | +0.159 [0.132, 0.188] | +0.593 [0.555, 0.632] |
| pca / wsvd | B | −1.24 | −1.20 / −1.18: **all 192 pairs oos** (scale 10–38) | −1.22 / −1.18: **all oos** |

Cluster CIs are in the tables file.

### 3c. Donor controls (paired: basis natural native − control matched to basis natural; k=192)

| basis | same_year control T (l2/preact) | natural − same_year (l2) | wrong_donor control T (l2) | natural − wrong (l2) |
|---|---|---|---|---|
| svd R | −0.005 / −0.004 | 0.433 [0.385, 0.479] | 0.162 | 0.267 [0.220, 0.311] |
| svd B | 0.058 / 0.052 | 1.18 [1.11, 1.26] | 0.501 | 0.739 [0.653, 0.831] |
| pca R | 0.053 / 0.035 | 1.67 [1.58, 1.76] | 0.715 | 1.01 [0.887, 1.13] |
| RB (any) | 0.040 / 0.027 | 1.69 [1.59, 1.77] | 0.712 | 1.01 [0.892, 1.14] |

* Matching the same-year control up to the natural norm needs scale ~2.3 (median), so 3–34% of those pairs
  are oos.
* Wrong-donor T stays 0.5–0.7 because its year shift partly aligns with `v_rd`. This is **not** a zero null and
  **not** evidence of content specificity.

### 3d. Reduced models: unperturbed quality and natural-intervention fidelity (k=192)

| scope | basis | method | KL | mean e_R | RMS e_R | mean e_RB | RMS e_RB | local err ratio (R) |
|---|---|---|---|---|---|---|---|---|
| LT | svd | zero / bias / gauss | 0.171 / 0.155 / 0.149 | +0.070 / +0.092 / +0.057 | 0.137 / 0.131 / 0.107 | −1.23 / −1.20 / −1.24 | 1.29 / 1.27 / 1.30 | 0.53 / 0.34 / 0.33 |
| LT | coord | zero / bias / gauss | 0.155 / 0.170 / 0.164 | −0.023 / +0.090 / +0.058 | 0.121 / 0.135 / 0.115 | −1.31 / −1.20 / −1.23 | 1.39 / 1.26 / 1.30 | 0.53 / 0.37 / 0.35 |
| LT | random ×5 | zero / bias / gauss | 0.205 / 0.171 / 0.165 | −0.167 / +0.099 / +0.070 | 0.253 / 0.147 / 0.128 | −1.46 / −1.20 / −1.23 | 1.53 / 1.26 / 1.29 | 0.89 / 0.36 / 0.35 |
| LT | pca | zero / bias / gauss | 0.038 / 3.41e-4 / 3.41e-4 | +0.147 / −0.003 / −0.003 | 0.196 / 0.018 / 0.018 | +0.146 / −0.003 / −0.003 | 0.196 / 0.021 / 0.021 | 0.34 / 0.03 / 0.03 |
| LT | wsvd | zero / bias / gauss | 0.056 / 2.98e-4 / 2.98e-4 | +0.025 / −4.4e-4 / −4.5e-4 | 0.177 / 0.018 / 0.018 | +0.023 / −0.002 / −0.002 | 0.180 / 0.021 / 0.021 | 0.24 / 0.03 / 0.03 |
| FL | svd | zero / bias / gauss | 0.207 / 0.150 / 0.141 | +0.050 / +0.082 / +0.075 | 0.124 / 0.134 / 0.130 | −1.25 / −1.21 / −1.22 | 1.31 / 1.28 / 1.29 | 0.53 / 0.48 / 0.47 |
| FL | random ×5 | zero / bias / gauss | 0.223 / 0.240 / 0.200 | −0.165 / −0.237 / −0.216 | 0.253 / 0.289 / 0.264 | −1.46 / −1.53 / −1.51 | 1.53 / 1.60 / 1.58 | 0.89 / 0.72 / 0.68 |
| FL | pca | zero / bias / gauss | 0.065 / 0.043 / 0.043 | +0.167 / −0.189 / −0.187 | 0.217 / 0.261 / 0.243 | +0.167 / −0.190 / −0.187 | 0.217 / 0.262 / 0.244 | 0.34 / 0.39 / 0.37 |
| FL | wsvd | zero / bias / gauss | 0.078 / 0.045 / 0.052 | −0.045 / +0.021 / +0.032 | 0.185 / 0.148 / 0.133 | −0.047 / +0.019 / +0.030 | 0.188 / 0.149 / 0.133 | 0.24 / 0.20 / 0.20 |

* All ranks, `gauss_wo`, and the random per-seed values are in the tables file.
* Random seed spread at LT k=192, RMS e_R: bias 0.147 [0.132, 0.157]; gauss 0.128 [0.112, 0.139].
* **Important caveat for PCA/wsvd in LT:** the discarded σ on the final-token moment pool is only ~0.01 (svd: up
  to 1.03). The moment pool is the same task (other nouns), and its final-token inputs are nearly confined to a
  low-dimensional subspace (PCA k=96 explains 99.8% of the variance). So B is almost constant across these
  prompts: σ is small, and the measured bias error is small. LT gauss−bias differences are of order 1e-6, which is below what we can
  resolve numerically here. They show **no added benefit and no formal equivalence**. This reflects how close
  the pool is to the evaluation prompts; it is **not** evidence of general compression quality or of a general
  mechanism.
* In `full_layer` scope, PCA/wsvd do worse than in LT (bias/gauss KL 0.01–0.13, RMS e_RB 0.09–0.31); those bases
  were fitted on final-token inputs only.
* `full_layer` Gaussian vs bias is mixed. In some cells `gauss` lowers the RB error (e.g. wsvd k=192: RMS e_RB
  0.133 vs 0.149), while KL gets worse in the same cell (0.052 vs 0.045). This is neither a universal Gaussian
  failure nor a universal benefit. Paired gauss−bias cluster CIs are in `notes/codex_day2_independent_audit.json`.

### 3e. Narrow reading for the Day-2 question (exploratory; for review)

* **Dimension.** For the W_in SVD cut, Haar-random subspaces (5 seeds) and an `E[x²]` coordinate mask of the same
  dimension give **similar point values** for:
  * the R/B split of the natural year response (k=192 T_R: svd 0.429, random 0.401–0.467, coord 0.440);
  * the size of the B loss in reduced models;
  * the positive R error of `bias` in LT;
  * the lower R error of `gauss` vs `bias` in LT.

  At native size, the paired SVD-vs-random/coord differences are small (k=192 random_s101 R: −0.017
  [−0.047, 0.015]). This is **not** a formal equivalence (no margin, no equivalence test). It is consistent with
  a dimension effect, and we saw no sign that the year transfer is concentrated in W_in's top singular
  directions: the SVD R part even carries a *smaller* share of ‖δ‖ than a random subspace of equal dimension.
* **Intervention size.** Under norm matching, SVD-vs-random differences become nonzero, and their **sign depends
  on the regime** (l2 vs pre-activation energy; see 3b). Cross-basis R and B contrasts compare **different
  physical directions**; only RB is a common physical target. So there is no single per-unit-size ordering, and
  no SVD-specific claim follows.
* **Activation geometry.** For PCA and whitened SVD fitted on the same-task pool:
  * most of the natural δ norm lies in R (k=192 PCA: RMS‖δ_R‖ 3.37 vs RMS‖δ_B‖ 0.14);
  * T_R ≈ T_RB, and mean T_B and mean I are ≈ 0;
  * per-pair T_B is not zero (RMS ≈ 0.012–0.013, max 0.035–0.042).

  Matched to the SVD-R norm, they still give more T (l2 +0.16, preact +0.59). Their small LT error goes with
  σ_B ≈ 0.01 on that pool (caveat above) and is not shown to generalise.
* Day-1 statements are kept: RB loss is real for the SVD cut; RB alone does not rank R fidelity; no universal
  Gaussian harm or benefit; no mechanism established.

## 4. What did not pass or is limited

* No gate or test failed. The cross-basis **B** contrasts for PCA/wsvd under matching have **0/192 pairs in
  support** (all oos, scale 10–38), so they are not interpretable.
* Matched regimes were run without an interaction decomposition (`I` is native-only).
* The oos filter is a diagnostic; passing it is **not** evidence that a matched intervention stays on the
  activation manifold.
* Many contrasts with no simultaneous-inference correction. Same-year matched controls have 3–34% oos per cell (with and
  without-oos values are both reported).
* The basis fit, the moments and the recipients all come from one task distribution (GT prompts, 2 frames).
  This favours activation-geometry bases. Testing on the new validation/test frames (F2/F3, F4/F5) with the
  calibration basis/moments frozen is the agreed **minimum held-out test for the narrow claim**. A general
  in-domain or general-text pool is a separate, new research question; it is **not** a hidden prerequisite of
  this 3-day pilot.
* The centred pool covariance has algebraic rank ≤ 575, so k=384 does not exceed it. Relative to the ridge
  threshold λ = 8.133134e-5 (the fixed rule), 632 of 768 covariance eigenvalues lie below λ and only 136 above.
  The regularisation is therefore strong in those low-variance directions. This is a λ-threshold count, not a
  numerical rank.
* Scope/moment-pool confound unchanged; scopes reported separately.
* Not done (by design): norm-matched regimes in reduced models; resample ablation; validation; test;
  confirmatory claims.

## 5. Day-3 order (agreed in review; nothing executed here)

1. **Before opening validation**, freeze:
   * the primary scope;
   * the candidate analyses;
   * the selection rules:
     * SVD + PCA as the two main candidates. PCA was chosen **after** seeing calibration results (a recorded,
       data-driven choice).
     * wSVD as a secondary sensitivity analysis (oblique geometry).
     * The **5 pre-fixed random seeds (101, 202, 303, 404, 505) kept as the dimension control**: not a third
       main candidate family, and no seed selection.
     * The quality-matching rule, oos handling, and the resample definition (local `E[h]` vs downstream `E[T]`).
2. **Quality matching** (GPT Pro): at the existing three ranks and the same scope, compare validation KL under a KL rule locked in advance;
   comparisons across the existing ranks are allowed. If no common KL range exists, label the comparison
   **same-rank**, and do not force one by widening ranks or scopes.
3. **ε**: justify it on the scale of the original RB response, not so that PCA passes. Freeze the numeric ε and
   the confirmatory rank from calibration/validation **before test** (PROTOCOL §9). Antigravity's ε = 0.09 and
   15% oos were example proposals only; they are not adopted thresholds.
4. **Resample is not a prerequisite** for validation. If wanted, run it only as a diagnostic: PCA, k=192,
   full_layer, with an independent, scope-consistent calibration pool and Monte-Carlo uncertainty. A difference
   alone does not show content loss or a failed Gaussian assumption.
5. Validation = frames F2/F3 and test = frames F4/F5 (new frames). They have not been looked at and must not be
   until the freezes above are in place. Bases and moments stay fixed from the calibration pool (no refit on
   validation).
6. If held-out data only repeat a known data-sensitive reconstruction, an MI-novelty claim is an explicit
   **NO-GO**.

## 6. Claude's short independent verdict

**Conditional GO for Day 3**, on the narrow reading only:
1. The Day-1 SVD-cut phenomena (large B share, RB loss, bias R over-response, gauss < bias in LT R error) have
   point values similar to equal-dimension random and coordinate cuts. This is consistent with dimension rather
   than SVD alignment; no formal equivalence is shown.
2. The activation-geometry result (PCA/wsvd) is strong at face value but confounded by the same-task pool
   (σ_B ≈ 0.01). Validation on new frames with the calibration-pool basis/moments frozen is the minimum held-out
   test.
3. Gauss vs bias stays basis- and scope-dependent, with no universal direction.

**Objection to any stronger reading.** No SVD-specificity, no mechanism, no compression claim, and no matched
"equal quality". Day 3 must freeze its rules before opening validation/test. Day-3 held-out and ε-freeze work
was **not** run.

## 7. Commands

```bash
cd /Users/emre/Documents/Codex/ML_Projects/spectral_causal_pilot
.venv/bin/python -m pytest -v                       # 49 tests
.venv/bin/python run_day2.py quadcheck              # sigma range + node rule (already fixed in config)
.venv/bin/python run_day2.py smoke                  # failure-gated Day-1 replay and sanity
.venv/bin/python run_day2.py run                    # Day-2 measurement (15-min hard budget)
.venv/bin/python -m scp.report_day2 runs/<day2_run_id>
```

## 8. Artifacts

| what | path |
|---|---|
| protocol / config / registration | `PROTOCOL_DAY2.md`, `config/day2_v1.json`, `registration/DAY2_REGISTRATION.json` |
| code / tests | `scp/bases.py`, `scp/matching.py`, `scp/day2.py`, `scp/report_day2.py`, `run_day2.py`, `tests/test_day2.py` |
| test log | `logs/pytest_day2_20261030T160558Z.log` |
| quadcheck | `runs/20261030T160333Z_day2_quadcheck/quadcheck.json` |
| smoke | `runs/20261030T160634Z_day2_smoke/smoke.json` |
| run | `runs/20261030T160653Z_day2_run/{results.json, summary_tables.md, control_rows.csv, reduced_rows.csv, unperturbed_rows.csv, partial_results.json, run_manifest.json, log.txt, snapshot/}` |
| run stdout | `logs/day2_run_stdout_20261030T160652Z.log` |
| independent audit (Codex) | `notes/codex_day2_independent_audit.json`. **Re-derived from the raw CSVs and matching:** all reported point statistics; common masks; local/logit/delta-norm RMS; scale quantiles; alignment; interaction; seed-spread points. Local/logit error ratios differ by ≤ 1.8e-7 (f32 aggregation). Selected gauss−bias paired cluster CIs were recomputed. **Not covered:** full-vector mean-error norms (they cannot be rebuilt from the scalar raw rows); the full CI grid was not re-run. |
| Claude verdict note | `notes/claude_day2_verdict.md` |
| Day-2 review (Codex, GPT Pro, Antigravity; final) | `notes/three_model_review_day2.md` |
