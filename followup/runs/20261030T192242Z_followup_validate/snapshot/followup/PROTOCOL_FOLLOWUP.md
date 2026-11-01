# Follow-up protocol: limited propagation audit (followup-v1, DRAFT until registration)

Decision record: `FOLLOWUP_DESIGN.md` (final, accepted by GPT Pro, Fable Extra and Antigravity) and
`notes/followup_debate.md`. This file fixes the implementation details. The config `followup/config/followup_v1.json`
is the machine-readable counterpart. Where the two differ, the stricter reading applies and the difference is a
blocker to report, not to resolve silently.

This is a limited propagation audit:
* not circuit discovery;
* not a mechanism, novelty or compression claim;
* not a WeightWatcher/RG test;
* input-only prediction is not tested.

## 1. Fixed setting

* **Model:** GPT-2 small at revision `607a30d7…`, MPS float32, eval mode, all parameters frozen, offline project
  cache.
* **Site:** block 9, MLP input after ln_2, final real token only. Residual bypass and all other positions are
  untouched.
* **Reduction:** the frozen Day-3 artifact `calibration_day3/calib_artifact.pt` (sha256 `cd56bb2f…`), loaded and
  verified by `scp.day3.load_artifact` with `config/day3_v1.json`:
  * `P|pca|192` (orthogonal column projector P);
  * `mu|pca|192|final_token` (frozen discarded-preactivation mean μ);
  * method `bias`, scope `last_token`.

  Nothing is refitted.
* **Local MLP algebra:** CPU float64 copies of W_in (`c_fc`), b_fc and W_out (`c_proj`).
* **Old code:** the frozen Day-1..3 code (`scp/`) is imported read-only. Its source tree hash is bound into the
  follow-up registration as a dependency (`scp_dependency_source`). None of the old source, config, registration,
  manifest or artifact files is modified. The follow-up globs (`followup/*.py`, `run_followup.py`,
  `tests_followup/*.py`) are disjoint from the Day-3 `SOURCE_GLOBS`, so the Day-3 tree hash is unchanged.

## 2. Data (fresh, outcome-blind)

**Frames.** Ten new frames N0..N9 (config). Each has the form
`The {noun} … the year 17{yy} … the year 17` and differs from F0..F5. A seeded permutation (seed 20261101) assigns
4 to fit, 2 to validation and 4 to test.

**Nouns.**
* 66 new nouns, disjoint from the 36 historical Day-1..3 nouns.
* Each template (frame, noun) gets its **own** noun. No noun is shared between templates within or across splits,
  so template clusters do not share nouns.
* Templates per frame: fit 4, validation 4, test 8. That gives 16 / 8 / 32 templates; 32 ≥ 24 test clusters.

**YY pools.** Years 02..98 in consecutive triples; one year of each triple goes to each split (seeded). The pools
are disjoint across splits. There is no claim that year values were never seen before.

**Recipients and donors.**
* For each template there is exactly one recipient per design cell: direction (up/down) × gap bin (10–19, 20–29,
  30–40). That gives fit 96, validation 48 and test 192 recipients.
* Recipient years are distinct within a template.
* The donor is the same template at year d, with d − r (up) or r − d (down) in the bin, and d in the split pool.
* Donor prompts may be reused inside their template cluster and never cross clusters.
* (r, d) is drawn uniformly from the pairs whose two prompts pass the tokenizer check (`scp.data.check_prompt`:
  ` 17`+`YY` start slot; final token ` 17`). Exclusions are for tokenization only, and there were 0.

**Leakage.** No follow-up prompt (recipient or donor) may equal any exact prompt text in `manifests/*.csv`
(moment pool, calibration pairs with all control roles, sealed validation/test prompts) or `manifests_day3/*.csv`.
The prepare stage checked 2178 historical texts and found 0 overlaps. Frames, nouns, YY pools and prompts are also
disjoint across the follow-up splits.

**Outcome-blindness.** Frames were not selected or edited on the basis of model performance. Prepare loads only the
tokenizer. Outputs are `followup/manifests/{fit,validation,test}_recipients.csv` and `index.json`, with hashes,
written once.

## 3. Intervention, labels, scale

* x' = x_r + α P(x_a − x_r), α ∈ {0.5, 1, 1.5}, at the block-9 MLP input, final token. One family is used in all
  splits; there is no mixed donor. α = 1.5 is a registered extrapolation/stress condition, not a natural
  activation.
* orig(·) and red(·) are the final-position full-vocabulary logits of the original model and of the PCA192+bias
  last-token reduced model. base means unperturbed (x = x_r) and ' means perturbed (x').
* **Label:** Y = ‖centred[(red' − red_base) − (orig' − orig_base)]‖ / s. Centred means the mean over the 50,257
  logits is removed, and the norm is computed in CPU float64.
* **Scale:** s = sqrt(mean over **fit** recipients at α = 1 of ‖centred(orig' − orig_base)‖²). It is computed in
  fit only (the code refuses other splits) and frozen. If s is non-finite or ≤ 1e-12, the evaluation is invalid.

## 4. Local and downstream quantities (all features from unperturbed forwards)

Row form, with R = Pᵀ, Q = (I − P)ᵀ, z = x W_in + b_fc, and û = (xR)W_in + b_fc + μ:
* q = z − û = (xQ)W_in − μ;
* b(x) = (gelu_new(û) − gelu_new(z)) W_out, i.e. h_red − h_orig (the c_proj bias cancels);
* δ = α(x_a − x_r)R; δb = b(x_r + δ) − b(x_r);
* a_R = (δR)W_in; a_B = (δQ)W_in ≡ 0 (checked);
* D1 = [g'(û + a_R) − g'(û)] ⊙ q W_out + g'(û + a_R) ⊙ a_B W_out, with g' the analytic gelu_new derivative.
  D1 ≈ −δb to first order in q.

**Input reconstruction norm.** ‖x_r − x̂_r‖ = ‖x_r Q − m_B‖, with x̂ = xR + m_B. Here m_B = m(I−P) is the discarded
part of the **frozen** calibration final-token activation mean m.
* m_B is recovered from the frozen μ = m_B W_in by least squares. W_in has full row rank, so the solution is
  unique; the relative residual must be ≤ 1e-8 and is recorded.
* It is unchanged by the R-only intervention.

**Downstream sensitivities.** J is the Jacobian of the centred final-position logits with respect to the block-9
MLP output at the final token (the leaf), evaluated at the **original unperturbed** recipient.
* The leaf is the only tensor that requires grad. The model is frozen and in eval mode; no donor, feature or
  upstream tensor is in the graph.
* Gradients are enabled explicitly (`torch.enable_grad`). The leaf replaces only the MLP output, which the block
  adds to its unchanged residual.
* Eight fixed Rademacher probes u_j ∈ {±1}^50257 come from CPU generator seed 20261102 and are identical for every
  recipient and stage; their sha256 is recorded.
* g_j = Jᵀu_j is computed on MPS float32 in batches of 16 and cached per recipient in CPU float64.

**Features** (s from fit):

| feature | definition | role |
|---|---|---|
| alpha | α | B |
| disc_input_norm | ‖x_r Q − m_B‖ | B |
| delta_norm | ‖δ‖ | B |
| preact_delta_norm | ‖a_R‖ | B |
| disc_preact_norm | ‖q‖ | B |
| preact_norm_product | ‖q‖·‖a_R‖ | B |
| b_rec_norm, b_prime_norm | ‖b(x_r)‖, ‖b(x')‖ | B |
| delta_b_norm | ‖δb‖ | B |
| kl_unpert | KL(orig_base ‖ red_base), full vocabulary | B |
| jnorm | sqrt(mean_j ‖g_j‖²) | B |
| b_iso | jnorm·‖δb‖/(s√768) | B |
| dcurv_norm | ‖D1‖ | B |
| d_shuf | sqrt(mean_j (g_j·δb[π])²)/s, π a fixed permutation of the 768 coordinates (seed 20261103) | B |
| d_exact | sqrt(mean_j (g_j·δb)²)/s | A only |
| d_taylor | sqrt(mean_j (g_j·D1)²)/s | A_in only |

Descriptive only, never predictors:
* d_exact on probes {0..3} and {4..7}; the 4+4 reliability is reported on fit;
* max |a_B|.

**Models.**
* B = 14 baseline features.
* A = B + d_exact (primary).
* A_in = B + d_taylor (secondary, descriptive). A_in is not input-only, because B contains exact local errors.

## 5. Fitting and validation

* **Ridge.** Standardized per model on fit rows only (fit mean, population std; a zero std is an error). The
  intercept is unpenalized. λ ∈ {0.1, 1, 10} is fitted on fit, and predictions are clamped at 0.
* **λ selection.** Separately for A, B and A_in, by the validation fixed-weight MSE (§6 weights). Ties within a
  relative 1e-12 go to the larger λ.
* **Fit-mean predictor:** the fit mean of Y (clamped at 0).
* **Validation gate.** If neither chosen primary model (A or B) has a lower validation MSE than the fit-mean
  predictor, it is a development NO-GO. Freeze records `proceed_to_test = false`, and predict-test refuses.
* **Freeze.** The freeze is write-once. It binds s, the selected λ and coefficients, the standardization, the
  fit-mean, the gate result, the fit/validation runs with their output hashes, the registration hash and the
  source tree.

## 6. Test evaluation

**Primary statistic.** On the test split (192 recipients × 3 α = 576 rows):
G = 1 − MSE(A)/MSE(B).
* MSE is a fixed-weight mean of per-cell mean squared errors over the cells frame × direction × gap bin × α, with
  uniform weights (4·2·3·3 = 72 cells, each 1/72).
* The design is complete. Every template has exactly one recipient per (direction, bin) and all three α, and the
  code refuses incomplete designs.

**Bootstrap.** Paired template-within-frame cluster bootstrap with B = 10,000 (seed 20261104).
* In each frame its 8 templates are resampled with replacement. All rows of a template (6 recipients × 3 α, with
  their donors) travel together.
* Cell means are recomputed from the resampled rows, with the same multiplicities for A and B.

**Decision.**
* **GO** if the one-sided 95% lower bound (5% quantile) is > 0.10. This is an operational predictive-gain
  threshold.
* **TARGET_GAIN_RULED_OUT** if the one-sided 95% upper bound (95% quantile) is ≤ 0.10, i.e. the target gain is
  ruled out on this cohort.
* **INCONCLUSIVE** otherwise.
* None of these means a universal absence of effect, falsifies WW/RG, or identifies a mechanism.

**Reporting.**
* G is always reported, including zero or negative values.
* MSE(A), MSE(B) and MSE(B) − MSE(A) (with a two-sided 95% interval) are always reported.

**Invalid evaluation.** The MSE(B) floor is 1e-12, fixed.
* If the point MSE(B), or MSE(B) in **any** bootstrap draw, is non-finite or ≤ the floor, the ratio interval is
  invalid. No draw is discarded.
* The decision is then **EVALUATION_INVALID** (inconclusive), with the raw MSEs and their difference reported.

**Secondary (descriptive).** G_Taylor = 1 − MSE(A_in)/MSE(B), with the same bootstrap; negatives are retained. No
G_Taylor/G ratio is computed.

**Other descriptive outputs:** per-α and per-frame MSE and point G, fit-mean MSE.

**Interval caveats.** Intervals are nominal, approximate and conditional on the eight fixed probes, the frozen
artifact and these frames. Probe uncertainty is not covered by the template bootstrap.

## 7. Stages, locks and order

`run_followup.py`, one stage at a time:

1. `prepare`: outcome-blind manifests (write-once).
2. `smoke`: technical checks on the OLD Day-1 calibration pairs only (32 seeded pairs). See §8.
3. `register` (after the Codex audit): write-once `followup/registration/FOLLOWUP_REGISTRATION.json`. It binds:
   * the protocol, `FOLLOWUP_DESIGN.md`, the config, the follow-up source tree, the scp dependency tree, the
     manifests, the Day-3 config hash and the artifact hash;
   * the smoke run, which must be ok with all gates passed, the same source/dependency/manifests/design, and a
     config identical up to the `protocol_version` label;
   * the test log.

   The DRAFT label must be finalised first.
4. `fit`: fit outcomes, s, ridge for all λ.
5. `validate --fit-run`: validation outcomes, λ selection, gate.
6. `freeze --fit-run --val-run`: write-once `FOLLOWUP_FROZEN.json`.
7. `predict-test`:
   * runs test unperturbed forwards and VJPs only. The label path raises `LabelLock`.
   * computes test features and predictions;
   * writes `test_features.csv`, `test_predictions.csv` and `test_probe_vjps.pt`;
   * writes the write-once `FOLLOWUP_TEST_PREDICTIONS_LOCK.json` with their sha256, **before any perturbed test
     forward exists**.
8. `evaluate-test`:
   * verifies the registration, the freeze and the lock hashes;
   * re-runs the unperturbed test forwards and requires the locked KL feature to be reproduced (relative error
     ≤ 1e-5);
   * then computes the test labels and the results;
   * writes the write-once `FOLLOWUP_TEST_EVALUATED.json`.
9. `report --eval-run`: renders the results into a new report run.

**Run discipline.**
* Every stage after register re-verifies all bound hashes.
* Every run directory (`followup/runs/<UTC>_followup_<stage>/`) is new. It holds a manifest with bound hashes,
  environment, a source/config/manifest snapshot, a log, raw CSV outputs, output sha256 values and timestamps.
* Failed runs are kept.
* There is no predictor search, scope expansion or probe increase after any outcome.

## 8. Technical smoke (old calibration only; gates fixed before the smoke)

The smoke uses 32 Day-1 calibration pairs (rec/natural donor) and is never run on the follow-up splits. Gates
(config `smoke.gates`):

| check | gate |
|---|---|
| leaf-forward logits vs original logits | max abs ≤ 1e-4 |
| FD of probe VJPs, MPS float32, steps 0.5 and 1.0, random-unit and δb directions | max rel ≤ 2e-2 |
| FD in a CPU float64 model copy, steps 1e-4 and 2e-4 | max rel ≤ 1e-6 |
| MPS float32 vs CPU float64 probe VJPs | rel ≤ 1e-3 |
| local float64 b vs model output difference, unperturbed and α = 1 | rel ≤ 1e-4 |
| local δb vs model, α = 1 | rel ≤ 1e-3 (the primary feature) |
| self-patch (δ = 0) logits, original and reduced | exactly 0 |
| hidden states at positions before the last, under intervention | exactly unchanged |
| α = 0 local δb | exactly 0 |
| same-shape rerun of the VJPs | bitwise identical |

**Descriptive, not gates:**
* a cross-batch-shape VJP difference (batch 16 vs 8), against the pre-set tolerance rel ≤ 1e-4; it isolates MPS
  roundoff from real nondeterminism;
* the exact-JVP (double-backward) norm vs the 8-probe D on 4 recipients. This is technical only and never changes
  the registered estimator;
* old-CAL correlations of the features with Y;
* the 4+4 probe-half reliability;
* timings, the runtime projection for 336 recipients (2688 VJPs, 1008 interventions), and peak memory.

## 9. Tests

`tests_followup/` checks, without any follow-up model outcome (tiny random GPT-2, CPU float64):
* numerics:
  * analytic gelu_new derivative vs FD;
  * D1 sign and O(q²) error;
  * local b vs the scp `Reduction` module;
  * retained-only geometry and m_B recovery;
  * zero/self intervention;
* downstream estimator:
  * synthetic linear downstream (predicted vs true norm);
  * VJPs vs the full Jacobian, exact JVP and two-step FD;
  * fixed probe reuse;
  * eval/frozen/no-nesting guards;
* hook scope: last token only, residual bypass untouched;
* features:
  * feature definitions;
  * centred labels;
  * s from fit only;
  * label lock;
* data:
  * manifests (structure, disjointness, leakage, donors inside clusters);
  * leakage detection;
* fitting and evaluation:
  * ridge and λ ties;
  * cluster-bootstrap weights and rows travelling together (vs brute force);
  * negative G and the decisions;
  * the denominator floor with no draw discarded;
  * incomplete-design rejection;
* CLI and locks: stage arguments, missing registration, bound-change detection, write-once, the lock-before-labels
  order;
* the unchanged Day-3 source tree.

The old 66 tests are run once as a regression.
