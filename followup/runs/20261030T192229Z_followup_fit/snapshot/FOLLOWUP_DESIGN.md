# Follow-up decision and success criteria — before implementation/outcomes

2026-10-30. The app debate is recorded in `notes/followup_debate.md`. Its earlier mixed-donor proposal is kept
there as history only and is **obsolete**.

* GPT Pro proposed the propagation audit. In its final cross-review it adopted Antigravity's retained-only (R-only)
  objection and narrowed its own proposal.
* The coordinator's final choice follows that narrowing.
* **Final acceptance by all three reviewers** (`notes/followup_debate.md`):
  * GPT Pro accepted the limited audit and withdrew its reporting objection after the clarification below.
  * Antigravity accepted the final R-only design. Its phrase "all methodological issues closed" is not adopted.
  * Fable Extra accepted GO, labelled "limited propagation audit; input-only prediction not tested". It keeps its
    preference for an input-derived primary.
  * This is agreement to execute, not validation of the implementation or of any outcome.
* No follow-up outcomes have been generated.

Implementation lives in `followup/` (protocol: `followup/PROTOCOL_FOLLOWUP.md`; config: `followup/config/`).

## Fixed question and scope

**Question.** Do unperturbed downstream directional sensitivities improve the prediction of the centred
full-vocabulary intervention-response error? The comparison is against a baseline of:
* exact local reconstruction errors;
* a local curvature norm;
* scalar/isotropic sensitivity;
* a shuffled-alignment control;
* intervention size;
* unperturbed KL.

**Scope.** This is a limited propagation audit. It is not circuit discovery, new mathematics, deployed compression,
or a WeightWatcher/Wilson RG test.

**Fixed setting.**
* GPT-2 small, block 9 MLP input after ln_2, last token only.
* Frozen Day-3 PCA k=192 + bias artifact (`calibration_day3/calib_artifact.pt`, last-token scope, final-token
  moments).
* The recipient residual bypass and all other positions are unchanged.
* No SAE, general-text PCA, or new rank/layer/model search.

**Intervention (retained-only, one family for fit/validation/test):**

x' = x_r + α P (x_a − x_r), α ∈ {0.5, 1, 1.5}, with P the PCA192 projector.

* The donor x_a comes from the same template (frame + noun) at a different year. There is **no mixed donor**.
* α = 1.5 is an extrapolation along the donor direction, not a natural activation.

**Splits.**
* Fresh 96 fit / 48 validation / 192 test recipients, one donor each, in 4/2/4 new frames.
* At least 24 distinct template clusters in test.
* Total: 336 × 3 = 1008 interventions and 336 × 8 = 2688 VJPs.
* Donors stay within their template cluster.
* Prompt/noun pools and YY pools are disjoint across the new splits, and all historical exact prompts are excluded.
  There is no claim that year values were never seen before.
* Selection and tokenization are outcome-blind.

## Quantities

* b(x) = h_reduced(x) − h_original(x) (local MLP output, last token); δb = b(x') − b(x_r).
* J is the downstream centred-logit Jacobian with respect to the local MLP output, taken at the **original
  unperturbed** recipient only.
* Eight fixed seeded Rademacher probes u_j over the full vocabulary give g_j = Jᵀ u_j (same probes for every
  recipient).
* D = sqrt(mean_j (g_j · δb)²) / s.
* s = RMS over **fit** recipients at α = 1 of the original-model R-response vector
  ‖centred(orig(x') − orig(x_r))‖ over the full vocabulary. It is frozen after fit.
* Target Y = ‖centred[(red' − red_base) − (orig' − orig_base)]‖ / s.

## Predictors

**Strong baseline B:**
* input reconstruction norm ‖x_r − x̂_r‖ = ‖(x_r − m)(I − P)‖. Here x̂ = x P + m (I − P) is the input the reduced
  layer effectively sees (P W_in + μ ⇔ x̂ W_in), and m is the **frozen** calibration final-token activation mean.
  * Only m(I − P) enters. It is recovered exactly from the frozen μ = m(I − P) W_in (W_in has full row rank), with
    no new forward pass.
  * The R-only intervention leaves it unchanged at x';
* local b endpoint norms and ‖δb‖;
* intervention and preactivation norms;
* discarded-preactivation norm ‖q‖ (q = x(I−P)W_in − μ, frozen μ);
* **preactivation norm product** ‖q‖ · ‖a_R‖ (a_R = δ P W_in), as an explicit feature. Separate norms in a linear
  ridge do not represent the product;
* α;
* the recipient's unperturbed original-vs-reduced KL;
* Jnorm = sqrt(mean_j ‖g_j‖²);
* B_iso = Jnorm · ‖δb‖ / (s √768);
* **Dcurv local norm** ‖D1‖;
* **shuffled D**: D computed with δb's residual coordinates permuted by one fixed seeded permutation.

**Models.**
* **Primary** A = B + D (exact local δb).
* **Secondary** A_in = B + D_Taylor, with D_Taylor = sqrt(mean_j (g_j · D1)²) / s. The D_Taylor feature is
  input-derived, but the secondary pipeline is not input-only, because B contains exact local errors.

**First-order local Taylor vector** (orig-minus-red sign), retained-only, so a_B = 0:
* û = x P W_in + b + μ_discarded;
* q = x (I−P) W_in − μ_discarded;
* a_R = δ P W_in;
* D1 = [g'(û + a_R) − g'(û)] ⊙ q · W_out (+ g'(û + a_R) ⊙ a_B · W_out, which is 0 by construction).

**Fitting.** Standardized ridge, fitted on fit rows only.
* λ ∈ {0.1, 1, 10} is selected separately per model by validation MSE; ties go to the larger λ.
* Predictions are clamped at zero.
* Scaling and feature definitions are frozen before test.

## Primary criterion

**Statistic.** G = 1 − MSE(A) / MSE(B) on test (all three α).
* Fixed uniform cell weights over frame × direction × gap bin × α.
* Paired template-within-frame cluster bootstrap, B = 10,000. All recipient/α rows and within-cluster donors travel
  together.

**Decision.**
* **GO** only if the one-sided 95% lower bound is > 0.10. This is an operational predictive-gain threshold.
* A one-sided 95% upper bound ≤ 0.10 rules out this target gain at the stated confidence.
* Otherwise the result is inconclusive.
* Neither outcome means a universal absence of effect.

**Reporting.**
* G is **always reported**, including zero or negative values. Positivity of G is not a precondition.
* MSE(A), MSE(B) and their difference are always reported.

**Invalid evaluation.** The MSE(B) denominator has a fixed numeric floor of 1e-12 (Y is dimensionless).
* The ratio CI is invalid if the point MSE(B), or MSE(B) in any bootstrap draw, is ≤ the floor or non-finite. No
  draw is discarded silently.
* In that case MSE(A), MSE(B) and their difference are reported, and the primary decision is
  **EVALUATION_INVALID** (inconclusive).
* An s ≤ floor is likewise invalid. An invalid evaluation is never a success.

**Secondary.** The secondary G_Taylor = 1 − MSE(A_in)/MSE(B) is descriptive, with negatives retained. No ratio of
G_Taylor to G is computed.

**Validation gate.** If neither chosen validation model (A, B) beats the frozen fit-mean predictor on validation MSE,
stop before test.

**Constraints.**
* No post-result predictor search, scope expansion or probe increase.
* The result is conditional on the eight fixed probes.

## Numerics and controls

* Analytic gelu_new derivative check.
* Zero/self intervention.
* Linear-downstream synthetic case.
* Projection/scope correctness.
* Gradient finite-difference sanity, on old calibration only.
* 4+4 probe-split reliability, descriptive, on fit.
* The optional exact-JVP smoke is technical only. It is never a test-adaptive replacement or a probe increase.

## Governance

* Implementation by Claude Code Opus 5.5 High, in separate `followup/` files.
* The frozen Day-1..3 source, configs and artifacts must not change.
* New protocol/config/source/manifests get timestamped hashes (registration) before any fit/validation outcome
  measurement.
* The ridge selection is frozen after validation.
* Test features and predictions are serialized and hashed **before** any perturbed test forward (the labels) is
  computed.
* Tests, a small old-calibration numerical smoke and a runtime benchmark precede registration. There is no peeking
  at new test outcomes.
