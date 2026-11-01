# Day-3 post-hoc diagnostic from existing records (2026-10-30)

**Post-hoc, descriptive only.**
* No new model run, no new criterion, no pass/fail claim.
* It does not change the registered S2 (which passed) and is not confirmatory.
* It is arithmetic on the existing per-pair rows of `runs/20261030T164723Z_day3_test` and
  `runs/20261030T164225Z_day3_validate` (`reduced_rows.csv`, `control_rows.csv`), recomputed by Claude with a
  standalone script that reads only those CSVs.
* The values agree to all printed digits with the numbers Codex/the user read from `results.json`.
* Any balancing it shows is arithmetic error-offsetting, **not** a network mechanism.
* **Provenance:** GPT Pro asked this question in its final review. GPT Pro had **not** seen these numbers when it
  wrote that review: its package did not include the test decomposition or the local-vector figures.
  * The decomposition below answers only GPT Pro's point that the *held-out* arithmetic error-offset had not
    been shown.
  * GPT Pro's more basic objection to network-mechanism claims (scalar underidentification) is **not** resolved
    by these numbers.

**Record distinctions:**
* The R/B/RB errors and the local/logit vector ratios were already part of the **pre-planned descriptive grid**
  (`reduced_rows.csv`, `results.json`).
* Only the focus on them after GPT Pro's review, and any `max{…} ≤ ε` criterion, are **post-hoc**.
* GPT Pro did not see these extra numbers. Its approval of the joint sentence must not be read as covering them.
* Reduced-model Δ_B = 0 is structural and is **not** evidence of success. The original B RMS is small but nonzero,
  and there is no claim that all B-direction information is preserved.

## Question (GPT Pro)

Does PCA + bias keep working when R, B and RB are intervened on separately, or is the RB result an
endpoint/cancellation effect?

## 1. PCA-bias, k=192, last_token, natural donor: RMS e_T by subspace

| split | S = R | S = B | S = RB |
|---|---|---|---|
| test | 0.04740315 (mean e +0.0124) | 0.05837510 (mean e −0.0164) | 0.06841688 (mean e −0.0103) |
| validation | 0.04636886 (mean e −0.0026) | 0.05889596 (mean e −0.0200) | 0.07397255 (mean e −0.0271) |

* Reduced-model Δ_B ≡ 0 is structural, so `e_B = −T_B^orig`. The original-model RMS of T_B for PCA k192 is
  0.05837502 (test) and 0.05889607 (validation); max |e_B + T_B^orig| is ≤ 1.4e-5. The S = B column therefore
  measures the **original B response that the reduced model cannot carry**. It is not a compensation error.
* Each value is a single point estimate. A post-hoc rule like "max{RMS e_R, e_B, e_RB} ≤ ε" would be a new
  criterion. It is not adopted, and no "pass" is claimed (point values here: ≤ 0.074 vs ε 0.0897; no CI computed).
* Any grid CIs for the R/B cells in `results.json` come from the Day-2 ordinary template bootstrap (B = 2000). They
  must not be mixed with the Day-3 stratified inference (B = 10,000).

## 2. Signed decomposition `e_RB = e_R − C`, `C = T_B^orig + I^orig` (test, k=192, LT)

`MSE_RB = MSE_R + (−2E[C·e_R]) + E[C²]`:

| model | MSE_R | −2E[C·e_R] | E[C²] | MSE_RB | identity residual (max per pair) |
|---|---|---|---|---|---|
| SVD bias | 0.01649879 | −0.23689640 | 1.84499338 | 1.62459545 | 1.27e-5 |
| SVD gauss | 0.01201595 | −0.16325220 | 1.84499338 | 1.69375672 | 1.05e-5 |
| PCA bias | 0.00224706 | −0.00064897 | 0.00308282 | 0.00468087 | 1.30e-5 |

Validation shows the same pattern: SVD bias 0.0242 / −0.3471 / 1.9371 / 1.6142; SVD gauss 0.0148 / −0.2302 /
1.9371 / 1.7218; PCA bias 0.00215 / −0.00046 / 0.00379 / 0.00547.

## 3. Reading (arithmetic only)

* **SVD.** E[C²] ≈ 1.84 dominates, i.e. the B + interaction part the reduced model cannot carry. Gauss has the
  lower R-only error, but bias's larger negative cross term (−0.237 vs −0.163) offsets more of the B loss. That is
  how P1 comes out "gauss higher" in RB. This is metric-level error offsetting, not a shown network mechanism.
* **PCA-bias.** All three terms are small: E[C²] = 0.0031, MSE_R = 0.0022, cross term −0.0006. The low RB error
  is **not** mainly a cancellation artefact. It comes from small R error together with a small B + interaction
  response in this held-out data.

  The B part is not zero: RMS T_B 0.058 on test, about 3.3% of the original RB RMS 1.758. RMS‖δ_B‖ on test is
  0.577, vs 0.14 on calibration.
* **Limits:**
  * still scalar-T only: the local/logit vector relative errors are ~13.8% / ~15.3% in LT
    (`notes/codex_day3_vector_ratio_audit.json`);
  * same task and frames as the test;
  * no CI for this post-hoc view;
  * not a statement about the network's internal mechanism.
