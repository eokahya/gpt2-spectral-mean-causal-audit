# Claude — Day-3 test results and independent reading (2026-10-30)

## Run chain (no rule changed after registration)

* Registration `registration/DAY3_REGISTRATION.json`, 16:42:18Z, `day3-v1`, ε = 0.08973707820478917.
* Validation `runs/20261030T164225Z_day3_validate` (ok, 240 s); independent audit by Codex:
  `notes/codex_day3_validation_independent_audit.json`.
* Freeze `registration/DAY3_FROZEN_SELECTION.json` and `DAY3_FROZEN_EPSILON.json`; `verify_frozen` OK.
* **Test `runs/20261030T164723Z_day3_test`**: status ok, 207 s; source status `registered`; frozen file hashes
  recorded in the manifest; no KL reselection on test.

## Test numbers (frames F4/F5, N = 192 pairs, 18 template clusters)

**Gate valid.**
* Mean PD 0.905 [0.893, 0.915]; T_RB 1.706 [1.647, 1.765]; natural − same_year 1.705 [1.636, 1.772].
* Per frame: F4 T_RB 1.707, F5 T_RB 1.705.
* Original RB RMS 1.758; ε / original RB RMS = 5.1%.

Pre-specified inference (B = 10,000, stratified frame-within template-cluster bootstrap; nominal, approximate,
conditional intervals):

| check | statistic | estimate | interval | rule | result |
|---|---|---|---|---|---|
| **P1** | SVD k192 LT RB: RMS gauss − RMS bias | **+0.0268** (1.3014 − 1.2746) | 95% [0.0235, 0.0302] (iid [0.0237, 0.0300]) | direction | **gauss higher** |
| **S1** | RMS(PCA-bias) − RMS(SVD-bias) | −1.206 | one-sided 97.5% upper −1.1365 (iid −1.158) | upper < 0 | **supported** |
| **S2** | RMS(PCA-bias) | 0.0684 | one-sided 97.5% upper 0.0780 (iid 0.0791) | upper < ε = 0.0897 | **passes** |

* S2 NRMSE: 0.038 (calibration denominator), 0.039 (test denominator).
* Per frame (descriptive):
  * P1: F4 +0.025 [0.020, 0.031], F5 +0.028 [0.025, 0.032];
  * S2 upper: F4 0.08907559 (ε 0.08973708; descriptive), F5 0.073. On validation F3 was 0.100 > ε; per-frame
    values do not change the criterion.

Frozen KL-matched contrasts (descriptive; the KL gap was recomputed on test, and every frozen pair stays within
the rule):

| family | pair | test KL gap | RMS e_RB difference [95% cluster CI] |
|---|---|---|---|
| Q1 (SVD-bias vs PCA-bias) LT and FL | none matched on validation | — | no iso-KL claim |
| Q2 (SVD gauss vs bias) LT | k384 / k384 | 0.003 | +0.029 [0.025, 0.033] |
| Q2 (SVD gauss vs bias) FL | k192 / k192 | 0.050 | +0.007 [0.004, 0.009] |
| Q3 (PCA gauss vs bias) LT | k384 / k384 | 1.5164e-5 | −1.53e-7 [−1.32e-6, 1.01e-6]: not distinguished, at numerical scale, not evidence of equivalence |
| Q3 (PCA gauss vs bias) FL | k192 / k192 | 0.045 | +0.008 [0.002, 0.014] |

All four matched pairs are same-rank comparisons.

Descriptive context:
* LT k192 KL:
  * svd bias 0.142, gauss 0.132;
  * pca bias 0.0022 (calibration 3.4e-4, so it rose on held-out data);
  * wsvd bias 0.0026.
* Random-subspace bias RB RMS at k192 (5 seeds): 1.252–1.274, vs SVD 1.275.
* T_R / T_B at k192: SVD 0.40 / 1.26, random_s101 0.41 / 1.24, PCA 1.68 / 0.016.
* SVD R-only RMS: gauss 0.110 < bias 0.128.

## Independent reading (narrow, conditional; Codex wording corrections applied)

1. **Registered primary (P1): no support for Gaussian superiority.** On held-out test frames, with the
   calibration bases and moments frozen, the calibrated Gaussian compensation gives a *slightly larger*
   end-to-end RB error than mean (bias) compensation at SVD k=192 (+0.027; 1.5% of the original RB RMS). The
   direction matches calibration (+0.035) and validation (+0.042).

   The Day-1 identity `e_RB = e_R − (T_B + I)` and the lower Gaussian R-only error (0.110 vs 0.128) on test fit
   the earlier error-offset account. That account is arithmetic, and no network mechanism is established.
2. **Secondary (amendment, PCA chosen after calibration): both checks pass.** At the same rank, an
   activation-PCA subspace with mean compensation reproduces the final-token year-transfer RB response with far
   smaller error than the weight SVD (S1). Its absolute RB error is inside the pre-registered 5% operational
   tolerance (S2).

   The S2 margin is modest: the upper bound is 0.078 against ε = 0.0897, and the F4 frame alone (descriptive) comes to 0.08907559 vs ε 0.08973708.
   ε was proposed after calibration errors had been seen. PCA's unperturbed KL rose about 6× from calibration
   to test. So S2 passing means: an **operational tolerance on an aggregate scalar response (T)** is met on new
   frames of the **same task**. It is **not** a percentage of causal preservation, **not** strong mechanism
   preservation, and not generalisation beyond the task. Nor is it whole-vector preservation. For PCA-bias k192
   test RB:
   * the local MLP-out vector RMS relative error is 0.138 in LT and 0.409 in FL;
   * the centred logit vector RMS relative error is 0.153 in LT and 0.465 in FL.

   These ratios use vector-response denominators and are not the same metric as the scalar T NRMSE of 0.039.
3. **Weight SVD vs random subspaces:** point values stay similar on test (RB RMS, T_R/T_B). There is no
   formal equivalence test; this is consistent with the Day-2 dimension reading.
4. **Same-rank KL-matched Gaussian-vs-bias comparisons** are descriptive. Gauss has higher RB error in
   specific contrasts (Q2 LT k384, Q2 FL k192, Q3 FL k192). In PCA LT k384 the two are not distinguished: the
   difference is at numerical scale, which is not evidence of equivalence. None of this supports a universal
   Gaussian benefit or harm.
5. **Not claimed:** causal/mechanism preservation, circuit identification, compression or speed, equal quality,
   generalisation beyond the GT task and its 6 frames, global FWER control.
6. **MI-novelty:** GPT Pro's pre-specified condition applies. The held-out result is essentially a
   data-sensitive reconstruction: an activation subspace fitted on same-task inputs reproduces same-task
   responses. In my reading that makes an **MI-novelty claim a NO-GO**. What remains is a *causal-audit
   observation*:
   * the registered Gaussian-vs-mean hypothesis did not hold in the expected direction;
   * at this layer and rank, which subspace is kept matters far more than the compensation.

Open caveats for the final review:
* 2 frames per split;
* one layer, one task, one model;
* the scope/moment-pool confound in full-layer runs;
* PCA's held-out KL rise;
* ε chosen with knowledge of the calibration errors.
