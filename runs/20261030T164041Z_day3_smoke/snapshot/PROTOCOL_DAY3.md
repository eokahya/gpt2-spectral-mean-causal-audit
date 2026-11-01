# PROTOCOL — Day 3 held-out validation and test (day3-v1, DRAFT)

**Status: DRAFT for pre-registration review (Codex, GPT Pro, Antigravity). Not registered; nothing frozen.**
No validation or test model forward pass is run before `registration/DAY3_REGISTRATION.json` exists (it is
written after review). Day-1/Day-2 protocols, configs, registrations, manifests and runs are unchanged.

## 1. Scope (fixed; nothing new)

* GPT-2 small, block 9, `W_in` only. Ranks 96 / 192 / 384. No new rank, layer, model, task or general-text pool.
* Bases, all from the **Day-2 calibration fit**, serialised once and never refitted:
  * `svd` — main;
  * `pca` — main, **data-driven choice made after the Day-2 calibration results**;
  * `wsvd` — secondary sensitivity (oblique geometry);
  * `random_s{101, 202, 303, 404, 505}` — fixed dimension control. This is not a third main family, and no
    seed is selected.

  Coordinate masks are dropped.
* Compensations `zero | bias | gauss | gauss_wo`. Scope `last_token` is primary, `full_layer` secondary.
  Gauss–Hermite uses 128 nodes with σ ≤ 2.5 (Day-2 settings).
* The reduction is a dense, functional rank-k audit. There is no compression or speed claim.
* Original-model geometric controls: the Day-2 definitions (native / l2 / preact; common and oos masks)
  unchanged. oos rules are not changed after any held-out data are seen. Cells that are unsupported on
  calibration stay noted; for example, PCA/wsvd matched-B had 0/192 pairs in support.
* Resample ablation is **not run** in bounded Day 3: the main aim is the held-out check, and resample is only an
  optional diagnostic.

## 2. Held-out data

* Source: the sealed Day-1 prompt lists only.
  * Validation = frames F2/F3 × validation nouns × validation years (`manifests/sealed_prompts_validation.csv`).
  * Test = F4/F5 × test nouns × test years (`manifests/sealed_prompts_test.csv`).

  `manifests/` is never written.
* Per split, 192 pairs are built with the Day-1 algorithm: 6 strata (direction × diff bin) × 32, plus a
  same_year and a wrong_donor control donor. The seed is fixed in config; the construction does not depend on
  any output.
* Pairs are written to `manifests_day3/{validation,test}_pairs.csv` and hashed **before** any model
  measurement.
* Any tokenization failure in a sealed pool aborts the stage before anything is written.
* Validity tests check: balance, diffs, no overlap across splits and with calibration, control membership,
  pair/control text+ids equal to the sealed records, and the 12-cell composition (frame × direction × diff bin)
  with every template of a frame present in each of its 6 cells. The last check guarantees that every bootstrap
  denominator is positive.

## 3. Calibration artifact (the most critical invariant)

* `calibration_day3/calib_artifact.pt` holds, on CPU in float64:
  * every basis matrix `P` (column map, 768×768);
  * the fixed moments `μ, σ` (3072) for the `final_token`, `all_token` and `weights_only` pools, per basis × rank.

  These come only from the Day-2 calibration moment pool, together with the pool's input hashes and the
  calibration B-energy.
* `calibration_day3/calib_index.json` records the file SHA-256, per-tensor SHA-256, shapes and dtypes, the
  model weights SHA, the fit settings (families, seeds, ranks, ridge, fit pool, layer, σ bound) and the pool hashes.
* Loading checks:
  * hash, shape, dtype, device and finiteness;
  * σ ≥ 0;
  * each `P` is a projector;
  * the exact tensor set required by the runtime config;
  * that fit settings, model weights and pool manifest match the runtime.
* The held-out code path cannot call the moment pool: it raises. There is no validation/test moment, PCA fit or
  refit.
* The artifact hash is checked again after every run.
* A held-out input-shift diagnostic (B-energy of held-out recipients vs calibration) is reported only. Nothing
  adapts to it.

## 4. Positive gate (per held-out split, same criterion as calibration)

Original model on that split:
* mean PD > 0;
* CI_lo(mean T_RB natural) > 0;
* CI_lo(mean[T_nat − T_same_year]) > 0.

Per-frame values (2 frames) are reported separately. Per-pair gate data (PD, T for natural / same_year /
wrong_donor, frame, stratum) go to `gate_rows.csv` before the decision. A failed gate makes that split's assay
**invalid**: no reduced-model claim is made, nothing is adapted, and the raw outputs are kept. Passing the gate
does not guarantee against a weak-signal split.

## 5. Analyses (anchor rank k = 192, fixed in advance — no best-rank search)

Readouts:
* T, at the final position, scored on the J = 00..99 logit set with the pair's `v_rd`;
* the local MLP-out response (768-d, final token);
* the logit response (full-vocabulary 50,257 logits, centred);
* unperturbed KL(orig‖M) and PD.

The unit is the pair (N = 192 per split). Templates: 18 clusters, 2 frames. Nothing is claimed beyond these two
frames.

* **Inference bootstrap (new for Day 3; Day-2 grid CIs unchanged):**
  * Stratified over the 12 cells frame × direction × diff bin, with fixed weights `N_c/N`.
  * Primary — frames fixed; in each draw, the 9 templates of each frame are drawn with replacement. Each cell's
    mean squared error is the multiplicity-weighted mean, and the cells are combined with the weights `N_c/N`
    before taking the √. The point estimate equals the plain pair mean; template correlation across strata is
    kept.
  * Supporting — pair-iid: `N_c` pairs are resampled within each cell.
  * **B = 10,000.**
* **P1 — registered primary candidate (Day 1, unchanged):** SVD, k=192, `last_token`, natural donor, S = RB:
  `D = RMS(e_T^gauss) − RMS(e_T^bias)`, paired.
  * Reported: point value, two-sided 95% stratified-cluster CI (primary), pair-iid CI.
  * Direction flag only: CI < 0 → gauss lower; CI > 0 → gauss higher; otherwise not distinguished.
  * **No equivalence flag.** ε is an absolute reconstruction tolerance, not a margin for a gauss−bias difference.
  * A positive or undistinguished D does **not** support Gaussian superiority.
* **Secondary confirmatory checks (amendment: PCA was chosen after calibration; not a primary switch):**
  Bonferroni within the secondary family {S1, S2} only, using **one-sided 97.5% upper bounds** (q = 0.975 of the
  stratified-cluster bootstrap); pair-iid is also reported. This is **not** a global FWER guarantee that
  includes P1.
  * **S1** — `RMS(e_RB^{pca,bias}) − RMS(e_RB^{svd,bias})`, k=192, LT, same physical RB target. Supported iff
    upper < 0.
  * **S2** — `RMS(e_RB^{pca,bias})`, k=192, LT. Preserved iff upper < ε. Also reported as NRMSE with both
    denominators: the fixed calibration original RB RMS (1.7947) and this split's original RB RMS.
* All intervals are **nominal, approximate and conditional** on the fitted calibration bases/moments and on the
  2 frames per split. Per-frame point values and CIs are reported (descriptive).
* **Descriptive only (labelled as such, no inference):** all other bases, methods, scopes, ranks, S = R and B,
  random seeds (per seed + spread), wsvd, `gauss_wo`, controls, interaction, decomposition `e_RB = e_R − C`, and
  the Day-2-style grid CIs (B = 2000).
* **ε (accepted in pre-registration review):** ε = 0.05 × RMS(T_orig, natural, RB) on **calibration**
  (Day-1 CSV arithmetic): 1.7947415640957833 → **0.08973707820478917**.
  * It is a calibration-informed operational 5% tolerance on the original response scale. It is not fitted to
    any candidate's error, and it is not rescaled on test.
  * It was proposed after calibration candidate errors had been seen.
  * The full ε object (value, reference RMS, reference CSV path and SHA-256) is **frozen in the Day-3
    registration before validation**, and the freeze files copy it.
  * Each held-out split reports its original RB RMS and ε / original RB RMS. If the test-split scale approaches
    ε, passing S2 is not called strong mechanism preservation.

## 6. Validation quality matching (secondary, descriptive)

* Families (locked list):
  * **Q1** — SVD-bias vs PCA-bias;
  * **Q2** — SVD-gauss vs SVD-bias;
  * **Q3** — PCA-gauss vs PCA-bias.

  Each is run separately for `last_token` and `full_layer`. Candidates are the 9 discrete (k₁, k₂) pairs from the
  existing ranks.
* Validation mean KL(orig‖M) is computed in float64 from `unperturbed_rows.csv`. Symmetric relative gap
  `g = 2|KL_A − KL_B| / (KL_A + KL_B)` (0 if both are 0). A pair is **matched within rule** iff `g ≤ 0.10`.
  The absolute KL gap is reported as well.
* This is a deliberately strict rule with no floor: at small KL there may simply be no match, and then the
  result is "none".
* Selection: minimum g; ties broken by minimum max-rank, then minimum rank sum, then lexicographic (k₁, k₂).
* If no pair matches: **no iso-KL claim**. There are no interpolated models, extrapolation, new ranks or scopes.
* "Matched within rule" is not "equal quality". An across-rank difference is a configuration comparison, not a
  pure Gaussian effect.
* The selected pairs are frozen before test. On test:
  * there is **no** re-selection; only the frozen pairs are evaluated (descriptive RMS e_RB difference);
  * their KL gap is recomputed on the test split. If g > 0.10 there, no near-quality interpretation is made.

## 7. Freeze before test

`run_day3.py freeze` requires a validation run that is completed, `split = validation`, gate-valid, and whose
recorded source tree, config, Day-3 protocol, calibration artifact and Day-3 registration hashes all equal the
**current** ones. Otherwise the change has to be archived and validation re-run. It then writes `registration/DAY3_FROZEN_SELECTION.json` and
`registration/DAY3_FROZEN_EPSILON.json` with real content:
* the P1/S1/S2 specifications, anchor rank, KL-matching result per family/scope and ε value + derivation;
* hash bindings to `PROTOCOL_DAY3.md`, `config/day3_v1.json`, the source tree, the calibration artifact, the
  Day-3 registration, the held-out manifests, and the validation run's `results.json` and CSVs.

The test stage recomputes every binding, the full ε object (compared with both the freeze file and the
registration) and the KL selection (from the validation rows). It **hard-fails** on:
* any mismatch;
* a missing or dummy file;
* a split mix;
* a source tree that differs from the frozen one. Amendments do not bypass this after freeze; only an explicit
  invalidate/new-version process would.

This fulfils and strengthens Day-1 §9.

## 8. Execution rules

* A source-tree change after registration is a **hard failure** for validation. It is allowed only through
  `run_day3.py amend --reason ...`, which archives the new source snapshot and records its hash; a validation run
  must then be re-run before freezing. After freeze, no source change is allowed for test.
* A stage whose total walltime exceeds 900 s after writing `results.json` is recorded as `budget_exceeded`.
* No stage overwrites existing manifests, artifacts, freeze files or runs. Failed runs are kept.
* MPS float32; CPU float64 for bases, matching and metrics; no device fallback.
* 900 s per stage with a 150 s reserve. Checkpoints after the controls and after each basis.
* Raw per-pair `control_rows.csv`, `reduced_rows.csv` and `unperturbed_rows.csv`; run snapshot and environment.
* Test outputs never change criteria. Validation results change only what §6/§7 say they select.
