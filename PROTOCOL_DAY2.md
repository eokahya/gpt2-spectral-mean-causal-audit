# PROTOCOL — Day 2 geometric controls (day2-v1)

**Status.** Exploratory, calibration split only. It extends pilot-v1 (`PROTOCOL.md`) and amendment v1.1, which
stay byte-identical, as do `config/pilot_v1.json`, `manifests/` and all Day-1 runs. Validation and test are not
opened. No ε, confirmatory rank or best seed is chosen. RB stays the primary *candidate*; R is secondary. Day-1
negative/limiting statements are kept.
Registration: `registration/DAY2_REGISTRATION.json` (hashes of this file, `config/day2_v1.json`, the Day-1
manifests, `metadata/`, source and tests), written before the Day-2 measurement run.

**Question.** Are the year transfer seen with the SVD cut and the compensation behaviour explained by
dimension, activation geometry and intervention size? No new layer, model, task or rank: GPT-2 small block 9,
`W_in` only, `k ∈ {96, 192, 384}`, the same 192 calibration pairs and the same, separate 576-prompt GT-moment pool.

## A. Input-space bases (same k; all fitted on the original-model final-token MLP input `X` of the moment pool)

Each basis gives a *column* map `P` on the 768-d input (`x_R = P x`, `x_B = Q x`, `Q = I − P`).
Row-vector code uses `x_row @ Pᵀ`. Nothing is fitted on pairs, task labels or T outputs.

| id | definition | projector |
|---|---|---|
| `svd` | right singular vectors of `W_in` (Day-1 reference) | orthogonal, `V_k V_kᵀ` |
| `pca` | top-k eigenvectors of the centred covariance `Σ = cov(X)` (eigh, eigenvalues descending) | orthogonal |
| `coord` | top-k coordinates by `E[x_j²]` (uncentred second moment); ties broken by lower index | orthogonal, diagonal 0/1 |
| `random_s{s}` | Haar orthogonal `O` (QR of an iid N(0,1) 768×768 matrix with `sign(diag R)` fix), first k columns; seeds **101, 202, 303, 404, 505**, all reported, none selected | orthogonal; nested in k |
| `wsvd` | covariance-whitened weight SVD (B below) | **oblique** |

## B. Whitened weight SVD (explicit algebra)

* `Σ = cov(X)` is **centred** (pool mean `m`) and `n = 576 < 768`. The LayerNorm also gives an exact null
  direction, so `Σ` is singular. Fixed ridge rule, independent of any output:
  `C = Σ + λI`, `λ = 0.01 · tr(Σ)/768`. Eigenvalue range and the condition numbers of `Σ`, `C` and `L` are recorded.
* `C = L Lᵀ` (Cholesky, CPU float64). SVD `W_in L = U' S' V'ᵀ`. Whitened rank-k map
  `W_k = (W_in L)_k L⁻¹ = W_in P`, where `P = L V'_k V'_kᵀ L⁻¹` (column map). `P² = P` and `P + Q = I`, but in
  general **`P ≠ Pᵀ`**: it is not a Euclidean-orthogonal projector. Row form: `x_row @ Pᵀ`, `δ_row @ Pᵀ`,
  `δ_row @ Qᵀ`. Tested: idempotence, complement, `W_in P = (W_in L)_k L⁻¹`, residual
  `‖(W_in − W_in P)L‖²_F = Σ_{i>k} s'_i²`, full rank (`k=768 → P = I`), row/column orientation.
* Role of centring and the mean: `W_k` is the rank-k map minimising `‖(W_in − W_k) L‖_F`, i.e.
  `E‖(W_in − W_k)(x − m)‖²` with `C` in place of `Σ`. It targets the **fluctuation** of the pre-activation.
  The discarded constant part has pool mean `μ_q = E[W_in Q x] = W_in Q m`. The `bias` compensation adds exactly
  this, so the **pre-activation mean** over the pool is restored exactly. There is **no** first-order or mean
  guarantee after `gelu_new` or downstream. `zero` drops `μ_q`; `gauss` additionally uses the per-neuron variance. This is our own definition: no SVD-LLM
  replication is claimed. SVD-LLM whitens with the uncentred Gram matrix.

## C. Geometric controls in the ORIGINAL model

Same intervention as Day 1: original-model caches fixed; recipient residual bypass fixed; only the final
prediction token's MLP-branch input is patched; score with the fixed pair vector `v_rd`. For every
basis × k × donor ∈ {natural, same_year, wrong_donor} × S ∈ {R, B}, plus RB (basis-independent, run once),
under three **separate** regimes:

1. `native`: `δ_S = δ @ P_Sᵀ` unscaled.
2. `l2`: `δ_S` rescaled so that `‖δ‖₂` equals the target.
3. `preact`: `δ_S` rescaled so that `‖δ @ Wc‖₂` (original pre-activation change, no bias) equals the target.

`l2` and `preact` are separate controls: a single scalar cannot match both, and we never say it did.
Targets (column `match_target` in the CSV):
* donor = natural, basis ≠ svd: the same pair's and S's **SVD natural** delta (`svd_natural_same_S_k`).
  For basis = svd the scale is 1.
* donor ∈ {same_year, wrong_donor}: **this basis's natural** delta for the same S and k
  (`same_basis_natural_same_S_k`); for RB, the natural RB delta.

Pre-specified rules:
* **Degenerate**: native norm < 1e-8, or < 1e-6 × target, or target < 1e-8. Such a row is not rescaled (delta 0),
  marked `degenerate`, excluded from aggregates and counted.
* **Out of support**: scale ∉ [0.25, 4]. The row is kept, flagged `oos`, counted; scale quantiles
  (0, 5, 50, 95, 100 %) are reported per cell; when oos > 0, aggregates are also given without oos rows.
* After matching: relative norm error < 1e-9 (float64) and < 1e-5 (float32 as sent to MPS) is **asserted**,
  and NaN/Inf guards run, before any forward pass.
* `wrong_donor` is not a zero-expectation null: its own year shift partly aligns with `v_rd`. The per-pair
  alignment `⟨v_rw, v_rd⟩/‖v_rd‖²` is written to the CSV. Natural-vs-wrong comparisons alone do not show
  content specificity.
* **Two distinct paired contrasts**, never mixed. The matched natural delta (target: SVD) and the matched
  donor-control delta (target: this basis's native natural) are generally *not* at the same norm, so they are
  never subtracted directly.
  1. *Cross-basis*: `T[b, natural, regime ∈ {l2, preact}, matched to SVD] − T[svd, natural, native]`, same
     pair, S, k (plus the native-vs-native version, labelled `native`).
  2. *Donor-control*: `T[b, natural, native] − T[b, donor ∈ {same_year, wrong_donor}, regime, matched to b's
     native natural]`, same pair, S, k (plus native-vs-native).
  Each contrast uses one common pair mask (non-degenerate on both sides), and an oos-filtered variant (oos
  removed from both sides). Paired pair-iid and template-cluster bootstrap CIs; n is reported.

Readouts: T, `‖Δ local MLP-out‖`, `‖Δ logits‖` (centred), native/matched delta norm and pre-activation
energy, scale, flags. Per basis and k: the original-model interaction `I = Δ_RB − Δ_R − Δ_B` (native).

## D. Compensation reference (reduced models), native natural interventions only

For every basis × k × {zero, bias, gauss, gauss_wo} × scope ∈ {last_token, full_layer} (scopes kept
separate): unperturbed KL(orig‖M) and PD (and top-1 agreement) on recipients; natural donor
`S ∈ {R, B, RB}` with the same physical delta `δ @ P_Sᵀ` (basis-specific) applied to the original and the reduced
model, each compared with its own original counterpart; local / logit / T fidelity.
* Norm-matched regimes are run in the original model only; there is no reduced × control × regime cross product.
* Moments as on Day 1: `final_token` pool for `last_token`, `all_token` pool for `full_layer` (the scope/pool
  confound stays), all with the basis's row map `Qᵀ`. `gauss_wo` estimates only the *moments* from weights
  (isotropy) **given the basis**. For `pca`, `coord` and `wsvd` the basis itself is learned from calibration
  activations, so the whole method is not "weights-only". Moments are fixed;
  `Δ_B ≡ 0` in reduced models is structural and is not counted as success.
* Gauss–Hermite: node count and σ bound for Day 2 are fixed in `config/day2_v1.json` **before** measurement by a
  `quadcheck` over all bases' calibration moments. Rule: `σ_bound` = 1.25 × max observed σ, rounded up to
  0.5; nodes = smallest of {64, 128, 256, 512} with error < 1e-7 vs 1024 nodes on the grid `m ∈ [−20, 20]`,
  `σ ∈ [0, σ_bound]` and on actual `(z+μ, σ)` samples. No erf-GELU; Day-1 settings unchanged.
* Signed decomposition for every reduced cell: `e_RB = e_R − C`, `C = T_B^orig + I^orig`, with
  `MSE_RB = MSE_R − 2E[C e_R] + E[C²]` and the identity residual reported.

## E. Statistics, budget, records

* The unit is the **pair (N = 192)**; 18 (frame, noun) templates but only 2 frames. Row counts are not N.
* Mean signed T / e_R / e_RB and RMS, each with pair-iid and template-cluster bootstrap 95% CIs (B = 2000).
* Random bases: per-seed results plus the across-seed spread (mean, SD, min, max of per-seed statistics).
  Prompt and seed uncertainty are **not pooled** into one CI.
* Native and matched-control results are reported separately.
* No best seed, rank or basis is selected from Day-2 outputs. Matched quality and mechanism specificity are not
  claimed.
* MPS float32 for the model; bases, matching and metrics on CPU float64; no device fallback. Batch = 192 pairs.
* Hard budget 900 s, with a 150 s aggregation reserve. Checkpoint (rows + partial JSON) after the original
  controls and after every basis. The Day-1 positive-control gate is re-checked first.
* Raw per-pair CSVs, failed runs and snapshots are kept.
* Smoke is a **failure gate** (it raises; the run is not started until it passes). Pre-specified tolerances
  in `config/day2_v1.json` `smoke_gates`: Day-1 replay |ΔT| ≤ 1e-4 for the original model and ≤ 1e-3 for reduced
  models (the Day-2 quadrature node count may differ); positive control valid; wsvd reduced B-null
  ≤ 2e-3 logits; RB patch vs donor MLP-out ≤ 1e-4; matching asserts.
* Raw per-pair CSVs: `control_rows.csv`, `reduced_rows.csv`, `unperturbed_rows.csv` (KL, PD, top-1 per variant and pair).
  The budget is checked before every control cell (basis × k) and before every reduced variant.
* Smoke: replay of the Day-1 SVD k=192 numbers (original T_R/T_B/T_RB and reduced T) against
  `runs/20261030T153707Z_pilot/per_pair_rows.csv`; max abs difference reported.
