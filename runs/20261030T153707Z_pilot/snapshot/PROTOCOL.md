# PROTOCOL — Spectral-component mean replacement: causal audit pilot (pilot-v1)

**Working title.** *When Can Spectral Components Be Replaced by Their Average? A Causal Audit of Transformer MLPs.*

**Status.** Exploratory pilot (3-day). Nothing in this pilot is confirmatory. No novelty or publication claim is
made. All data produced under this protocol version come from the **calibration split** and are labelled
`exploratory_pilot`. The test split is sealed (code refuses to read it without a frozen-epsilon file, see §9).

**Registration.** `registration/REGISTRATION.json` stores the SHA-256 of this file, of `config/pilot_v1.json`,
of every split manifest and of the source tree, plus a UTC timestamp, written *before* the first pilot
measurement. Any later edit to this file is a new protocol version.

Related work we are aware of and do not claim to supersede: greater-than circuit (arXiv:2305.00586), LASER
(arXiv:2312.13558), Beyond Components (arXiv:2511.20273; its App. B.2 already performs class-mean swaps),
SVD-LLM (arXiv:2403.07378), Optimal Ablation (arXiv:2409.09951). We claim no new mathematics, no "first
singular-direction analysis", no renormalisation-group result, and we do not identify the discarded bulk with
memorisation.

---

## 1. Model, site, orientation

* Model: `openai-community/gpt2` (GPT-2 small), revision pinned in config (`607a30d783dfa663caf39e06633721c8d4cfcd7e`),
  float32, device `mps` (Apple M2 Max). No silent device fallback: if `mps` is requested and unavailable the run aborts.
* Site: zero-based block **9**, MLP input projection `W_in` only. `W_out` (`c_proj`) is never modified.
  **Fixed backup: block 10 only**, used only if the block-9 positive control (§5) is invalid. No other layer.
* Orientation. HF `Conv1D` `c_fc.weight` has shape `[d_model, d_mlp] = [768, 3072]` and computes `x @ Wc + b`
  (row vectors). The column-vector matrix in `a = W_in x + b` is `W_in = Wc^T` (`[3072, 768]`).
  SVD `W_in = U Σ V^T` (computed on **CPU, float64**, recorded in the run manifest), `V` spans the 768-dim
  input space.
* Retained / discarded input subspaces at cut `k`: `P_R = V_{:,:k} V_{:,:k}^T`, `P_B = I − P_R`.
  The plain rank-`k` truncation equals `W_in P_R` exactly.
* `x` = output of `ln_2` of block 9 (= the input of the MLP branch only; in GPT-2 `ln_2` feeds nothing else).
  Decomposition: `a = z + q`, `z = W_in P_R x + b_in` (retained pre-activation), `q = W_in P_B x` (discarded
  pre-activation). `g = gelu_new` (tanh approximation, GPT-2's actual activation).

## 2. Rank cuts

Pre-specified, not chosen from any measured response: `k ∈ {96, 192, 384}` (1/8, 1/4, 1/2 of `d_model`).
Retained Frobenius energy of `W_in` at each cut is *reported*, not used to choose. At most 8 cuts will ever be
used under this project; the remaining ≤5 are reserved. **TODO (not done, not required for the smoke):**
WeightWatcher-style spectral cutoff `k*`. No cutoff value is invented in its place.

## 3. Compensations (all applied to the discarded pre-activation of `W_in` only)

| name       | reduced MLP hidden unit `i`                               | moments from |
|------------|-----------------------------------------------------------|--------------|
| `zero`     | `g(z_i)` (plain SVD truncation)                           | — |
| `bias`     | `g(z_i + μ_i)`                                            | GT-moment pool (calibrated mean of `q_i`) |
| `gauss`    | `E_{ε~N(0,1)} g(z_i + μ_i + σ_i ε)`                       | GT-moment pool (calibrated mean and variance of `q_i`, per neuron, diagonal) |
| `gauss_wo` | same, with `μ_i = (W_in P_B β)_i`, `σ_i = τ‖(W_in P_B)_{i,:}‖₂`, `τ² = ‖γ‖²/d` | weights only (`γ, β` = `ln_2` affine). **Isotropy assumption only**: LN-normalised coordinates treated as isotropic, unit variance, zero mean. |

* The Gaussian expectation of `gelu_new` is computed by **probabilists' Gauss–Hermite quadrature**
  (**64 nodes**; weights renormalised to sum to 1, and `σ=0` returns `g(z+μ)` exactly). The node count was fixed
  before any pilot measurement by the `quadcheck` stage: smallest of {32, 64, 128, 256} with max abs error
  < 1e-7 vs 512 nodes both on the calibration-pool `(z+μ, σ)` of every rank/pool and on a grid
  `m ∈ [−20, 20]`, `σ ∈ [0, 2.0]` (32 nodes: 1.4e-6 on the grid; 4.8e-3 at σ = 4). Observed calibrated
  `σ_max = 1.40`. Accuracy is only claimed for `σ ≤ 2.0`; a reduction with larger `σ` is refused at run time. The closed-form
  expectation of erf-GELU is **not** used (it is a different function). Node-count convergence and an
  independent Monte-Carlo check are unit tests.
* Moments are **fixed** after calibration; at evaluation no moment or `σ` is updated from the intervened `x`.
* Interpretation limit: if `q` were Gaussian and independent of `z`, `gauss` would return `E[g(z+q) | z]`, i.e.
  only a *local, mean-level* guarantee for the hidden unit. No end-to-end or per-example guarantee is claimed.
  In GPT-2 `q` and `z` are not independent; that is part of what is audited.
* Moment pools: `final_token` moments (final prediction position of each GT-moment prompt) are used in
  `last_token` scope; `all_token` moments (pooled over all real positions, incl. position 0) are used in
  `full_layer` scope. A separate **general-text moment pool is a TODO** (not built today; kept separate by design).
* Resample-ablation (`E[loss]` over resampled `q`, as opposed to `loss(E[h])`) is a **future diagnostic
  reference only**, not a compression method. Not run today.

## 4. Scopes (kept strictly separate)

* `last_token` — *local diagnosis*. The reduced/compensated `W_in` is used **only at the final prediction
  position** of block 9; all other positions use the original MLP.
* `full_layer` — *actual module replacement*. The reduced/compensated `W_in` is used at **every position** of
  block 9 (what a compressed model would do).

With *identical* compensation moments the final-position local MLP-output response would be identical in both
scopes (MLPs are position-wise; earlier positions do not see the patch). In this protocol the scopes use
**different moment pools** (`final_token` vs `all_token`, §3), so for `bias`/`gauss` a final-position local
difference between scopes reflects the moment pool, not the scope itself; `zero` and `gauss_wo` use the same
(or no) moments in both scopes. Every result row records its moment pool. Scopes are reported, never pooled.

## 5. Task, intervention and score

**Prompts.** Greater-than task (Hanna et al. 2023 style): `"The {noun} <frame> the year 17{YY} <frame> the year 17"`,
six frames F0–F5 and 36 event nouns (listed in config). Every (frame, noun, YY) prompt is tokenizer-checked:
`" 17"` and `"YY"` must be separate single tokens at the start-year slot and the prompt must end in the single
token `" 17"`; failures are excluded deterministically *before* sampling. `YY ∈ [02, 98]`.
Examples are **never selected by model output**.

**Score tokens.** `J` = the 100 two-digit tokens `"00".."99"` (each verified to be one GPT-2 token).
For a recipient year `r` and donor year `d` (`r ≠ d`):
`v_rd = center_J(1[j > d] − 1[j > r])`, and for any intervention `S`
`T_S = ⟨ℓ_patch − ℓ_rec, v_rd⟩_J / ‖v_rd‖²` (logits restricted to `J`).
`T = 1` means the logit change is, in projection, one unit of the donor-minus-recipient greater-than indicator.
For **all controls the `v_rd` of the actual different-year pair is used**, so `‖v_rd‖ > 0` always; we never divide
by a zero vector.

**Intervention.** Only on the branch into `W_in` (the MLP input), only at the final prediction token, using
cached **original-model** activations:
`x' = x_rec + P_S (x_don − x_rec)`, `S ∈ {R, B, R+B}`. For `S = R+B` the full difference is used, i.e. `x' = x_don`
(full MLP-branch donor transfer). The residual bypass of block 9 and all other positions at the intervention
time are untouched; layers 10–11 and the unembedding recompute normally. Donor/recipient caches are never mutated.
**The same physical delta `P_S(x_don − x_rec)` (with `P_S` from the original `W_in` SVD) is applied to the
original model and to every reduced model** — not model-native directions. Upstream of block 9 nothing changes,
so `x_rec` is identical in both (asserted at run time).

**Pairs (pilot).** Recipient and donor share the template (frame, noun) and differ only in `YY`.
`|d − r| ∈ [10, 40]`, balanced over direction (`d > r`, `d < r`) × difference bin (`10–19`, `20–29`, `30–40`):
6 strata × 32 = **192 pairs** (fixed budget, fixed seed).

**Controls (per pair, scored with that pair's `v_rd`).**
* `same_example`: donor = recipient (delta exactly 0; identity check).
* `same_year`: donor = a different calibration template with year `r` (context change, no year change).
* `wrong_donor`: donor = a different calibration template with a year ∉ {r, d}.
  No control is claimed to have zero expectation; for every control we report the **effect norm**
  (`‖Δ MLP-out‖`, `‖Δ logits‖`) separately from the **pre-activation energy** (`‖P_S δ‖`, `‖W_in P_S δ‖`).

**Positive control / validity gate** (original model, `S = R+B`, natural donor):
valid iff (a) mean greater-than probability difference `PD` of the original model on calibration recipients
is > 0, and (b) the pair-bootstrap 95% CI of mean `T_{R+B}` is > 0 and the paired difference
`T_{R+B}(natural) − T_{R+B}(same_year)` has CI > 0. If invalid: stop, report cause, use block 10 only after a
new protocol version. No findings are produced without a valid positive control.

## 6. Comparisons and metrics (all reported separately)

For every variant `M` (rank, compensation, scope) and intervention `S`, the response
`Δ_S^M = f_M(x') − f_M(x_rec)` is compared with **its own original counterpart** `Δ_S^orig` (same `S`, same delta):
error `e_S = Δ_S^M − Δ_S^orig`. Readouts, never merged:

1. local final-token MLP-output response (768-d),
2. final-token logits (full vocabulary, centred; and `J`-restricted),
3. `T` error (scalar),
4. unintervened `KL(p_orig ‖ p_M)` at the final token (recipient prompts),
5. greater-than probability difference `PD = Σ_{j>r} p_j − Σ_{j≤r} p_j` (Hanna et al.; unintervened; full-vocab softmax).

For each: **mean (signed) error** and **RMS per-example error** are reported separately, with 95% bootstrap CIs
(pair-level i.i.d. and template-cluster). Interaction `I = Δ_{R+B} − Δ_R − Δ_B` is a diagnostic of the
**original** model; in any reduced model `Δ_B ≡ 0` and `I ≡ 0` structurally — that is not a success criterion.

## 7. Primary confirmatory comparison (candidate — NOT yet frozen)

*Candidate:* calibrated `gauss` vs `bias`, **same rank**, `last_token` scope, natural donor, `S = R+B`;
statistic `RMS_i(e_T,i)` for each method and the paired difference `RMS_gauss − RMS_bias`.
A "preserved response" claim requires the upper 95% CI bound of `RMS(e_T)` to be below a margin `ε_T`;
a non-significant difference is never read as preservation. `ε_T` (and, if adopted, the comparison margin)
will be **frozen from calibration/validation data before the test split is opened**. Today's numbers are
exploratory; ranks, thresholds and rules are not to be chosen from test output.
Secondary: same quantities in `full_layer` scope, for `S = R`, `S = B`, local and logit readouts, and `gauss_wo`.

## 8. Splits (seeded; manifests in `manifests/`)

* Frames: calibration {F0, F1}, validation {F2, F3}, test {F4, F5}.
* Nouns: 36 nouns randomly assigned (seed) to four disjoint groups of 9: `moment_pool`, `calibration`,
  `validation`, `test`.
* Years `02..98`: sorted, cut into consecutive triples, each triple randomly assigned (seed) one year each to
  calibration / validation / test (stratified across the range).
* GT-moment pool = calibration frames × `moment_pool` nouns × calibration years (all valid prompts). It shares no
  (frame, noun) template with any recipient/donor prompt.
* Pilot pairs and their control donors = calibration frames × `calibration` nouns × calibration years.
* Validation and test prompts are generated and hashed but **not run** in this pilot.
Templates in the pilot: 2 frames × 9 nouns = 18 template clusters; frames alone are only 2 clusters, so
cluster-level uncertainty is weak and is not over-claimed.

## 9. Execution rules

* Seed `20261030` everywhere (numpy, torch). Bootstrap `B = 2000`.
* Walltime budget for the pilot: 900 s, **enforced**: before each (rank, method, scope) variant the run stops
  if elapsed > 900 − 120 s (aggregation reserve); rows (`per_pair_rows.csv`) and `partial_results.json` are
  checkpointed after every variant; a run that stops is kept with status `budget_exceeded`.
* Every run copies the exact source, config, protocol, manifests and registration into `runs/<id>/snapshot/`.
* Failed raw runs are kept (`runs/`), never deleted.
* Every run writes `run_manifest.json`: model revision, safetensors SHA-256, seed, dtype/device, SVD
  device/dtype, package versions, config/source/manifest hashes, walltimes.
* Test split: `--split test` is refused unless `registration/FROZEN_EPSILON.json` exists.
