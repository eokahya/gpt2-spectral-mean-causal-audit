# Discarded-state response floor gate — development protocol (drfg-dev-v1, DRAFT until registration)

**Sources.**
* `notes/research_direction_rg_2026-10-31.md`: the selected hypothesis and the reviewer debate.
* `notes/gpt_pro_opus55_discarded_state_gate_prompt_2026-10-31.txt`: GPT Pro's implementation prompt.
* `notes/gpt_pro_gate_addendum_2026-10-31.txt`: GPT Pro's addendum. It replaces the projector, donor-sampling and
  power-planning sections of the prompt, and includes the coordinator's resource clarifications.

**Machine-readable counterpart:** `config/gate_dev_v1.json`. If the two disagree, that is a blocker to report, not
something to resolve silently.

**Status.** This is a development and feasibility pilot.
* It is not a confirmatory test, an RG or WeightWatcher test, or a novelty claim.
* The closed propagation audit (INCONCLUSIVE) and all Day-1..3 and follow-up files are unchanged and are not used
  as data.

## 1. Question and fixed system

**Question.** With the recipient context c, the retained state z and the retained displacement d held fixed, does
the response to the fixed retained intervention change when the discarded component q changes? The comparison is
under the explicitly declared donor distribution Q.

**Fixed system.**
* **Model:** GPT-2 small at HF revision `607a30d7…`, eval mode, frozen weights. It runs on MPS in float32. Statistics
  and geometry are computed on CPU in float64.
* **Site:** block 9 (zero-based), the MLP branch input (ln_2 output), final real token only.
* **Projector:** the frozen Day-3 artifact `calibration_day3/calib_artifact.pt` (sha256 `cd56bb2f…`), loaded and
  verified by `scp.day3.load_artifact`. `P = P|pca|192` is used **directly**:
  * checks: symmetric (max |P − Pᵀ| ≤ 1e-12), idempotent (≤ 1e-10), trace 192 (±1e-8), and exactly 192 eigenvalues
    above 0.5;
  * no refit;
  * V is obtained from eigh(P) only in tests.
* **No input mean.** The artifact stores no input-space mean, and none is constructed. The stored `mu|pca|192|*`
  are discarded **pre-activation** means; they are not used anywhere here.
* **What is measured:** the original model's responses to hybrid branch inputs. The reduced model is not evaluated.
* **Old code:** the frozen `scp/` package is imported read-only and its source tree (`f04e0ece…`) is bound as a
  dependency.
* **Isolation:** the new globs (`discarded_response_floor_gate/*.py`, `discarded_response_floor_gate/tests/*.py`) are
  disjoint from the Day-3 and follow-up source globs.

## 2. Hybrid inputs (column notation; the code uses the equivalent row form)

For recipient r, retained donor a and discarded donor i (all branch inputs at the final token):

* `x_minus_i = P x_r + (I − P) x_i`
* `x_plus_i = x_minus_i + P (x_a − x_r)`

**Mean-free.** This equals the centred form `mu + V z + q_i` and `mu + V(z+d) + q_i`, with `z = Vᵀ(x_r − mu)`,
`d = Vᵀ(x_a − x_r)` and `q_i = (I − P)(x_i − mu)`, for every mu (tested). So no input mean is needed.

**What is held fixed.** c (the recipient's prompt, residual bypass and every other position) and z, d are the same
for all donors of a group. The hook replaces **only** the MLP branch input at the final token. The residual the
MLP output is added to, and all other positions, are untouched, and later layers are recomputed by the original
network.

## 3. Observable, response, statistic, scale

**Observable.** J is the 100 single tokens "00".."99" (`scp.data.year_token_ids`). The id list has sha256
`5c0ba575…`, matches `manifests/split_manifest.json`, and its ids are distinct; m = 100.
* C_J(l) = l_J − mean_J(l_J).
* ‖v‖²_m = Σ_j v_j² / m.

**Response.** F(x) = C_J(final-position logits) with the branch input set to x; R_gk = F(x_plus_gk) − F(x_minus_gk).

**Statistic.** All **8** donors of the pool are **enumerated** (addendum §2). Q is the uniform empirical
distribution over the 8 discarded-donor activations of the group:

U_g = (1/8) Σ_k ‖R_gk − R̄_g‖²_m = (1/64) Σ_{i<j} ‖R_gi − R_gj‖²_m

* The denominator is 8, not 7.
* Both forms are computed, and they must agree to a relative 1e-10.
* Enumeration removes within-pool sampling noise only, not group or template uncertainty.
* There is no fallback to four draws.

**Scale** (development groups only, frozen afterwards):
* R_nat,g = F[branch = x_a] − F[branch = x_r];
* s² = Σ_g w_g ‖R_nat,g‖²_m, with w_g = 1/48 (frame 1/4 × template 1/6 × group 1/2).
* s is not recomputed on confirmatory data. Its uncertainty is not covered by any later interval.

**Development estimates.**
* L̂² = Σ_g w_g U_g / s² and L̂ = √L̂².
* Template values T_ft = (U_ft1 + U_ft2)/(2 s²).
* Frame moments: m_f = mean of the frame's six T values, v_f = their sample variance (ddof 1, **total** variance of
  the two-group template mean; no within-template subtraction), and m̄ = mean_f m_f.

**Information limit.** For any predictor h(u) with u = (c, z, d), or any stochastic predictor whose randomness is
independent of the realised q given u: E‖R − h‖²_m / s² ≥ L². This is the standard conditional-variance identity,
not a new theorem.
* No donor identity, slot index, seed or q-dependent state may be an input of h. They may appear in audit files.
* The bound concerns individual paired responses under this Q. It is not a bound for natural prompts or for other
  rank-192 representations.

## 4. Development design and prompts (outcome-blind)

**Frames.**
* 20 new frames G00..G19, each of the form `The {noun} … the year 17{yy} … the year 17`. They are distinct from the
  16 historical frames in `config/pilot_v1.json` and `followup/config/followup_v1.json` (checked).
* A seeded permutation (seed 20261031) assigns 4 **recipient** frames, 8 **discarded-donor** frames (disjoint from
  the recipient frames) and 8 unused frames.

**Nouns.** 130 new nouns, disjoint from the 102 historical nouns (checked).
* The seeded permutation assigns 24 recipient nouns (one per template), 48 donor nouns and 58 unused nouns.
* Donor nouns are disjoint from recipient nouns.

**Groups.** 4 recipient frames × 6 templates × 2 groups = **48 groups** (24 template clusters).
* For each frame, a seeded permutation π of the 6 direction × gap cells gives template j the cells π[j] and
  π[(j+3) mod 6]. So each cell appears exactly twice per frame and 8 times in total.
* Gap bins [10–19], [20–29], [30–40], up/down. These are reused unchanged from the registered `config/pilot_v1.json`
  `task.diff_bins`.

**Recipient and retained donor.**
* Same template (frame + noun); the recipient is at year r and the retained donor at year a.
* (r, a) is drawn uniformly (build seed 20261032) from the valid pairs in the cell.
* The four years of a template's two groups are distinct, so every prompt is unique.

**Discarded donors.** K = 8 per group, all at the recipient year r.
* Slot k uses donor frame k and donor noun `base[(g + 6k) mod 48]`. So every donor context (frame, noun) is used
  exactly once in the design, the 8 donor nouns within a group are distinct, and no prompt or pool is shared
  between groups.
* Each donor noun appears in 8 different groups under 8 different frames. This mild cross-group dependence is
  recorded as a limitation; unique nouns for 384 contexts were not available.

**Uniqueness and leakage.**
* All 480 prompt texts and token sequences are distinct (text and id sha256 recorded).
* No prompt equals any historical prompt in `manifests/*.csv`, `manifests_day3/*.csv` or
  `followup/manifests/*.csv`.
* Every development prompt, context and noun is excluded from any future confirmatory pool. Year values may
  repeat; that alone is not leakage.

**Tokenization.** Every prompt must pass `scp.data.check_prompt`: the start slot ` 17`+`YY`, and the final token
` 17`. A failure **aborts** prepare; no prompt is excluded or replaced.

**Not natural.** Q is **not** p(q | z, c), and z and q are not claimed to be independent. Hybrid inputs need not
lie on the natural activation manifold.

## 5. Support diagnostics (descriptive only, mean-free)

Recorded per (group, donor):
* retained distance ‖P(x_i − x_r)‖;
* discarded displacement ‖(I−P)(x_i − x_r)‖;
* uncentred discarded norms ‖(I−P)x_i‖ and ‖(I−P)x_r‖;
* ‖x_minus‖, ‖x_plus‖, ‖x_r‖, ‖x_a‖;
* pre-activation norms ‖x W_in + b‖ for x_minus, x_plus and x_r;
* ‖P(x_a − x_r)‖.

Limits:
* No stratum or donor subset is chosen on behavioural results. Q is never replaced by a better-looking subset.
* Distance is not extrapolated to zero.
* Norm or distance similarity is not a certificate of naturalness.

## 6. Numerical validity (deterministic; fixed before development)

**Rows per group.** One batch of 19 copies of the recipient prompt:
* 8 x_minus and 8 x_plus rows (16 scientific forwards);
* x_a and x_r rows (2 natural-scale forwards);
* a duplicate of x_minus_0 (1 numerical control).

**Development measurement INVALID** (scientific rejection is not implied) if any of:
* a non-finite output;
* the two U forms disagree beyond a relative 1e-10;
* the identity control (branch = x_r vs the unpatched recipient, raw J logits) exceeds a max abs of 1e-4;
* the duplicate control exceeds a max abs of 1e-4;
* s² ≤ 1e-8;
* s² < 1e6 × max_g ‖F_dup − F_minus0‖²_m.

No sample or group is dropped.

**Power moment guard** (addendum; deterministic):
* frame moments non-finite or negative → **invalid** measurement;
* m_f ≤ max(1e-10, 100 × noise_T), with noise_T = max_g ‖F_dup − F_minus0‖²_m / s² → **near_zero_floor**;
* v_f ≤ 1e-12 m_f² → **zero_variance**.

A near-zero floor or zero variance is a **valid** measurement for which the Gamma family is not estimable. The
outcome is then POWER_NOT_ESTABLISHED, with no invented variance.

## 7. Budget and compute accounting

**Forward counts** (counted separately per group): 16 scientific (768 for 48 groups), 2 natural-scale, 1
numerical control, and 10 activation extractions.

**Model-stage cap.**
1. A warmup forward on an **old calibration** prompt.
2. Time the first 3 groups.
3. Projected = elapsed + max(group time) × remaining groups × 1.25.
4. If the projection exceeds **1800 s** or the peak memory (max of RSS and the MPS driver allocation) exceeds
   **24 GiB**, stop and keep the partial outputs.

**Statistics.**
* At most 6 CPU threads, bounded memory chunks (100 simulated datasets × 10,000 bootstrap draws).
* A separate **1800 s** planning cap. If the full prescribed simulation does not finish, partial outputs are kept
  and labelled **POWER_NOT_ESTABLISHED (planning compute cap)**. Reduced simulations are never reported as the
  prescribed result.

## 8. Power planning (addendum §4–5; written before development)

**Candidates.** N ∈ {48, 96, 144, 192, 240} groups. Two groups per template, 4 frames, N/8 templates per frame. The
last 16 groups of the 256 cap are unused, to preserve balance. θ0 = 0.0025 (L0 = 0.05), θ1 = 0.0064 (L1 = 0.08).

**Scenarios.** For θ ∈ {θ0, θ1}, with r = θ/m̄ and μ_f = r·m_f, there are four:
* V_f = κ v_f, κ ∈ {1, 4};
* V_f = κ r² v_f, κ ∈ {1, 4}.

Template values are drawn from a Gamma with shape μ_f²/V_f and scale V_f/μ_f. There are no additive shifts.

**Simulation.** For each candidate × scenario × {null θ0, alternative θ1}: 2000 outer simulated datasets. Each is
analysed with the same 10,000-draw within-frame template bootstrap as the future confirmation:
* frame weights 1/4; a template carries both groups;
* exact multinomial counts;
* linear quantiles.

**Success.** The one-sided 95% lower bound on θ is > θ0, which is equivalent to the lower bound on L being > 0.05.

**Seeds.**
* Main seed 20261031, with deterministic substreams.
* Bootstrap counts: `(1, iN, frame)`. For each N they are reused across scenarios and hypotheses, matching a
  fixed-seed confirmatory bootstrap.
* Gamma draws: `(2, iN, iScenario, iHypothesis, frame)`, so the result does not depend on the chunk size.

**Monte Carlo uncertainty.** Two-sided Clopper–Pearson intervals at Bonferroni level 0.05/40 (5 N × 4 scenarios ×
2 hypotheses).

**Decision.**
* **POWER_PLAN_CANDIDATE:** the smallest N for which, in **all** scenarios, the power CP lower bound is ≥ 0.80 and
  the null false-positive CP upper bound is ≤ 0.075. The 0.075 bound is an operational calibration alarm for the
  nominal 5% bootstrap.
* **NO_GO_BUDGET:** every N has at least one scenario whose power CP upper bound is < 0.80.
* **POWER_NOT_ESTABLISHED:** otherwise, or if the moment guard fails or the compute cap is hit.
* The margin is never shrunk, and the model, rank and old experiment are never changed.

**Assumptions, not guarantees.** The Gamma shape, the fourfold variance sensitivity and transfer to new frames.
The plan is conditional on the development s.

## 9. Stages (`.venv/bin/python -m discarded_response_floor_gate <stage>`)

1. **`prepare`:**
   * runs with the tokenizer only;
   * writes `manifests/{groups,prompts}.csv` and `index.json` (write-once), `protected_manifest.json` (hashes of
     all old protected files, written once before any new run) and a prepare run with the projector checks, J
     hash and seed manifest.
2. **`verify-protected`:** re-hashes every protected file; fails on any change, removal or addition.
3. **`smoke`:** **old Day-1 calibration prompts only.** It uses 4 pilot pairs and same-year moment-pool donors.
   Gates:
   * identity ≤ 1e-4;
   * own-q algebra ≤ 1e-12;
   * same-q U ≤ 1e-10;
   * d = 0 response ≤ 1e-6;
   * earlier positions and residual bypass exactly unchanged;
   * MPS f32 vs CPU f64: R relative ≤ 1e-3 and U relative ≤ 1e-2;
   * U two-way relative ≤ 1e-10;
   * zero-control noise ≤ 1e-3 θ0 in L² units;
   * the patch changes the logits;
   * the budget projection is within the caps.

   It also records timings and memory.
4. **`register --smoke-run … --tests-log …`:** write-once `registration/DEV_REGISTRATION.json`. It binds:
   * the protocol, config, source, scp dependency, manifests, protected manifest, artifact hash and J hash;
   * the smoke run, which must be ok with the same bound files, and a config identical up to the label.

   The protected files must verify. The DRAFT label must be finalised first.
5. **`develop`:** re-verifies the registration and runs once. Outputs:
   * `activations.pt`;
   * `raw_J_logits.pt`: raw J logits [48 × 19 × 100], centred R and R_nat;
   * `groups_U.csv`, `controls.csv`, `support.csv`;
   * `development.json`: s, L̂², L̂, the template T values, frame moments, validity, and a **descriptive**
     development bootstrap (not a confirmatory decision).
6. **`power --dev-run …`:** `power.json` and `power_rows.csv`.
7. **`report --dev-run … [--power-run …]`:** re-aggregates with the separate implementation (`aggregate.py`, same
   author; not an independent audit) and re-verifies the protected files.

Every run directory is new, and failed runs are kept. Each run records bound hashes, the environment, a snapshot,
a log, raw outputs, output sha256 values and start/end times.

## 10. Future confirmatory rule (draft only; not run here)

**Bootstrap.** A fixed-seed 10,000-draw template-within-frame bootstrap with fixed frame weights. Templates are
resampled within frames, and each carries all its groups and every group's 8 donors.

**Reporting.** The lower and upper bounds are separate one-sided 95% bounds.

**Decision.**
* L lower > 0.05 → the floor is supported for this interface and distribution.
* L upper ≤ 0.05 → the target-size obstruction is ruled out.
* Otherwise → INCONCLUSIVE.
* Numerical, sampling or protocol invalidity → EVALUATION_INVALID (not a scientific rejection).

**Confirmatory requirements.** Fresh pools, disjoint from every development prompt, context and noun. The sample
size comes from the development power report, and confirmation needs separate approval and a frozen protocol.

## 11. Interpretation limits

* A large L limits only q-blind **individual** response prediction for this P, rank, interface, observable and Q.
* A small L does not show learnability, data sufficiency, optimality of PCA + bias, or any closure.
* A stochastic closure may reproduce the response **distribution** despite the paired floor.
* No naturalness, independence, other-representation, circuit, WeightWatcher k* or Wilsonian RG claim is made.
* The variance identity, tower property and nested projectors are not new RG methods.
* There is no novelty guarantee.
