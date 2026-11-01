# Discarded-state response floor gate — development protocol v3: TECHNICAL REMEASUREMENT (drfg-dev-v3-techremeasure)

## V3. Technical remeasurement (2026-10-31; supersedes the v2 extraction layout only)

**Status of v2.** The registered v2 development run `20261031T103329Z_gate_develop` is **EVALUATION_INVALID** and stays
so:
* 15 of 48 identity controls exceeded the fixed 1e-4, with a maximum of 2.13623046875e-4;
* thresholds are not relaxed, nothing is reclassified, and failures are not dropped.

v2 raw outputs exist, including s² = 1.0934 and the U ranges. L and the frame moments being null does **not** mean
the design is unexposed or uncontaminated. v3 is therefore a **development technical remeasurement**: it is neither
a fresh nor a blind measurement, and it is not confirmatory.

**Evidence** (coordinator, old Day-1 calibration only; `notes/gate_numerical_diagnostic/`):

*Run `20261031T104432Z_diag_v2`, MPS float32* — 132 mixed cases (108 padded, 24 unpadded; rows 0, 24, 47; pads 0–4):
* same-layout hook identity was exactly 0 for every row;
* the development-style cross-layout self-control was 0 for unpadded and pad-1 cases, up to 1.2207e-4 at pads 2/3
  and up to 1.3733e-4 at pad 4;
* its injected-input part reached 1.0681e-4 and its no-hook layout part 1.2207e-4;
* input drift was 1.5e-6.

*Same run, CPU float64* — 33 cases: every self, cross-layout and input-drift control was 0, and the batch-1 vs 19
GEMM difference was about 3.7e-13.

*Projector reconstruction audit* — every one of the 480 stored activations, reconstructed as own-q P·x + (I−P)·x in
CPU float64, had error 3.6e-15; after the float32 cast all 480 were bit-identical.

**Interpretation.** These results reproduce a float32 precision and layout sensitivity in the tested cases. They do
not prove that every bug is absent: padding- or row-conditional mechanisms outside the tested layouts are not
excluded. The first diagnostic (pad 2 at row 0, plus pad 0) missed the other pad and row conditions. Both
diagnostic versions and their runs are preserved.

**Decision** (actual reviews): GPT Pro (observed 10:39:53 UTC) supports a bounded technical repair; Fable and
Antigravity support the conditional remedy. Their stronger claims ("no contamination", universal bug exclusion) are
**not** adopted.

**The repair: canonical no-padding layout.**
* Every activation extraction is one forward of **one prompt replicated 19 times**, with no padding, no mixing of
  prompts, every mask 1 and the last index at len − 1 (checked in code).
* The recorded branch input x and the **unpatched reference logits** are taken from **row 17**.
* Row 17 is also the identity row (`nat_r`) of the 19-row group forward, which is the same physical layout (19
  copies of the recipient). The identity control therefore compares the same row of the same layout.
* Every retained and discarded donor gets its own canonical forward, so each branch input's provenance is its own
  prompt's canonical row 17.
* There is no batch-1 extraction, because the lm_head batch shape differs. The dtype is fixed at float32 on MPS.

**Unchanged by the repair.**
* model, projector, J and estimand;
* the 48 groups and 480 prompts and their Q supports. `manifests/` is a **byte-identical copy** of the v2
  `manifests_v2/`, verified against the v2 registration; there is no resampling and no new nouns, rank or model;
* the estimator, every numerical and scientific threshold, and the smoke gates;
* power planning: 2000 × 10,000, model-conditional Gamma, broader power not established;
* budgets: 1800 s model stage, 1800 s statistics, 24 GiB, at most 6 CPU threads.

**Compute accounting** (logical vs computational):
* extraction: 480 logical prompts → 480 canonical forwards → 9120 computational rows;
* group stage: 48 forwards → 912 rows (768 scientific, 96 scale, 48 control);
* the time and memory caps are checked every 48 extracted prompts, after extraction and after every group.

**Smoke (old calibration only).**
* Recipients alternate between lengths 12 and 14, and same-year donors of both lengths occur. These are the length
  mismatches that were padded in the v2 mixed extraction.
* The smoke uses the canonical path, keeps every old gate and adds a `canonical_layout` gate.
* It keeps the CPU float64 spot comparison of R and U.
* It adds a **descriptive** legacy contrast: v2-style mixed, right-padded extraction of the same prompts with extra
  pads 0–4.
* The remeasurement runtime is projected from the canonical path before any development forward.

**Rules.**
* If the existing gates cannot be met within the budget, **STOP**. Nothing is relaxed.
* Remeasure **all 48** groups, including the scale and the controls.

**Preservation.**
* All v2 bytes are unchanged and verified: the source tree `74690c53…`, configs, protocols, manifests_v2, the
  registration, and the prepare, smoke and failed-develop runs with their outputs (`v2_preservation` in
  `config/gate_dev_v3.json`, `verify-v2`).
* The v3 protection manifest covers all of `discarded_response_floor_gate/**`, the older protected files, and both
  diagnostic versions and runs.
* v3 is a self-contained copy of the v2 modules. It imports `scp` read-only and never imports the v2 package.

**Paths (v3).** Package `discarded_response_floor_gate_v3/`, config `config/gate_dev_v3.json`, manifests
`manifests/` (copy), registration `registration/DEV_REGISTRATION_v3.json`, protocol `DEV_PROTOCOL_v3.md`.

The rest of this document is the v2 protocol, which stays in force except for the extraction layout and paths
above. Where it says "extraction batches of 48", read "canonical 19-row forward per prompt".

---


## 0. Amendment v2 (2026-10-31, before any development outcome)

**What changed.** Following GPT Pro's addendum (`notes/gpt_pro_donor_assignment_addendum_2026-10-31.txt`, option B),
the v1 donor-noun offsets 6k are replaced by **D = (0, 1, 4, 9, 15, 22, 32, 34)**:
* group g, slot k uses noun `base[(g + D[k]) mod 48]` and donor frame k;
* the frozen order of the 48 donor nouns and the 8 donor frames is unchanged.

**Why.** The v1 offsets form the subgroup 6·Z48. That gave only 6 distinct donor-noun sets, and 168 group pairs
shared all 8 donor nouns.

**Properties of D** (checked in code at prepare and recorded in `manifests_v2/index.json`):
* the 28 positive differences are distinct, so every nonzero modular difference occurs at most twice;
* every group pair shares at most 2 donor nouns, with a mean of 56/47;
* every (frame, noun) context is used once, each noun 8 times, and the 8 nouns of a group are distinct.

The coordinator independently checked the 28 differences and the multiplicity of at most 2.

**Unchanged:**
* recipients, retained donors, frames, the noun lists and their order, cells and years;
* global noun and frame frequencies;
* the 8-donor enumeration, the estimator, thresholds, power method, budgets and smoke gates.

Only each group's Q_g changes. `manifests_v2/` must reproduce the recipients, retained donors, cells, years and donor
frames of the v1 manifest exactly (hash-checked).

**v1 records are preserved and not used:** `config/gate_dev_v1.json`, `DEV_PROTOCOL.md`, `manifests/`, the prepare run
`20261031T095918Z_gate_prepare` and the smoke run `20261031T101947Z_gate_smoke`. Their hashes are in
`config/gate_dev_v2.json` → `amendment`.

**Acceptance** (exhaustive, outcome-blind; any failure blocks registration, with no relaxation and no alternative
order, offset or seed search):
* all 1128 group pairs have overlap ≤ 2, and the total is 1344 = 48·C(8, 2);
* recipient and donor nouns and frames are lexically separate;
* prompt and token uniqueness, and historical leakage 0;
* format and template agreement and tokenization. This is grammar in the format sense only: semantic naturalness
  is **not** checked, and odd prompts remain a limitation;
* frame, template and stratum counts are unchanged.

The group-level [48 × 48] and template-level [24 × 24] noun-overlap matrices are saved in `manifests_v2/`.

**Interpretation.**
* Unique contexts and low overlap do **not** make lexical effects independent and do not establish bootstrap
  coverage.
* The total of 1344 shared-noun incidences is fixed for every offset set.
* The independent-Gamma simulations remain conditional planning scenarios. They do not validate coverage under a
  random-donor-lexicon population. Neither D nor κ = 4 proves coverage.
* Broader power claims that are unsupported remain POWER_NOT_ESTABLISHED.
* Only the registered 48 development groups are run. Confirmation needs a separately approved donor-sampling and
  uncertainty design.

**Paths (v2).** Config `config/gate_dev_v2.json`, manifests `manifests_v2/`, registration
`registration/DEV_REGISTRATION_v2.json`, protocol `DEV_PROTOCOL_v2.md`. The smoke must be rerun under v2, because
register requires the smoke to be bound to the same config file, manifest directory, protocol and source.

The rest of this document is the v1 protocol text, with the v2 paths and offsets substituted.

**Sources.**
* `notes/research_direction_rg_2026-10-31.md`: the selected hypothesis and the reviewer debate.
* `notes/gpt_pro_opus55_discarded_state_gate_prompt_2026-10-31.txt`: GPT Pro's implementation prompt.
* `notes/gpt_pro_gate_addendum_2026-10-31.txt`: GPT Pro's addendum. It replaces the projector, donor-sampling and
  power-planning sections of the prompt, and includes the coordinator's resource clarifications.

**Machine-readable counterpart:** `config/gate_dev_v2.json`. If the two disagree, that is a blocker to report, not
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
* The seeded permutation assigns 24 recipient nouns, 48 donor nouns and 58 unused nouns.
* Each recipient noun is used by exactly one template, so the 24 recipient nouns are globally unique.
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
* Slot k uses donor frame k and donor noun `base[(g + D[k]) mod 48]`, with D = (0, 1, 4, 9, 15, 22, 32, 34) (§0). So
  every donor context (frame, noun) is used exactly once in the design, the 8 donor nouns within a group are
  distinct, and no prompt or pool is shared between groups. Any two groups share at most 2 donor nouns.
* Each donor noun appears in 8 different groups under 8 different frames. **Donor contexts are disjoint, but donor
  nouns are reused** under this finite assignment (total 1344 shared-noun pair incidences). That lexical reuse is a cross-group dependence that the
  template bootstrap does not model. It remains a limitation of any interval; unique nouns for 384 contexts were not
  available.

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
* **Clock.** It starts after model load and an untimed warmup on old calibration pair 0: one extraction forward and
  one 19-row group-shaped forward whose branches equal the prompt's own input. Both are excluded from the clock.
  The clock covers extraction and all group forwards.
* **Hard caps.** Elapsed time ≤ **1800 s** and peak memory ≤ **24 GiB** (the larger of RSS and the MPS driver
  allocation). They are checked **after extraction and after every group**.
* **Projection.** After the 3rd group: elapsed + max(group time) × remaining groups × 1.25 must be ≤ 1800 s.
* **On a stop.** The run stops **before the next forward**; a running forward cannot be pre-empted. Activations,
  partial raw logits, controls and support rows are kept, and the stop status (time cap, memory cap or projection)
  is recorded.
* **After the final group.** A cap exceeded only after the final group is recorded, not hidden; there is no
  further forward to stop.
* **Smoke.** The smoke uses the same warmup and then exactly 3 timed groups. Pair 0 is never a smoke group.

**Statistics.**
* At most 6 CPU threads.
* Simulated datasets are processed **one at a time**. Each dataset's own 10,000-draw bootstrap holds one
  [10,000 × n_t] int16 index array per frame in memory (§8).
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
analysed with **its own independent** 10,000-draw within-frame template bootstrap. It uses the **same resampler**
as the confirmatory analysis (`bootstrap.resample_theta`, also used by `template_bootstrap`):
* templates are drawn with replacement within each frame;
* frame weights are 1/4, and a template carries both groups;
* exact percentiles with linear interpolation.

**Success.** The one-sided 95% lower bound on θ is > θ0, which is equivalent to the lower bound on L being > 0.05.

**Seeds.** Main seed 20261031. Dataset j of (N, scenario, hypothesis) uses:
* data stream `(2, iN, iS, iH, j)`;
* a separate bootstrap stream `(3, iN, iS, iH, j)`.

**Properties.**
* The power estimate averages over the bootstrap randomisation, as the future confirmatory analysis will with its
  own fixed seed.
* Results are reproducible and do not depend on how datasets are processed or restarted (tested).
* Memory per dataset is one [10,000 × n_t] int16 index array per frame.
* Measured on synthetic inputs, the prescribed simulation projects to about 210 s on this machine, under the
  1800 s cap.

**Monte Carlo uncertainty.** Two-sided Clopper–Pearson intervals at Bonferroni level 0.05/40 (5 N × 4 scenarios ×
2 hypotheses).

**Decision.**
* **POWER_PLAN_CANDIDATE:** the smallest N for which, in **all** scenarios, the power CP lower bound is ≥ 0.80 and
  the null false-positive CP upper bound is ≤ 0.075. The 0.075 bound is an operational calibration alarm for the
  nominal 5% bootstrap.
* **NO_GO_BUDGET:** every N has at least one scenario whose power CP upper bound is < 0.80.
* **POWER_NOT_ESTABLISHED:** otherwise, or if:
  * the moment guard reports a valid but not estimable moment (near-zero floor or zero variance);
  * the compute cap is hit;
  * the power table is incomplete: all 5 × 4 × 2 prescribed rates must be present exactly once with n = 2000
    before any positive or NO_GO decision.
* **EVALUATION_INVALID:** non-finite or negative measured moments.
* The margin is never shrunk, and the model, rank and old experiment are never changed.

**Assumptions, not guarantees.** The Gamma shape, the fourfold variance sensitivity and transfer to new frames.
The plan is conditional on the development s.

**Model-based calibration.** The simulated null false-positive rate is computed under that Gamma model with
independent templates. It is **not** a guarantee of bootstrap coverage or false-positive control for the fixed
finite donor assignment, with its reused donor nouns, or for natural language.

## 9. Stages (`.venv/bin/python -m discarded_response_floor_gate <stage>`)

1. **`prepare`:**
   * runs with the tokenizer only;
   * writes `manifests_v2/{groups,prompts,group_noun_overlap,template_noun_overlap}.csv` and `index.json`
     (write-once; the acceptance checks and the v1 reference comparison must pass), `protected_manifest.json` (hashes of
     all old protected files, written once before any new run) and a prepare run with the projector checks, J
     hash and seed manifest.
2. **`verify-protected`:** re-hashes every protected file; fails on any change, removal or addition.
3. **`smoke`:** **old Day-1 calibration prompts only.** After the warmup, it uses 3 timed pilot pairs and
   same-year moment-pool donors.
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
4. **`register --smoke-run … --tests-log …`:** write-once `registration/DEV_REGISTRATION_v2.json`. It binds:
   * the protocol path and hash, the config path and hash, source, scp dependency, the manifest directory and its
     hashes, the protected manifest, the artifact hash and the J hash;
   * the smoke run, which must be ok and made under the same config file, manifest directory and bound files, with
     a config identical up to the label.

   The protected files must verify. The DRAFT label must be finalised first.
5. **`develop`:** re-verifies the registration and runs once. Outputs:
   * `activations.pt`;
   * `raw_J_logits.pt`: raw J logits [48 × 19 × 100], centred R and R_nat;
   * `groups_U.csv`, `controls.csv`, `support.csv`;
   * `development.json`: s, L̂², L̂, the template T values, frame moments, validity, and a **descriptive**
     development bootstrap (not a confirmatory decision).
6. **`power --dev-run …`:** `power.json` and `power_rows.csv`.
7. **`report --dev-run … [--power-run …]`:** re-aggregates with `aggregate.py` and re-verifies the protected
   files. `aggregate.py` is a second aggregation implementation **by the same author**; it is **not** an
   independent researcher audit.

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

**Scope of the bootstrap inference.** The bootstrap resamples templates within the fixed frames; donors and their
construction are held fixed. Stating this scope does **not** establish that the intervals have their nominal
coverage.
* Donor lexical reuse (disjoint contexts, nouns reused under the finite D assignment, at most 2 shared nouns per
  group pair) remains an unmodelled dependence limitation.
* The power plan's false-positive calibration is model-based (Gamma, independent templates). It is not a guarantee
  for this finite donor assignment.
* Nothing here is inference about transfer to a natural-language population of frames, nouns or prompts.

## 11. Interpretation limits

* A large L limits only q-blind **individual** response prediction for this P, rank, interface, observable and Q.
* A small L does not show learnability, data sufficiency, optimality of PCA + bias, or any closure.
* A stochastic closure may reproduce the response **distribution** despite the paired floor.
* No naturalness, independence, other-representation, circuit, WeightWatcher k* or Wilsonian RG claim is made.
* The variance identity, tower property and nested projectors are not new RG methods.
* There is no novelty guarantee.
