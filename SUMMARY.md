# SUMMARY — pilot-v1 (+ amendment v1.1), day 1 (2026-10-30)

Day 1 is complete; Days 2–3 have not started. Final Day-1 reviewer responses (Codex, GPT Pro, Antigravity: joint
conditional GO, no blocking objection), limits and the plan are in **`notes/three_model_review_day1.md`**.
Plan: Day 2 = norm/pre-activation-matched geometric controls at the existing three cuts; Day 3 = validation
quality matching, freezing the rules before test, and held-out confirmation if appropriate.

**Label: EXPLORATORY PILOT, calibration split only.** Nothing here is confirmatory. ε is not frozen;
validation and test splits were not run (test is sealed in code). No rank, threshold or rule was chosen from
pilot output. Numbers below are first measurements for Codex / GPT Pro / Antigravity review, not findings.

## 1. Environment

Apple M2 Max, 32 GB, macOS 26.6.2. Python 3.13.0 (python.org framework) in project `.venv`;
torch 2.15.0 (MPS available/built), transformers 5.22.1, numpy 2.5.4, scipy 1.18.1, tokenizers 0.23.3,
safetensors 0.8.0, huggingface_hub 2.4.0, pytest 9.2.0 (`requirements.txt`, full `requirements.lock.txt`,
`env/hardware_and_versions.txt`). No existing torch+transformers install was found, so these came from PyPI
into `.venv`; GPT-2 small came from `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`
into `.cache/hf` (safetensors SHA-256 `248dfc39…3a707`). Model: MPS float32. SVD and moments: CPU float64
(recorded in each manifest). No device fallback happened.

## 2. What was built

* `PROTOCOL.md` (pilot-v1), `config/pilot_v1.json`, seeded split manifests (`manifests/`), and
  `registration/REGISTRATION.json` written at **2026-10-30T15:36:37Z, before any pilot measurement**
  (protocol `88cb9abf…`, config `e8f7165d…`, manifests and source tree hashed).
* Code `scp/` + `run.py` stages `splits | register | quadcheck | smoke | bench | pilot`, `scp/report.py`
  (markdown tables). Every run dir has `run_manifest.json` (seed, model revision + weights hash, dtype/device,
  SVD device, package versions, config/protocol/source/manifest hashes, per-stage and per-variant walltimes),
  `log.txt`, and a `snapshot/` copy of the exact code/config/protocol/manifests/registration.
* Pilot design as registered: block 9, `k ∈ {96, 192, 384}`, compensations `zero | bias | gauss | gauss_wo`,
  scopes `last_token | full_layer`, 192 pairs (2 directions × 3 diff bins × 32), donors `natural | same_year |
  wrong_donor` (+ `same_example` identity), subspaces `R | B | RB`.

### Fixes made before registration (all after review or failing tests; nothing tuned on results)

1. `check_prompt` scanned one token too few, so every prompt was rejected → fixed; 0 prompts excluded.
2. Codex review: PD now uses `Σ_{j>r} − Σ_{j≤r}` (Hanna et al.), not `j<r`.
3. Codex review: the protocol claim that local final-position responses are identical across scopes was wrong
   under our design (scopes use different moment pools). Text corrected; every row now has `moment_pool`.
4. Gauss–Hermite with 32 nodes is **not** converged for large σ (σ=4: 4.8e-3). The threshold was not
   loosened. `quadcheck` on the calibration pool's real `(z+μ, σ)` (σ_max observed 1.40) plus a grid
   `m∈[−20,20]`, `σ∈[0,2]` picked the smallest node count with error < 1e-7 vs 512 nodes: **64**
   (grid error 1.1e-8; real data ≤ 8e-13). Config set to 64 before registration. Accuracy is only claimed for
   σ ≤ 2.0, and a reduction with larger σ is refused at run time. numpy's `hermegauss` overflows at
   ~1000 nodes, so the rule now comes from `scipy.special.roots_hermitenorm`.
5. Codex review: the locality test now checks real non-final positions only (right-pad positions may attend to
   the final token and are never read). The test helper unwraps `SpectralMLP` to reach `c_fc`.
6. MPS has no float64. Casts now go through the CPU first.
7. Codex review: the 900 s budget is enforced. The run stops before a variant if elapsed > 780 s, writes a
   checkpoint (`per_pair_rows.csv` appended, `partial_results.json`) after every variant, and every run gets a
   snapshot.

Failed or early runs are kept: `runs/20261030T153332Z_quadcheck` (failed: MPS float64 cast) and the first
pytest log `logs/pytest_20261030T153120Z.log` (5 failures).

### After the first pilot: amendment v1.1 (`PROTOCOL_AMENDMENT_v1.1.md`, `registration/AMENDMENT_v1.1.json`)

These respond to Codex's protocol and code review. `PROTOCOL.md` and the config are unchanged (hash-checked). No
measured number, rank, threshold or primary candidate changed. Contents:
* J is the logit **score set**. A natural-continuation mask is recorded in `metadata/continuation_mask.json`:
  only `j=00` is invalid, in every frame, because `"1700"` is the single token `' 1700'`.
* No replication claim for arXiv:2305.00586.
* Explicit TODO controls (§6).
* `Reduction` is a functional rank-k audit (dense `P_R` + original `Wc`), **not** compression; the benchmark
  timings are harness timings.
* float64 metrics refuse non-CPU tensors.
* `check_prompt` boundary is explicit, with positive/negative tests.

## 3. Numerical checks: 30/30 tests pass (`logs/pytest_final_v1.1_20261030T154404Z.log`)

| check | result |
|---|---|
| gelu_new == HF `NewGELUActivation` (fp32 and fp64) | bitwise equal |
| Conv1D orientation: `Wc` [768,3072], `W_in = Wc.T`; `W_in P_R == U_kΣ_kV_kᵀ`; ‖W_in − W_in P_R‖² = Σ_{i>k} s_i² | atol 1e-10 / rtol 1e-9; same subspace from SVD of `Wc` (1e-8) |
| `P_R + P_B = I`, idempotent, symmetric, `P_R P_B = 0`, `tr P_R = k` | ≤1e-10 |
| full rank (k=768): every method and scope reproduces the original logits | ≤2e-4 |
| same-example identity (δ=0) | max \|Δlogit\| = **0.0** (test and real pilot run) |
| residual bypass invariant: `ln_2` input unchanged; Δblock-out = ΔMLP-out; other real positions unchanged | exact 0 / ≤1e-4 |
| `R+B` patch = full donor branch (and = independent pre-hook substitution) | MLP-out rel. diff 3.6e-7 on 192 pairs (smoke) |
| σ=0 Gaussian = exact gelu_new | bitwise equal |
| quadrature convergence (configured 64 nodes, σ≤2) and an independent 4M-sample Monte Carlo | <1e-7; within 5 SE |
| reduced-model B invariance (`Δ_B` = 0 for all methods, scopes and ranks) | local ≤1e-4, logits ≤1e-3 (in the pilot ‖W_in P_R δ_B‖ ≈ 4e-7) |
| `x_rec` identical in reduced and original models (runtime assert) | 0.0 |
| split disjointness and leakage (frames/nouns/years; moment pool shares no template or prompt with pairs), seeded determinism, pair design | pass |
| hook/wrapper removal (also on exception; no nesting); module restored; logits bitwise equal | pass |
| MPS vs CPU (8 prompts): original / reduced gauss full-layer logits | max \|Δ\| 6.0e-4 / 5.5e-4 |
| CLI refuses `--split test` (no frozen ε) and layers other than 9/10 | pass |
| tokenization: `"…year 1732 to the year 17"` accepted; merged `' 1700'`, wrong YY, missing slots rejected | pass |
| natural-continuation mask: only `j=00` invalid, all 6 frames | pass |
| PD counts `j = r` as wrong (all mass on `r` → PD = −1) | pass |
| float64 metrics refuse MPS tensors | pass |

## 4. Original behaviour, positive control, benchmark

`runs/20261030T153643Z_smoke/smoke.json` (also recomputed as the gate inside the pilot):

* Original GPT-2 on 192 calibration recipients: mean PD **0.915 [0.904, 0.924]**, PD>0 on 100%; top-1 is a
  two-digit year token on 100% and is > r on 100%.
* Full MLP9-branch donor transfer (`S = RB`, natural donor): **T = 1.72 [1.66, 1.79]** (template-cluster CI
  [1.65, 1.80], 18 clusters). Same-year control: 0.004 [−0.018, 0.026]. Wrong-donor: 0.785 [0.652, 0.916];
  that donor's own year shift partly aligns with `v_rd`, so it is not a zero-expectation null.
  **Gate: valid** (pre-specified rule). This gate
  only shows that the full-branch transfer moves T far more than a same-year context change. It does **not**
  establish content specificity beyond the missing controls (§6).
* Re-run after v1.1 (`runs/20261030T154234Z_smoke`): PD and T are bitwise-identical to the first smoke.
  PD over natural continuations (j≠00) is 0.915 [0.904, 0.924]; the largest per-prompt difference from full-J PD
  is 0.0017.
* Benchmark (`runs/20261030T153655Z_bench/bench.json`, sequence length ≤14):

| pairs | orig fwd | orig patched | gauss full-layer patched | gauss last-token patched | MPS driver mem | max RSS |
|---|---|---|---|---|---|---|
| 16 | 11.8 ms | 11.4 ms | 21.5 ms | 13.7 ms | 2.21 GB | 1.06 GB |
| 32 | 17.6 ms | 17.6 ms | 39.5 ms | 20.5 ms | 2.21 GB | 1.06 GB |

## 5. First pilot measurements (EXPLORATORY): `runs/20261030T153707Z_pilot`

Status `ok`. Total walltime **54.8 s** of the 900 s budget (24 variants at 1.1–3.0 s each; bootstrap
aggregation 10.6 s); B=2000. Retained Frobenius energy of `W_in`: k=96 → 0.361, 192 → 0.553, 384 → 0.801.
All tables are in `summary_tables.md` in that directory.

**(a) Where the natural response lives (original model, natural donor):**

| k | T_R | T_B | T_RB | I_T mean [CI] | RMS‖W_in δ_R‖ | RMS‖W_in δ_B‖ |
|---|---|---|---|---|---|---|
| 96 | 0.185 | 1.52 | 1.72 | 0.016 [0.003, 0.030] | 8.85 | 18.2 |
| 192 | 0.429 | 1.24 | 1.72 | 0.056 [0.037, 0.075] | 12.5 | 16.0 |
| 384 | 0.862 | 0.768 | 1.72 | 0.095 [0.068, 0.122] | 16.5 | 11.7 |

The B-mediated (discarded-subspace) part of the year-swap response is sizable at every cut. It exceeds the
R part only at k=96 and k=192; at k=384, R (0.862) > B (0.768). In every reduced model `Δ_B ≡ 0` (max |T_B| over
all reduced rows: 1.9e-5): the reduced model's response to a pure-B patch is structurally zero, because the
compensations are constant moment terms that do not depend on the discarded input (their output still depends on
`z`). This RB loss is real and meaningful for the restoration audit. It is **not** a universal irreducible error
floor for natural RB, since an R over-response can offset it (§5c). The interaction `I` is small but not zero.

**(b) Unintervened quality (KL(orig‖M), final token, mean):**

| k | zero LT / FL | bias LT / FL | gauss LT / FL | gauss_wo LT / FL |
|---|---|---|---|---|
| 96 | 0.326 / 0.420 | 0.234 / 0.287 | 0.218 / 0.261 | 0.277 / 0.290 |
| 192 | 0.171 / 0.207 | 0.155 / 0.150 | 0.149 / 0.141 | 0.151 / 0.150 |
| 384 | 0.041 / 0.048 | 0.061 / 0.042 | 0.061 / 0.042 | 0.044 / 0.045 |

At k=384 in `last_token` scope, the final-token-calibrated `bias`/`gauss` have *higher* KL than `zero`.
PD falls in every reduced model (for example k=96 gauss LT: −0.143 [−0.161, −0.128]).

Both views are reported side by side: **(c) RB = end-to-end restoration audit** (the registered primary
candidate, unchanged) and **(d) R = secondary compensation-fidelity readout**. The R advantage below is not
moved into the primary.

**(c) Candidate primary statistic (natural donor, S=RB, RMS e_T; T_orig mean 1.72):**

| scope | k | RMS e_T gauss | RMS e_T bias | gauss − bias [pair CI] | [cluster CI] |
|---|---|---|---|---|---|
| last_token (candidate) | 96 | 1.59 | 1.57 | +0.019 [0.017, 0.021] | [0.017, 0.021] |
| last_token (candidate) | 192 | 1.30 | 1.27 | +0.035 [0.031, 0.038] | [0.031, 0.038] |
| last_token (candidate) | 384 | 0.833 | 0.800 | +0.033 [0.029, 0.037] | [0.029, 0.037] |
| full_layer | 96 | 1.62 | 1.60 | +0.019 [0.016, 0.022] | [0.015, 0.022] |
| full_layer | 192 | 1.29 | 1.28 | +0.006 [0.002, 0.010] | [0.001, 0.010] |
| full_layer | 384 | 0.836 | 0.839 | −0.003 [−0.006, 0.001] | [−0.006, 0.000] |

Errors are almost entirely a signed shortfall (mean e_T ≈ −RMS), close to the missing `T_B`. Differences between
compensations are second order (≤0.07 among all four). Gauss−bias RB differences are ≤2% of the original RB RMS,
and the full_layer k=384 CI includes 0. A higher gauss RMS on this statistic does not show that the Gaussian
damages the preserved mechanism. No universal Gaussian harm or improvement is claimed.

**Exact decomposition, checked on the existing rows (no new experiment).** Because `Δ_B^M ≡ 0`, every pair
satisfies `e_RB^M = e_R^M − (T_B^orig + I^orig)` (max residual 1.9e-5 over all reduced rows). The offset
`T_B^orig + I^orig` is the same for every method. So a method that **over-responds in R** (positive `e_R`, as `bias`
does) partly cancels the structural B loss and gets a lower RB RMS. Example, k=192 LT: bias e_R mean +0.092
vs gauss +0.058, and the RB gap is +0.035. The RB ranking of `bias` over `gauss` is therefore partly an
error-cancellation artefact of this statistic. *Why* `bias` over-responds in R (e.g. a change of the GELU slope
gain) is **not measured**; it is a possible explanation only.

**(d) Response through the retained subspace only (natural donor, S=R, `last_token`):**

| k | RMS e_T zero / bias / gauss / gauss_wo | local RMS err ÷ RMS orig response, zero / bias / gauss / gauss_wo |
|---|---|---|
| 96 | 0.091 / 0.080 / **0.068** / 0.096 | 0.67 / 0.38 / **0.36** / 0.66 |
| 192 | 0.137 / 0.131 / **0.107** / 0.116 | 0.53 / 0.34 / **0.33** / 0.52 |
| 384 | 0.146 / 0.163 / **0.142** / 0.130 | 0.32 / 0.22 / **0.21** / 0.31 |

For the R-mediated response, calibrated moments cut the local error roughly in half compared with `zero`, and
`gauss` ≤ `bias` (secondary readout, exploratory). These gaps shrink in `full_layer` scope (all-token moments).
`bias` has a positive mean e_T (over-response) at every k. No mechanism (for example a reduced Gaussian slope
gain) is established by these numbers.

**(e) Controls.** Same-year T stays ≈ 0 in every model, but its effect norms are not zero: original RMS‖Δ local‖
is 16.3, against 48.2 for natural. Effect norms and pre-activation energies for each donor, S and k are in
section "Control effect norm vs pre-activation energy" of `summary_tables.md`.

### Points for reviewers (not acted on: these are decisions for the reviewers)

* The candidate primary statistic (natural donor, S=RB) is dominated by the structural loss of the B-mediated
  response; the reduced model's response to that part is structurally zero (constant moment compensations that
  do not depend on the discarded input). This loss is real and meaningful for the restoration audit, but by the
  exact decomposition in §5c the statistic can be offset by R over-response, so it alone does not rank the
  methods' R fidelity. **Joint decision (all three reviewers and Claude):** RB stays the
  primary candidate (end-to-end restoration audit); R is the secondary compensation-fidelity readout, reported
  with its signed bias. ε and the specific confirmatory rank are **still not frozen**. The registered protocol is
  unchanged.
* `last_token` and `full_layer` use different moment pools, so the scope comparison is confounded with the
  moment pool for `bias`/`gauss` (documented in PROTOCOL §4).
* Clusters: 18 (frame, noun) templates but only **2 frames**, so cluster CIs ≈ pair CIs and do not represent
  between-frame uncertainty.
* The same 192 pairs serve as the gate and as the measurements (both calibration). T can exceed 1:
  its scale is logit units of the indicator projection.

## 6. Not done yet

* Controls (amendment A4): norm-matched wrong donor, PCA-subspace cut, equal-dimension random-subspace split,
  whitened SVD, resample ablation. Without them no content-specificity claim is made.
* Compression: not measured or claimed (the audit stores dense `P_R` + the original `Wc`).
* Mechanism of the R over-response under `bias` / the Gaussian slope-gain explanation: not measured.
* WeightWatcher `k*` (TODO; no cutoff substituted). Ranks used: the pre-specified 96/192/384 only.
* General-text moment pool (TODO, kept separate by design).
* Resample-ablation diagnostic reference (`E[loss]` vs `loss(E[h])`): not run.
* Validation split: not run. ε: not frozen (only reference scales recorded: T_orig natural RB mean 1.72,
  SD 0.50, RMS 1.79). Test split: sealed.
* Block 10 backup: not needed (block-9 gate valid), not run.
* PD under intervention: not measured. Repeat-run determinism on MPS was checked only for the smoke stage
  (two runs, bitwise-identical PD/T).

## 7. Exact reproduction

```bash
cd /Users/emre/Documents/Codex/ML_Projects/spectral_causal_pilot
.venv/bin/python -m pytest -v                 # 26 tests
.venv/bin/python run.py splits                # regenerates manifests; must hash-match registration
.venv/bin/python run.py quadcheck             # node-count convergence on calibration moments
.venv/bin/python run.py smoke                 # behaviour, positive control, identities, MPS/CPU parity
.venv/bin/python run.py bench                 # 16/32-pair timing + memory
.venv/bin/python run.py pilot                 # exploratory pilot (refuses if protocol/config/manifests changed)
.venv/bin/python -m scp.report runs/<pilot_run_id>   # writes summary_tables.md
```

Building from scratch (new venv, model download) is described in `README.md`. `run.py register` will not
overwrite the existing registration.

## 8. Artifacts (all under `/Users/emre/Documents/Codex/ML_Projects/spectral_causal_pilot/`)

| what | path |
|---|---|
| protocol, config, readme | `PROTOCOL.md`, `PROTOCOL_AMENDMENT_v1.1.md`, `config/pilot_v1.json`, `README.md` |
| registration | `registration/REGISTRATION.json`, `registration/AMENDMENT_v1.1.json` |
| continuation mask | `metadata/continuation_mask.json` |
| reviews | `notes/three_model_review_day1.md`, `notes/codex_independent_audit.json`, `notes/claude_conditional_go_v1.1.md` |
| manifests | `manifests/split_manifest.json`, `manifests/pilot_pairs_calibration.csv`, `manifests/moment_pool_calibration.csv`, `manifests/sealed_prompts_validation.csv`, `manifests/sealed_prompts_test.csv` |
| environment | `env/hardware_and_versions.txt`, `requirements.txt`, `requirements.lock.txt` |
| code / tests | `scp/{__init__,common,data,spectral,model,metrics,experiment,report}.py`, `run.py`, `tests/*.py`, `pytest.ini` |
| test logs | `logs/pytest_20261030T153120Z.log` (first, 5 failed), `logs/pytest_20261030T153534Z.log`, `logs/pytest_20261030T153619Z.log`, `logs/pytest_final_20261030T153837Z.log` (26), `logs/pytest_final_v1.1_20261030T154404Z.log` (30) |
| quadrature check | `runs/20261030T153332Z_quadcheck` (failed, kept), `runs/20261030T153344Z_quadcheck` (32 nodes: not ok), `runs/20261030T153440Z_quadcheck` (64 nodes: ok) |
| smoke / positive control | `runs/20261030T153643Z_smoke/{smoke.json,smoke_per_pair.npz,run_manifest.json,log.txt,snapshot/}`; re-run with continuation mask `runs/20261030T154234Z_smoke` |
| benchmark | `runs/20261030T153655Z_bench/bench.json` |
| pilot | `runs/20261030T153707Z_pilot/{results.json,per_pair_rows.csv (46,656 rows),summary_tables.md,positive_control.json,partial_results.json,run_manifest.json,log.txt,snapshot/}`, stdout `logs/pilot_stdout_20261030T153706Z.log` |

Note: `registration/AMENDMENT_v1.1.json` records the source-tree hash at 15:43Z. Afterwards only
`scp/common.py` changed: `SNAPSHOT_GLOBS` now also copies the amendment and `metadata/` into run snapshots.
