# PROTOCOL AMENDMENT v1.1 (to registered pilot-v1)

Written 2026-10-30, **after** the first exploratory pilot run (`runs/20261030T153707Z_pilot`), in response to
Codex's protocol and code review. `PROTOCOL.md` (pilot-v1, SHA-256 `88cb9abf…`) is left byte-identical. This
file is hashed separately in `registration/AMENDMENT_v1.1.json`.

**Effect on measurements: none.** This amendment changes no rank, method, split, pair, threshold, metric
definition used in the pilot, or primary-statistic candidate. It adds clarifications, an extra recorded
behavioural metric, guards and tests, and an explicit TODO list.

## A1. PD definition (already in v1 before registration; now unit-tested)
`PD = Σ_{j>r} p_j − Σ_{j≤r} p_j`: the equal year counts as wrong (Hanna et al.). The code applied this before
registration, and `tests/test_data.py::test_prob_diff_counts_equal_year_as_wrong` now covers it.

## A2. Scope vs moments (already corrected in v1 §4 before registration)
Local final-position responses are identical across scopes only with identical moment parameters. `bias`/`gauss`
use `final_token` moments in `last_token` and `all_token` moments in `full_layer`, so all results are reported
per scope, and every row carries `moment_pool`.

## A3. J is a logit score set, not a tokenization claim
* `J` = the 100 single tokens `"00".."99"`, used as the **score set** for `T` and for PD. That each is one
  token is a different property from `"17"+"YY"` being the model's natural two-token form of a year.
* Natural-continuation mask, recorded in `metadata/continuation_mask.json`: `j` is valid iff `' 17' + 'jj'` is
  GPT-2's own tokenization of year `17jj` at the prediction slot. For all six frames only `j = 00` is invalid
  (`"1700"` tokenizes as the single token `' 1700'`).
* The smoke stage also reports PD restricted to valid continuations. `T` and the registered PD are unchanged.
* `check_prompt` requires both `' 17'` slots, with the final token as the second slot; the start slot must be
  followed by its own `YY` token. Positive/negative tests use `"…year 1732 to the year 17"` (valid) and
  `"…year 1700…"` (merged single-token year, rejected).
* **No replication claim.** Templates, nouns, year splits, pair design and the intervention site differ from
  arXiv:2305.00586. We only use its task format and PD definition.

## A4. Controls not yet implemented (explicit TODO)
Missing from v1: norm-matched wrong donor; PCA-subspace (data-covariance) cut as an alternative to the SVD cut;
random-subspace split of equal dimension; whitened SVD; resample ablation (diagnostic reference).
**The positive control alone does not establish content specificity beyond these controls.** v1 only shows that
the full-branch transfer moves T far more than a same-year context change.

## A5. What the reduced model is
`Reduction` stores the dense projector `P_R` (768×768) and the original `c_fc` weight. It is a **functional
rank-k audit** of the input map, not a memory or speed compression. The benchmark timings measure the audit
harness and are never compression results.

## A6. Numerical hygiene
float64 metrics (`t_score`, `prob_diff`, `kl`) raise if given a non-CPU tensor; the runner moves tensors
explicitly with `.cpu()`. Tested (MPS tensors are refused).
