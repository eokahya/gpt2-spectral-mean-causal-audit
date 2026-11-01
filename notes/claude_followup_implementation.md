# Claude — follow-up implementation status, ready for Codex audit (2026-10-30)

**Scope of this delivery:** code, tests, outcome-blind manifests and the old-calibration technical smoke.
* No follow-up registration, fit, validation or test outcome has been produced.
* No test label was computed.
* No old file was modified (`scp/`, `tests/`, `config/`, `registration/`, `manifests*/`, `calibration_day3/`).

## Files

* **Design:** `FOLLOWUP_DESIGN.md`.
  * Rewritten to the final R-only design. The obsolete mixed-donor text was removed and is kept only as history in
    `notes/followup_debate.md`.
  * The reviewer status is final acceptance by all three reviewers, as relayed there.
  * Codex's config audit is applied: the explicit ‖q‖·‖a_R‖ product, and the input-reconstruction norm defined
    against the frozen activation mean.
* **Protocol:** `followup/PROTOCOL_FOLLOWUP.md`, which records all implementation choices. The config is
  `followup/config/followup_v1.json`. Its `protocol_version` was `followup-v1-DRAFT` at the first smoke and is now
  `followup-v1` (see the final section); register refuses DRAFT.
* **Code:**
  * `followup/`:
    * `common.py`: hashes, write-once, runs;
    * `data.py`;
    * `local.py`: float64 local algebra, D1, gelu′;
    * `downstream.py`: leaf hook, probe VJPs, exact JVP;
    * `features.py`, `ridge.py`;
    * `stats.py`: cell weights, cluster bootstrap, G with floor;
    * `pipeline.py`: stages.
  * `run_followup.py`, `tests_followup/test_followup.py`.
* **Source hashes.** The follow-up source tree was `4f64ebe6…` (11 files) at the first smoke. It is now
  `c39af8e4…` after the test-only edit described in the final section. The registration will also bind the scp
  dependency tree, which is the unchanged Day-3 tree `f04e0ece5f30f5a5`, plus the protocol, design, config,
  manifests, Day-3 config and artifact hashes.

## Codex pipeline audit items (applied)

1. **scp dependency binding.** `bound_hashes` includes `scp_dependency_source = scp.common.source_hashes()`. It is
   verified by every post-registration stage and compared by `register` against the smoke run. The old
   SOURCE_GLOBS and files are untouched, and Day-3 `verify_frozen` still passes.
2. **Determinism.**
   * The gate is a same-shape rerun: all 32 pairs at batch 16, required bitwise identical. Result: 0.0 for both
     G and logits.
   * The cross-batch-shape check (batch 16 vs 8) is descriptive, with the tolerance rel ≤ 1e-4 fixed in the config
     before the smoke. Result: 5.0e-5, within tolerance.
3. **δb gate.** Local δb vs the model at α = 1, rel ≤ 1e-3, fixed in the config before the smoke. Result: 9.8e-5.

## Manifests (`followup/runs/20261030T191411Z_followup_prepare`)

* Frames:
  * fit N0/N5/N6/N9;
  * validation N7/N8;
  * test N1/N2/N3/N4.
* Recipients: 96 / 48 / 192, with 16 / 8 / 32 templates. Each template has a unique noun and one recipient per
  direction × gap cell.
* YY pools: 33 / 32 / 32 years, disjoint.
* 0 tokenization exclusions.
* 0 overlaps with 2178 historical prompt texts (moment pool, calibration pairs incl. controls, sealed
  validation/test, Day-3 pairs).

## Tests

`followup/logs/pytest_followup_20261030T191837Z.log`:
* old suite: 64 passed, 2 skipped. The two skips are the Day-3 guard tests, which skip by design because the Day-3
  registration/freeze files exist;
* `tests_followup`: 27 passed.

## Old-calibration technical smoke (`followup/runs/20261030T191925Z_followup_smoke`; 32 Day-1 pairs)

All gates (fixed before the run) pass:

| check | result | gate |
|---|---|---|
| leaf-forward logits | 0.0 | ≤ 1e-4 |
| FD of probe VJPs, MPS f32 (two steps, random and δb directions) | max rel 1.2e-3 | ≤ 2e-2 |
| FD, CPU f64 copy | max rel 6.9e-8 | ≤ 1e-6 |
| MPS vs f64 VJPs | rel 6.0e-5 | ≤ 1e-3 |
| local b, unperturbed / α = 1 | rel 2.9e-5 / 3.0e-5 | ≤ 1e-4 |
| local δb, α = 1 | rel 9.8e-5 | ≤ 1e-3 |
| self-patch, earlier positions, α = 0 local δb | exactly 0 | exactly 0 |
| same-shape rerun | 0 | bitwise |

Other checks:
* |a_B| max 1.6e-14;
* m_B recovery residual 1.2e-15;
* x' reconstruction error 7e-8.

**Descriptive, technical only** (not a criterion; these must not be used to change the design):
* **Exact-JVP check.** On 4 recipients, the 8-probe D over the exact ‖Jδb‖/s ratio is 0.99, 1.04, 1.23 and 1.49.
  This is a conditional, descriptive spread over four examples only. It does not establish a general noise level
  for the estimator. The registered estimator stays at 8 fixed probes, the result is conditional on them, and
  there is no probe increase.
* **Old-CAL correlations with Y:** d_exact 0.86, d_taylor 0.86, ‖δb‖ 0.77, d_shuf 0.67. Old-CAL Y mean 0.034.
  The 4+4 probe-half correlation is 0.75.
* **Runtime:** about 0.049 s per recipient for all stages in the first smoke and 0.019 s in the repeat (warm
  cache). The projection for the 336 registered recipients (2688 VJPs, 1008 interventions) is therefore about
  6–16 s of model time plus model loading.
* **Memory:** peak RSS 2.39 GB, MPS driver 1.14 GB.

## Items for the audit

* The noun and frame lists are mine. They are new, but chosen by me without any model output.
* The register stage accepts a config that differs from the smoke snapshot only in `protocol_version`, so the
  DRAFT label can be finalised without rerunning the smoke.

## Codex pre-outcome audit: complete (relayed)

Codex independently checked and reported OK:
* the smoke run status;
* all bound hashes against the current files;
* the old 33 file hashes and Day-3 `verify_frozen`;
* the CSV split, cell and leakage checks;
* the local, downstream, features, ridge, stats and pipeline code and the tests (27 new pass; 64 old pass, plus
  2 registered-guard skips);
* all 10 smoke gates.

Codex asked that the uncertainty of the 8 fixed probes stay in the report: the exact-JVP ratios on 4 old-CAL
recipients were 0.986–1.485. There is no probe increase.

## Final state before registration

**Changes after the audit.** Technical only; no outcome was produced.
1. `followup/config/followup_v1.json` `protocol_version`: `followup-v1-DRAFT` → `followup-v1`.
2. Codex edge case. Two unit tests had assumed that the real config carries the DRAFT label:
   * `test_verify_registration_detects_any_bound_change`;
   * `test_register_refuses_draft_and_existing`.

   They now use explicit `{**cfg, "protocol_version": "followup-v1-DRAFT"}` and `followup-v1` copies, so they do not
   depend on the file's label. Only `tests_followup/test_followup.py` changed. No experiment code, protocol,
   design or manifest changed.
3. The source binding changed with that edit, so the evidence was rerun:
   * the 27 follow-up tests with the final-label config: 27 passed. Log:
     `followup/logs/pytest_followup_20261030T192123Z.log`. The old 66 were not rerun, as instructed; their earlier
     log still applies, since scp is unchanged.
   * the old-CAL smoke: **`followup/runs/20261030T192134Z_followup_smoke`**. Status ok; protocol `followup-v1`; all
     10 gates pass. Every gate value equals the first smoke's value to the digits reported above.
4. **Smoke binding:** the run's bound hashes equal the current files exactly. That covers protocol, design, config,
   follow-up source tree `c39af8e4…`, scp dependency `f04e0ece5f30f5a5`, manifests, Day-3 config and artifact.

**Registration inputs for Codex:**
* `--smoke-run 20261030T192134Z_followup_smoke`
* `--tests-log followup/logs/pytest_followup_20261030T192123Z.log`

The first smoke, `20261030T191925Z_followup_smoke`, is kept but superseded: its source and config hashes no longer
match.

* Claude stops here. Codex runs registration and the fit/validate/freeze/predict-test/evaluate-test/report stages.
  No new outcome has been produced.
