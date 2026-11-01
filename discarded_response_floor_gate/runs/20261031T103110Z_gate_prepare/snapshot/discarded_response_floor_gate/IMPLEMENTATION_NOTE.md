# Implementation status, ready for the coordinator audit (2026-10-31)

**Author.** Claude Code Opus 5.5 High wrote all code and tests.

**What has been run.**
* The synthetic/unit tests.
* The outcome-blind `prepare`, which uses the tokenizer only.

**What has not been run.** No GPT-2 forward pass, smoke, registration, development run or power simulation. No
development outcome exists.

## Files

* `DEV_PROTOCOL.md`: the protocol, still a DRAFT label. `config/gate_dev_v1.json`: the config, labelled
  `drfg-dev-v1-DRAFT`.
* Code:
  * `common.py`: hashes, write-once files, the protected-file manifest, runs;
  * `data.py`;
  * `geometry.py`: the frozen P, hybrids, support diagnostics;
  * `hook.py`: replaces the MLP branch input at the last token;
  * `estimators.py`, `bootstrap.py`;
  * `power.py`: Gamma planning, Clopper–Pearson with Bonferroni, the compute cap;
  * `pipeline.py`: the stages;
  * `aggregate.py`: the separate aggregation implementation;
  * `cli.py`, `__main__.py`.
* Tests: `tests/test_gate.py`. **37 passed**, 0 skipped. Current log `logs/pytest_gate_20261031T101703Z.log`; the
  earlier log `pytest_gate_20261031T095959Z.log` is from the pre-audit source and is kept.
* Source tree sha256: `11f8594a1ec15046…` (13 files).

## Prepare

Run `runs/20261031T095918Z_gate_prepare`.

* **Manifests:** `manifests/groups.csv`, `manifests/prompts.csv`, `manifests/index.json`.
* **Design:**
  * 48 groups, 24 templates and 480 unique prompts, with distinct text and token hashes;
  * 384 discarded-donor contexts, each used once;
  * 0 overlaps with 2773 historical prompt texts;
  * tokenization: no failures.
* **Frames:**
  * recipient frames G04, G09, G11, G13;
  * donor frames G17, G18, G07, G10, G03, G12, G14, G08;
  * 8 frames and 58 nouns unused.
* **Protected files:** `protected_manifest.json` lists 735 old files, tree `31aac368…`. `verify-protected` reports
  ok. Day-3 `verify_frozen` passes, the scp tree is `f04e0ece…`, and the follow-up bound hashes still equal its
  registration.

## Points for the audit

1. **Semantic plausibility is not checked.** Frame × noun prompts are grammatical, but some are semantically odd,
   e.g. "The storm was established in the year 1791 and dissolved in the year 17". Only tokenization was checked.
2. **Donor nouns are shared across groups.** Each donor noun appears in 8 groups under 8 different frames, while
   each (frame, noun) context is used once. Fully unique nouns for 384 contexts were not available.
3. **Power bootstrap.** Replaced after the audit; see below.
4. **Smoke on old calibration prompts.** Donors are same-year moment-pool prompts. Different smoke groups with the
   same year may share donors; this is technical only.
5. **Two-step finalisation.** The DRAFT label must be finalised in `config/gate_dev_v1.json`, and if desired in the
   protocol title. Register accepts a config that differs from the smoke snapshot only in `protocol_version`. A
   protocol change after the smoke changes the bound protocol hash, so the smoke would need to be rerun.
6. **A superseded log was deleted.** Before the last source edit I removed a superseded test log of my own
   (`pytest_gate_20261031T095929Z.log`, 31 passed). The current log reflects the final source.

## Coordinator audit 2026-10-31 (`notes/coordinator_gate_code_audit_2026-10-31.txt`): six findings fixed

No scientific forward was run. The manifests and the protected manifest are unchanged; prepare was not rerun
because the data code did not change.

1. **Independent bootstrap per simulated dataset.** Each dataset j has its own data stream `(2, iN, iS, iH, j)` and
   an independent bootstrap stream `(3, iN, iS, iH, j)`. Both power and `template_bootstrap` use the same resampler
   (`bootstrap.resample_theta`), so the simulated and the confirmatory procedures match.
   * Sizes are unchanged: 2000 × 10,000, exact linear percentiles, 1800 s cap.
   * Memory is bounded: one [10,000 × n_t] int16 index array per frame per dataset.
   * The prescribed simulation projects to about 210 s, from a synthetic timing benchmark only (no development
     data).
   * Tests: the resampler against group-level brute force; resampler variance against bootstrap theory;
     reproducibility, per-dataset independence, data/bootstrap stream separation and restartability; and a null
     rejection rate in a broad band around 5% (a small sanity check, not the prescribed size).
2. **Development budget.**
   * The clock starts after model load and the untimed warmup.
   * Hard time and memory caps are checked after extraction and after **every** group; the projection is still
     made after the 3rd group.
   * A stop happens before the next forward, and partial raw outputs, controls and support rows are kept. A cap
     exceeded only after the final group is recorded.
   * The warmup is one extraction forward plus one 19-row group-shaped forward on old calibration pair 0. The smoke
     now times exactly 3 groups after that warmup, and pair 0 is never a smoke group.
   * Tested with a fake model and clock (`run_groups`, `resource_status`).
3. **Guard outcomes.** Non-finite or negative measured moments now give `EVALUATION_INVALID`. A valid near-zero
   floor or zero variance gives `POWER_NOT_ESTABLISHED`. Both are tested.
4. **Power table validation.** `decide` validates that all 5 × 4 × 2 prescribed rates are present exactly once with
   n = 2000 before any positive or NO_GO decision. Tested for missing rows, duplicate rows, short rows and a
   missing hypothesis.
5. **Locality control.** The smoke check now uses group 0's own `x_minus_0` and `x_plus_0`.
6. **Protocol wording.**
   * Donor contexts are disjoint, but donor nouns are reused under a finite Latin assignment. That lexical reuse
     remains an unmodelled dependence limitation.
   * The bootstrap resamples templates within the fixed frames with the donor construction held fixed. This
     statement of scope does not establish nominal coverage.
   * The power plan's null false-positive calibration is model-based (Gamma, independent templates). It is not a
     guarantee for the fixed finite donor assignment or for natural language.
   * The 24 recipient nouns are globally unique (one template each).
   * `aggregate.py` is labelled a same-author second implementation, not an independent audit.

**Documentation correction (pre-smoke; protocol and note only):**
* §7 now matches §8 and the code: simulated datasets are processed one at a time, with no 100-dataset chunks.
* The inference-scope wording no longer implies coverage. Donor lexical reuse and the model-based calibration are
  stated as limitations.
* Code, config, tests and manifests are unchanged. The protocol hash changes; the smoke has not been run yet.

## Commands (from the project root, on the M2 Max, unsandboxed MPS)

```bash
.venv/bin/python -m pytest discarded_response_floor_gate/tests -v -p no:cacheprovider
.venv/bin/python -m discarded_response_floor_gate verify-protected
.venv/bin/python -m discarded_response_floor_gate smoke
# finalise protocol_version (drfg-dev-v1-DRAFT -> drfg-dev-v1) only after the audit, then:
.venv/bin/python -m discarded_response_floor_gate register --smoke-run <smoke_run_id> --tests-log discarded_response_floor_gate/logs/pytest_gate_20261031T101703Z.log
.venv/bin/python -m discarded_response_floor_gate develop
.venv/bin/python -m discarded_response_floor_gate power --dev-run <develop_run_id>
.venv/bin/python -m discarded_response_floor_gate report --dev-run <develop_run_id> --power-run <power_run_id>
.venv/bin/python -m discarded_response_floor_gate.aggregate discarded_response_floor_gate/runs/<develop_run_id>
.venv/bin/python -m discarded_response_floor_gate verify-protected
```

`DEVELOPMENT_REPORT.md` and `CONFIRMATORY_PROTOCOL_DRAFT.md` will be written only after the development and power
runs.
