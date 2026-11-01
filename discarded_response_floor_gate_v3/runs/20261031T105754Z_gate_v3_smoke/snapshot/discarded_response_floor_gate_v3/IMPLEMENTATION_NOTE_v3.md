# v3 technical remeasurement: implementation status, ready for the coordinator audit (2026-10-31)

**Author.** Code and tests by Claude Code Opus 5.5 High.

**What has been run.** Only the synthetic/unit tests (tiny CPU GPT-2 and stored arrays) and the outcome-blind
`prepare`, which uses file copies and the tokenizer.

**What has not been run.** No GPT-2 forward, smoke, registration, remeasurement or power run.

**Source of decisions.** `notes/gate_v3_technical_remeasurement_instruction_20261031.txt`.

## What v3 is

**Package.** `discarded_response_floor_gate_v3/`, a self-contained copy of the v2 modules.
* It imports `scp` read-only and **never imports the registered v2 package** (tested).
* The registered v2 bytes are unchanged, and the v2 bound hashes still equal `DEV_REGISTRATION_v2.json`.

**Repair.** All extraction uses the **canonical no-padding layout**:
* One forward per logical prompt, with that prompt replicated 19 times: no padding, no mixing, every mask 1, last
  index len − 1. `check_canonical` raises on anything else.
* The branch input x and the unpatched reference logits come from **row 17**, the `nat_r` identity row of the
  19-row group forward. The group forward is the same physical layout, so the identity control compares the same
  row of the same layout.
* Donors each get their own canonical forward.
* There is no batch-1 extraction; dtype is float32 on MPS.

**Unchanged:** model, projector, J, estimand, Q supports, estimator, every numerical and scientific threshold, the
smoke gates, the power rules (2000 × 10,000, model-conditional Gamma), and the budgets (1800 s model, 1800 s
statistics, 24 GiB, ≤ 6 threads). The test `test_v3_config_labels_paths_and_unchanged_rules` checks these against the
v2 config.

**Manifests.** `manifests/` is a byte-identical copy of v2 `manifests_v2/` (5 files). The copy is verified against
the v2 registration hashes, and `manifest_copy_v3.json` records it write-once. The copy is re-checked for structure,
acceptance (overlap counts 24/864/240, total 1344) and tokenization. The design also rebuilds identically from the
config (tested). There is no resampling and no new nouns, rank or model.

**Compute accounting.**
* Extraction: 480 logical prompts → 480 canonical forwards → 9120 computational rows.
* Group stage: 48 forwards → 912 rows.
* These are recorded in the run manifest (`compute_counts`). The caps are checked every 48 extracted prompts, after
  extraction and after every group, and a stop keeps the partial activations.

**Smoke (old calibration only).**
* Recipients alternate between lengths 12 and 14, with same-year donors of both lengths. The smoke asserts that at
  least two prompt lengths occur.
* Canonical path; every old gate plus a `canonical_layout` gate; the CPU float64 spot comparison of R and U.
* A **descriptive** legacy contrast: v2-style mixed, right-padded extraction of the same prompts with extra pads
  0–4, as the identity control.
* Runtime projection from the canonical path.

**Labels.**
* `protocol_version` is `drfg-dev-v3-techremeasure` (final).
* Development output is labelled "development TECHNICAL REMEASUREMENT … not blind/unexposed … not confirmatory".
* v2 stays EVALUATION_INVALID. If the existing gates cannot be met within the budget, STOP.

**Preservation bindings.**
* `config/gate_dev_v3.json` → `v2_preservation` holds:
  * the sha256 of 26 v2 files: registration, both configs, both protocols, the v2 protected manifest, the v2 test
    log, the manifests_v2 files, and the prepare, smoke and failed-develop runs (manifests, logs, all outputs);
  * the v2 source tree hash `74690c53…`.
* `verify-v2` and every post-registration stage check these.
* The v3 `protected_manifest.json` (892 files) covers every older protected file, all of
  `discarded_response_floor_gate/**` (149 files), and both diagnostic versions with their runs.

## Files

* `DEV_PROTOCOL_v3.md`: §V3 is the technical remeasurement, with the active stages and commands. Inherited sections
  are marked ACTIVE (§1–§8, §10, §11) or HISTORICAL (§0, the v2 paths and CLI; §9 replaced). sha256 `ec644e4e…`.
* `config/gate_dev_v3.json`.
* `manifests/`.
* `protected_manifest.json`.
* Prepare run `runs/20261031T105301Z_gate_v3_prepare`.
* Source tree `a3e9965438e7c807…` (13 files).
* Tests: `tests/test_gate_v3.py`, **48 passed**, 0 skipped. Log `logs/pytest_gate_v3_20261031T105522Z.log`; the
  earlier log `pytest_gate_v3_20261031T105311Z.log`, from the pre-review source, is kept.

**New v3 tests** (on top of the carried-over estimator, bootstrap, power, hook, resource-cap, registration and CLI
tests):
* config labels and paths, and that the rules are unchanged from v2;
* the canonical-layout check (padding, mixing, row count, other prompt);
* real `Ctx.extract` and `Ctx.group_rows` on a tiny model: unhooked, self-branch and own-q pass-through equal the
  reference row exactly, for prompts of different lengths;
* an extraction resource stop that keeps partial outputs;
* float32 own-q equivalence on all 480 stored v2 activations: bit-identical after the cast;
* v2 bytes preserved and bound, with tamper detection;
* registration requires v2 preservation;
* the manifest copy is byte-identical and reproducible, and prepare refuses a second run;
* isolation from the v2 package;
* the legacy-contrast pads.

**Coordinator pre-run review** (`notes/gate_v3_pre_run_code_review_20261031.txt`):
1. **Legacy-contrast nesting.** Already fixed, with an end-to-end tiny-model regression test.
2. **Warmup accounting.** The warmup's extraction and group forward are now reset out of `ctx.counts` and recorded
   as `warmup_counts`. Development counts are exactly 480 forwards / 9120 rows plus 48 group forwards / 912 rows,
   and develop stops if the extraction counts differ. The smoke `canonical_layout` gate checks the exact counts.
   Tested.
3. **Inherited protocol text.** The active v3 stages and commands are in §V3. §0 and the v2 paths/CLI are marked
   HISTORICAL, §9 is replaced, and the counterpart config and isolation globs point to v3. Thresholds and power
   settings are unchanged.
4. **Scope of proof.** Stated in §V3: the canonical tests and the CPU64 smoke support measured numerical validity
   in the tested cases only. They are not a general bug-absence or fresh/unexposed confirmation.

**Fixed while testing.** Two real defects were caught before handoff:
* config generation had let the v2 `paths` block overwrite v3's;
* the legacy contrast nested the hook wrapper.

## Commands (from the project root, on the M2 Max, unsandboxed MPS)

```bash
.venv/bin/python -m pytest discarded_response_floor_gate_v3/tests -v -p no:cacheprovider
.venv/bin/python -m discarded_response_floor_gate_v3 verify-protected --config discarded_response_floor_gate_v3/config/gate_dev_v3.json
.venv/bin/python -m discarded_response_floor_gate_v3 verify-v2 --config discarded_response_floor_gate_v3/config/gate_dev_v3.json
.venv/bin/python -m discarded_response_floor_gate_v3 smoke --config discarded_response_floor_gate_v3/config/gate_dev_v3.json
.venv/bin/python -m discarded_response_floor_gate_v3 register --config discarded_response_floor_gate_v3/config/gate_dev_v3.json --smoke-run <v3_smoke_run_id> --tests-log discarded_response_floor_gate_v3/logs/pytest_gate_v3_20261031T105522Z.log
.venv/bin/python -m discarded_response_floor_gate_v3 develop --config discarded_response_floor_gate_v3/config/gate_dev_v3.json
.venv/bin/python -m discarded_response_floor_gate_v3 power --config discarded_response_floor_gate_v3/config/gate_dev_v3.json --dev-run <v3_develop_run_id>
.venv/bin/python -m discarded_response_floor_gate_v3 report --config discarded_response_floor_gate_v3/config/gate_dev_v3.json --dev-run <v3_develop_run_id> --power-run <v3_power_run_id>
.venv/bin/python -m discarded_response_floor_gate_v3.aggregate discarded_response_floor_gate_v3/runs/<v3_develop_run_id>
.venv/bin/python -m discarded_response_floor_gate_v3 verify-v2 --config discarded_response_floor_gate_v3/config/gate_dev_v3.json
```
