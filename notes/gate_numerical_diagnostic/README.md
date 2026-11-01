# v2 identity-control failure: read-only diagnosis and diagnostic tool (2026-10-31)

**Status.**
* v2 development `20261031T103329Z_gate_develop` stays **EVALUATION_INVALID**.
* There is no relaxation, reclassification, rerun or power run.
* Nothing here changes the registration, config, manifests or bound source. This directory is outside the gate's
  source globs; the v2 bound hashes still equal the registration, and the protected files verify.

## 1. What the saved development files already show (no model call)

The development identity control compares two things:
* `F_group(branch = x_r)`: the 19-row group forward of identical copies of the recipient, with no padding, and the
  branch input taken from extraction;
* the unpatched recipient logits from **extraction**: batches of 48 mixed-length prompts, right-padded to the
  batch maximum.

| recipient padding in its extraction batch | groups | identity diff > 1e-4 | identity diff exactly 0 | max |
|---|---|---|---|---|
| 0 | 5 | 0 | **5** | 0 |
| 1 | 15 | 2 | 7 | 2.14e-4 |
| 2 | 17 | 8 | 0 | 1.83e-4 |
| 3 | 10 | 5 | 0 | 1.53e-4 |
| 4 | 1 | 0 | 0 | 7.6e-5 |

* **Scale of the differences.** The J logits have max |logit| of 156–207, where the float32 ulp is 1.5e-5. The
  largest difference is therefore about 14 ulp, a relative 1.2e-6.
* **Duplicate rows** (the same layout and the same input) are exactly 0 in all 48 groups.
* **Reading.**
  * Every recipient that had **no padding** in extraction reproduces its unpatched logits **exactly** through the
    hook in the group layout.
  * The non-zero differences occur only for padded extraction rows and are at the ulp level of float32 logits.
  * This is **consistent with** float32 layout numerics (sequence length / padding in the extraction batch).
* **Correction (retracted overstatement).** An earlier version of this note said a hook bug would necessarily also
  break the unpadded cases. That is not true: padding-conditional or row-conditional bugs are possible. Finite
  diagnostics only rule out the mechanisms they actually exercise.
* **The threshold.** The pre-registered 1e-4 absolute identity tolerance was set before this layout effect was
  measured. It is **not** relaxed here; any remedy is a reviewer decision.

## 2. First diagnostic run (`runs/20261031T103857Z_diag`; script kept byte-identical as `diagnose_v1.py`, sha256 `d15f2bc5…`)

**Results.**
* Hook identity H in the same layout was 0 in every layout.
* No-hook drift vs a single prompt was up to 3.2e-4; padding-only drift was up to 1.5e-4.
* The lm_head GEMM shape alone gave up to 2.7e-4.
* The development-control decomposition D was **0** for all 16 targets.

**Coverage gap.** Length-12 targets had natural padding 2 in mix48 and length-14 targets had padding 0, but the
target was always at **row 0**. Pads 1, 3 and 4 and other batch rows, which all occur in the development extraction,
were never exercised. So D = 0 there does **not** reproduce or explain the development failure.

## 3. Diagnostic v2 (for the coordinator on the M2 Max)

**Data.** `diagnose.py` (v2) uses old Day-1 calibration prompts only: every pilot role (recipient, retained,
same-year, wrong donor) plus the moment pool, deduplicated, with lengths 12 and 14.

**Mixed 48-prompt cases.** For every target, a mixed 48-prompt batch is built for each pad level P in {0, 1, 2, 3, 4}
and each target row in {0, 24, 47}:
* the other 47 prompts are no longer than the target, and P masked pad positions are appended, so the target's
  padding is exactly P (asserted);
* length-12 targets also get a **natural** variant: length-14 prompts mixed in, giving padding 2 with no extra pad
  positions.

Padded and unpadded cases are both guaranteed and counted.

**Quantities.**
* Per case:
  * same-layout hook identity, for the target row and for all rows;
  * the development control D and its exact split into the injected-input part and the layout part (no hook);
  * branch-input drift against group19.
* Per target: hook identity in single and group19, layout drift, padding only, and the lm_head GEMM shape.
* **CPU float64** for the first 2 targets: the **same** suites in float64 (same-layout H and cross-layout X/G/D),
  plus every float32 layout against the float64 single-prompt reference.

**Output.** `runs/<UTC>_diag_v2/` (`summary.json`, `per_target.json`). The summary gives D by pad level (max and
non-zero count) and for both precisions.

**Inputs.** `diagnose.py` uses **old Day-1 calibration prompts only**: pilot recipients of length 12 or 14, mixed with
moment-pool prompts. It uses no development prompt, and it writes only to
`notes/gate_numerical_diagnostic/runs/<UTC>_diag/` (`summary.json`, `per_target.json`).

**Layouts per target:** single; group19 (19 copies); same48 (48 copies, no padding); pad1–pad4 (batch 1 with K
masked pad positions); mix48 (the target inside a 48-prompt mixed batch, as in development extraction).

**Quantities** (logit differences are reported raw, in float32 ulps, and as ‖ΔC_J‖²_m):
* **H** — hook identity in the **same** layout, with branch = that layout's own x. It must be exactly 0; otherwise
  the hook is a bug.
* **X** — **input-extraction drift**: the branch input x at the final token, layout vs single (no hook).
* **G** — logits drift across layouts with no hook.
* **padding-only** — the padK layouts vs single, giving padding effects without batch mixing.
* **M** — **lm_head GEMM shape only**: the same final hidden row through lm_head in row blocks of 1, 19 and 48.
* **D** — the development control decomposed. Two parts (exact sum, residual reported):
  * the **injected-input part**: `F_19(branch=x_mix48) − F_19(unpatched)`;
  * the **layout part (no hook)**: `F_19(unpatched) − F_mix48(unpatched)`.
* **C** — every float32 layout vs a CPU float64 single-prompt reference, for the first 3 targets.

**Reading guide.**
* H ≠ 0 means a hook bug.
* H = 0, with X and G non-zero for padded/mixed layouts and 0 for unpadded ones, means float32 layout numerics.
* D shows how much of the development control comes from the injected extraction input versus the forward layout.
* M isolates the GEMM.
* C shows that every float32 layout is within float32 error of the float64 reference; none is the "wrong" one.

**Tests.** `test_diagnose.py` (v2) runs on CPU with a tiny random GPT-2 and was run here: **5 passed**. It covers:
* padded, unpadded and natural cases, and rows 0, 24 and 47;
* exact hook identity in every layout for both precisions;
* the decomposition residual;
* the CPU64 same-layout and cross-layout controls;
* exact controlled padding;
* the v1 script hash matching the first run.

## Commands (from the project root)

```bash
.venv/bin/python -m pytest notes/gate_numerical_diagnostic/test_diagnose.py -q -p no:cacheprovider
.venv/bin/python notes/gate_numerical_diagnostic/diagnose.py --device mps --n-targets 8 --cpu64-spots 2
```

The second command loads GPT-2 on MPS, plus a CPU float64 copy for the spots. Each target runs about 18 mixed cases,
each with 3 forwards (one of 48 rows and two of 19 rows); the 2 float64 spots repeat this on CPU. The expected
runtime is a few minutes. It must run outside the sandbox, because MPS is not available inside it.
