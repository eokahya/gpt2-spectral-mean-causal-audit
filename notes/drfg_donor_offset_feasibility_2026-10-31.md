# Discarded-response floor gate: donor-noun offset feasibility (read-only, 2026-10-31)

**Scope.**
* Combinatorics only.
* No model call, and no change to config, source, manifests, smoke or registration.
* Script: `notes/drfg_donor_offset_feasibility_2026-10-31.py`. It is pure Python (deterministic DFS) and was run with
  the project `.venv`.
* Status: awaiting GPT Pro's decision (pre-outcome amendment vs a descriptive-only note).

## The issue (Fable)

In the current design, slot k of group g uses donor noun `base[(g + 6k) mod 48]`. The offsets
S = {0, 6, …, 42} form the subgroup 6·Z48, so:
* every nonzero ordered difference that occurs has multiplicity 8;
* the noun set of group g depends only on g mod 6, giving **only 6 distinct donor-noun sets**;
* 168 of the 1128 group pairs share **all 8** donor nouns, and 960 share none. Within a frame, 24 pairs share all 8.

Contexts (frame, noun) are still each used exactly once.

## Feasibility

**General construction.** Group g slot k uses `base[(g + S[k]) mod 48]`. For any 8 distinct offsets S:
* each (frame k, noun) context is used once, since g ↦ g + S[k] is a bijection;
* each noun is used 8 times, and the 8 nouns within a group are distinct;
* groups g and h share exactly mult(h − g) nouns, where mult(d) = #{(i, j): S[i] − S[j] ≡ d}.

**Multiplicity ≤ 1 is impossible.** S has 8·7 = 56 ordered differences, but Z48 has only 47 nonzero residues.
The DFS confirms that no such S exists.

**Multiplicity ≤ 2 is feasible.** With 2x + y = 56 and x + y ≤ 47, at least 9 residues must be doubled. That
minimum of 9 is attained.

| candidate | offsets S | max mult | doubled / single / absent residues | distinct noun sets | pairwise overlap counts (1128 pairs) | same-frame pairs (264) | same-template pairs (24) |
|---|---|---|---|---|---|---|---|
| current | 0, 6, 12, 18, 24, 30, 36, 42 | 8 | n/a | 6 | 0: 960, 8: 168 | 0: 240, 8: 24 | 0: 24 |
| A, first lexicographic with max ≤ 2 | 0, 1, 2, 4, 7, 11, 16, 22 | 2 | 20 / 16 / 11 | 48 | 0: 264, 1: 384, 2: 480 | 0: 16, 1: 8, 2: 240 | 2: 24 |
| **B (recommended)**, first lexicographic with the minimum 9 doubled residues | **0, 1, 2, 5, 9, 20, 26, 36** | 2 | 9 / 38 / 0 | 48 | 1: 912, 2: 216 | 1: 188, 2: 76 | 2: 24 |

**Checks passed by both A and B:**
* 8 distinct offsets;
* every one of the 384 (frame slot, noun) contexts used exactly once;
* every noun used exactly 8 times;
* 48 distinct noun sets;
* every group pair shares at most 2 donor nouns.

For B the residue 24 has multiplicity 2; it is self-inverse, so its multiplicity is necessarily even.

**Why B.** It spreads the reuse most evenly: every group pair shares 1 or 2 nouns, none share 0, and no pair
shares more than 2.

## Important caveat: total lexical reuse is fixed

With 48 donor nouns each used 8 times, the total number of shared-noun pair incidences is
48 · C(8, 2) = **1344** for **every** offset set (checked for all three).
* Offsets only change how the reuse is distributed. They remove complete duplicate noun sets; they do not reduce
  the total amount of lexical sharing.
* With A or B, 48 of the 1344 incidences fall inside templates (difference 1, which the template-cluster bootstrap
  absorbs), versus 0 currently. The cross-template total is 1296 vs 1344.
* Reducing the total would need more donor nouns. 58 unused nouns exist, but they are reserved for a confirmatory
  pool, so using them is a separate choice.
* Low overlap does not establish independence. The inference caveat (unmodelled lexical dependence; model-based
  power calibration) remains.

## Smallest safe amendment plan (if Pro chooses to amend before any outcome)

1. **New config file** `config/gate_dev_v2.json`, a copy of v1 with:
   * `protocol_version` `drfg-dev-v2-DRAFT`;
   * `design.donor_offsets: [0, 1, 2, 5, 9, 20, 26, 36]`;
   * `design.manifest_dir: "manifests_v2"`.

   `gate_dev_v1.json` stays as the record.
2. **Code** (smallest change):
   * `data.build` reads `design.donor_offsets`, defaulting to the old 6k so that v1 rebuilds exactly. It asserts 8
     distinct offsets and records the multiplicity profile.
   * The manifest directory comes from the config, so `manifests/` (v1) is never touched.
   * Registration remains a single write-once file, since v1 was never registered.
3. **Protocol.** A dated pre-outcome amendment section in `DEV_PROTOCOL.md` giving:
   * the reason (Fable), the offsets, the checks above and the fixed-total caveat;
   * that recipients and retained donors are unchanged (same seeds and assignment);
   * that only the slot → donor-noun map changes, and the 384 contexts stay unique.
4. **Tests.**
   * The offset properties: max multiplicity 2, contexts unique, 8 uses per noun, 48 distinct sets, total 1344.
   * The v1 manifests still rebuild byte-identically under the default offsets.
   * The v2 manifests pass the structural, tokenization and leakage checks.
5. **Run order (coordinator):**
   1. tests;
   2. `prepare` (v2 manifests);
   3. a **new smoke**, required because `register` demands that the smoke's bound source and manifests equal the
      current ones;
   4. register;
   5. develop.

   The v1 manifests (`discarded_response_floor_gate/manifests/`) and the smoke run `20261031T101947Z_gate_smoke` are
   kept as superseded pre-outcome records, not deleted. No development outcome existed before the amendment.

**If Pro instead keeps the current design:** record the 6-set structure (168 fully shared pairs) as a descriptive
limitation in the report. Nothing else changes.
