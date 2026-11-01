# Three-model review — Day 1 results (2026-10-30)

Scope: the **first-day** results of a 3-day pilot (pilot-v1 + amendment v1.1). Calibration split only,
exploratory. Day 1 is complete; Days 2 and 3 have not started. **Status: final Day-1 review; all three
reviewers agree on the narrow reading (§2), with no blocking objection.** Nothing below is confirmatory, no primary statistic or margin
has been chosen, and no new experiment was started for this report.

Reviewed run: `runs/20261030T153707Z_pilot` (46,656 rows; status ok; MPS float32). Supporting files:
`SUMMARY.md`, `runs/20261030T153707Z_pilot/summary_tables.md`, `notes/codex_independent_audit.json`,
`notes/claude_conditional_go_v1.1.md`, `PROTOCOL.md`, `PROTOCOL_AMENDMENT_v1.1.md`.

## 1. Model responses (as relayed by Codex; summaries, not full transcripts)

**Codex (coordinator; independent read-only numerical audit, `notes/codex_independent_audit.json`).**
* Recomputed gauss/bias RMS e_T for all six scope×rank cells from the CSV; they match `results.json` within
  ≤2.2e-16.
* Max |T_B| over reduced rows is 1.86e-5.
* Independently verified the signed decomposition `e_RB ≈ e_R − C`, `C = T_B^orig + I^orig` (max identity
  residual ≈1.85e-5), and the MSE split. Example, k=96 LT: MSE_R(gauss−bias) = −0.0018 while MSE_RB(gauss−bias)
  = +0.060, driven by the cross term `−2E[C·Δe_R]` (+0.062).
* Interpretation recorded: error cancellation in this metric is observed; the mechanism producing the R
  over-response is not established.
* Corrected the SUMMARY claim "B dominates at all three cuts" (at k=384, R 0.862 > B 0.768).
* Required that the R advantage is not moved silently into the primary.

**GPT Pro (round 1).**
"Conditional GO; continue to Day 2."
* At k=96/192 LT, the KL ranking does not give the intervention-fidelity ranking.
* It is not shown that the Gaussian damaged the preserved mechanism more.
* `e_RB = e_R − T_B − I`; bias's R over-response can offset the B loss.
* Gauss−bias differences are only ~1–2% of the original RB RMS.
* The FL384 CI includes 0.
* Priorities: (i) the CSV decomposition (done); (ii) at the existing 3 cuts only, norm- / pre-activation-matched
  PCA, coordinate, random and whitened controls; (iii) quality matching on validation; (iv) freeze
  primary/margin before test.
* A mechanistic-interpretability claim cannot be made until error cancellation is separated from geometric
  controls and the result replicates on held-out data.

**GPT Pro (round 2, final).** Agrees with the joint reading; conditional GO; no further blocking objection.
* The decomposition explains arithmetically why the Gaussian can reduce R-response error while increasing
  natural RB error: bias's R over-response partly offsets the missing B+I effect.
* A lower RB error does not mean more faithful R, and the RB deterioration is still a real total-response error.
* RB stays primary, R secondary. B+I is not a universal irreducible lower bound. The in-network cause is not
  yet determined.
* Until the controls and the held-out test are done, this is an error-offsetting finding specific to the
  measured calibration conditions; method superiority or mechanism preservation cannot be generalised.

**Antigravity (round 2).** "Conditional GO."
* Withdrew its first-round statement that Gaussian gain attenuation is the established cause: the Gaussian
  improving R while RB gets worse may be error cancellation.
* RB primary must not be changed; R is secondary; report the signed R bias.

**Antigravity (round 3, final).** Full agreement, no objection.
* The identity and the MSE cross term establish error cancellation at the metric level.
* Approves moving to the Day-2 three-cut controls, with no in-network mechanism claim, KL read only under the
  observed conditions, RB primary and R secondary.

**Claude (implementer; `notes/claude_conditional_go_v1.1.md`).** Conditional GO on the narrow reading below. The
RB ranking is partly an arithmetic artefact of the decomposition; R gauss ≤ bias is secondary and has no
mechanism; freeze the primary contrast and ε before validation; no specificity claim; no compression claim.

## 2. Joint conditional GO (narrow reading)

On the calibration split, at block 9 with three pre-specified cuts:
* The B-mediated part of the natural year-swap response is sizable at every cut. It exceeds the R part only at
  k=96 and k=192 (k=384: R 0.862 > B 0.768).
* **The reduced model's response to a pure-B patch is structurally zero** (Δ_B ≡ 0). The compensations are constant
  moment terms that do not depend on the discarded input; their output still depends on `z`.
* This RB loss is real and meaningful for an end-to-end **restoration audit**. It is **not** a universal
  irreducible error floor for natural RB: an R over-response can offset it (`e_RB = e_R − (T_B + I)`).
* Therefore RB (registered primary candidate, unchanged) does not by itself rank the methods' R fidelity. R is
  reported as a secondary compensation-fidelity readout, with its signed bias.
* Gauss−bias RB differences are ≤2% of the original RB RMS; full_layer k=384 CI includes 0.
* No universal Gaussian harm or improvement is claimed. No mechanism (e.g. slope-gain attenuation) is established.

## 3. Limits

* Calibration split only; the same pairs serve as gate and measurement; validation not run; test sealed; ε and
  primary not frozen.
* 18 (frame, noun) clusters but only 2 frames: cluster CIs do not represent between-frame uncertainty.
* One layer (9), three cuts, one task, one model.
* Missing controls: norm-/pre-activation-matched PCA, coordinate, random-subspace and whitened cuts;
  norm-matched wrong donor; resample ablation. The positive control alone does not establish content specificity.
* Scopes use different moment pools (final-token vs all-token), so the scope comparison is confounded with the pool.
* The reduction is a functional rank-k audit, not compression.
* J is a logit score set; no exact replication of arXiv:2305.00586 is claimed.

## 4. Decision and forward plan (agreed; nothing started)

**Decision:**
* RB (natural donor) stays the **primary candidate** as the end-to-end restoration audit.
* R is the **secondary** compensation-fidelity readout and is reported with its signed bias.
* Still **not frozen**: ε and the specific confirmatory rank / contrast.

Day 1 (done): pilot, audit, and CSV decomposition of the signed R error and the MSE cross term (Claude check +
Codex audit).

**Day 2:** at the existing three cuts only, norm- and pre-activation-energy-matched geometric controls (PCA,
coordinate, random-subspace, whitened).

**Day 3:**
* quality matching on the validation split;
* freeze the rules (primary contrast, confirmatory rank, ε) **before** the test split is opened;
* if appropriate, held-out confirmation.

Any mechanistic claim waits until error cancellation is separated from the geometric controls and the result
replicates on held-out data.

## 5. Reproduction

Exact commands: `SUMMARY.md` §7; environment setup: `README.md`.
