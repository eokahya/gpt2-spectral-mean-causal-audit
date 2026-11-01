# Follow-up: post-hoc paired loss contributions (descriptive arithmetic, 2026-10-30)

**Post-hoc and descriptive only.**
* GPT Pro asked for this after the registered result. It does **not** change the registered primary decision
  (**INCONCLUSIVE**).
* There is no confidence interval, no new threshold and no leave-out analysis. Nothing here is a causal
  explanation.
* No model run, fit or probe.
* Inputs are read only:
  * `followup/runs/20261030T192301Z_followup_predict_test/test_features.csv` and `test_predictions.csv` (locked);
  * `followup/runs/20261030T192318Z_followup_evaluate_test/test_labels.csv`;
  * the frozen s in `followup/registration/FOLLOWUP_FROZEN.json`.
* Script: `notes/followup_posthoc_loss_contributions.py`, run with the project root and the two output paths as
  arguments.
* Outputs:
  * `notes/followup_posthoc_loss_contributions.csv`: template×α, template, α and total levels;
  * `notes/followup_posthoc_loss_contributions_rows.csv`: 576 rows.
* Computed by Claude.
* **Codex independently verified the arithmetic** (`notes/codex_followup_contribution_audit.json`): all 96 cells,
  the α B-MSE weights, the net shares, the template and cell signs, and the top-5 net and gross-positive shares.
* All three reviewers (GPT Pro, Fable Extra, Antigravity) received these numbers and recorded the closure of this
  version with no remaining objection (`FOLLOWUP_SUMMARY.md` §7). The reviewers saw the packets; the raw-file audit
  is Codex's and Claude's.
* **Status: final.**

## Definitions

* **Per row.** A row is one test recipient at one α, so there are N = 576 rows.
  Δ_i = (Y_i − predB_i)² − (Y_i − predA_i)². Δ_i > 0 means A had the smaller squared error on that row.
* **Why a plain mean is enough.** The registered MSE weights the 72 frame × direction × gap × α cells equally
  (1/72), and each cell has exactly 8 rows. So the registered MSE equals the plain mean over all 576 rows.
* **G and contributions.**
  * G = mean(Δ)/MSE_B.
  * A group g contributes Σ_g Δ / (N·MSE_B) to G. These contributions add up exactly to G.
  * The net share of g is Σ_g Δ / Σ_all Δ. It can be negative or greater than 100%.
* **Reproduction.** The script reproduces the registered MSE_A = 2.87790e-4, MSE_B = 3.43551e-4 and
  G = 0.16230704 (asserted to within 1e-12). The contributions at every level sum to G.
* **Template×α cell.** One template at one α: n = 6 recipients (one per direction × gap cell). There are 32 × 3 = 96
  such cells.
* **Interpretation boundary.** These are counts and sums over 32 template clusters, not over 576 independent
  examples. They describe this test set's loss accounting only.

## 1. Contribution by α (the pooled G is not the simple mean of the α gains)

| α | n | MSE_B | MSE_A | point G_α | B-MSE weight w_α = MSE_B,α / (3·MSE_B) | contribution w_α·G_α | net share | rows with Δ > 0 / < 0 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 192 | 1.1151e-4 | 1.4094e-4 | −0.263920 | 0.108193 | −0.028554 | −17.59% | 107 / 85 |
| 1.0 | 192 | 3.0698e-4 | 2.7327e-4 | 0.109820 | 0.297852 | 0.032710 | 20.15% | 106 / 86 |
| 1.5 | 192 | 6.1216e-4 | 4.4916e-4 | 0.266268 | 0.593955 | 0.158151 | 97.44% | 113 / 79 |
| total | 576 | 3.4355e-4 | 2.8779e-4 | **0.162307** | 1 | 0.162307 | 100% | 326 / 250 |

* **How the pooled G arises.** The simple mean of the three G_α is 0.0374. The pooled G of 0.1623 is their
  B-MSE-weighted sum. α = 1.5, the extrapolation condition, has the largest baseline error, carries 59.4% of the
  weight, and accounts for 97.4% of the net gain.
* **α = 0.5.** The net contribution is negative, even though more of its rows have Δ > 0 (107) than Δ < 0 (85).

## 2. Templates (all α, n = 18 per template)

* 23 of the 32 templates have net Δ > 0 and 9 have net Δ < 0.
* In G units, the positive templates sum to +0.279768 (gross positive) and the negative ones to −0.117461 (gross
  negative). The net is 0.162307.
* The top 5 positive templates hold **81.03% of the net gain** and **47.01% of the gross positive**
  (0.131524 / 0.279768). The net share is larger because the 9 negative templates offset part of the gross
  positive.

| template | MSE_B | MSE_A | mean Δ | contribution to G | net share |
|---|---|---|---|---|---|
| N4\|armistice | 6.0955e-4 | 2.2478e-4 | +3.8477e-4 | +0.034999 | +21.56% |
| N3\|investigation | 4.6534e-4 | 1.6023e-4 | +3.0511e-4 | +0.027754 | +17.10% |
| N1\|apprenticeship | 3.7315e-4 | 8.8504e-5 | +2.8465e-4 | +0.025892 | +15.95% |
| N3\|union | 3.3543e-4 | 9.4668e-5 | +2.4076e-4 | +0.021900 | +13.49% |
| N2\|marriage | 3.8684e-4 | 1.5621e-4 | +2.3063e-4 | +0.020979 | +12.93% |
| N3\|feud | 7.0763e-4 | 4.8453e-4 | +2.2310e-4 | +0.020293 | +12.50% |
| N2\|shortage | 7.7301e-4 | 5.5463e-4 | +2.1838e-4 | +0.019864 | +12.24% |
| N3\|governorship | 5.1039e-4 | 2.9504e-4 | +2.1536e-4 | +0.019589 | +12.07% |
| N2\|translation | 3.2171e-4 | 1.5755e-4 | +1.6415e-4 | +0.014932 | +9.20% |
| N1\|inquiry | 2.5017e-4 | 9.2050e-5 | +1.5812e-4 | +0.014383 | +8.86% |
| N1\|settlement | 2.3136e-4 | 9.4597e-5 | +1.3676e-4 | +0.012440 | +7.66% |
| N1\|recession | 3.4805e-4 | 2.1953e-4 | +1.2852e-4 | +0.011690 | +7.20% |
| N1\|trial | 1.7116e-4 | 8.5617e-5 | +8.5548e-5 | +0.007782 | +4.79% |
| N4\|winter | 2.1091e-4 | 1.3724e-4 | +7.3674e-5 | +0.006702 | +4.13% |
| N1\|rivalry | 1.4464e-4 | 7.8957e-5 | +6.5684e-5 | +0.005975 | +3.68% |
| N2\|scarcity | 5.4817e-4 | 5.0962e-4 | +3.8554e-5 | +0.003507 | +2.16% |
| N2\|uprising | 2.3094e-4 | 2.0185e-4 | +2.9084e-5 | +0.002646 | +1.63% |
| N2\|outbreak | 3.3331e-4 | 3.0957e-4 | +2.3745e-5 | +0.002160 | +1.33% |
| N4\|quarantine | 6.9982e-5 | 4.6430e-5 | +2.3553e-5 | +0.002142 | +1.32% |
| N4\|insurrection | 1.7205e-4 | 1.5421e-4 | +1.7846e-5 | +0.001623 | +1.00% |
| N4\|standoff | 2.3289e-4 | 2.1669e-4 | +1.6191e-5 | +0.001473 | +0.91% |
| N1\|reconstruction | 1.6818e-4 | 1.6048e-4 | +7.6935e-6 | +0.000700 | +0.43% |
| N3\|residency | 5.2287e-5 | 4.8509e-5 | +3.7782e-6 | +0.000344 | +0.21% |
| N3\|tenure | 2.5176e-4 | 2.5657e-4 | −4.8145e-6 | −0.000438 | −0.27% |
| N4\|flood | 4.2525e-5 | 4.9263e-5 | −6.7374e-6 | −0.000613 | −0.38% |
| N2\|mutiny | 4.3919e-4 | 4.6467e-4 | −2.5481e-5 | −0.002318 | −1.43% |
| N4\|friendship | 1.8194e-4 | 2.1288e-4 | −3.0935e-5 | −0.002814 | −1.73% |
| N1\|dispute | 1.2223e-4 | 1.9929e-4 | −7.7059e-5 | −0.007009 | −4.32% |
| N2\|monarchy | 3.1532e-4 | 4.4220e-4 | −1.2688e-4 | −0.011541 | −7.11% |
| N4\|commission | 1.5673e-4 | 3.1306e-4 | −1.5633e-4 | −0.014220 | −8.76% |
| N3\|negotiation | 1.1949e-4 | 3.6357e-4 | −2.4409e-4 | −0.022203 | −13.68% |
| N3\|upheaval | 1.7173e-3 | 2.3363e-3 | −6.1899e-4 | −0.056305 | **−34.69%** |

* **N3|upheaval.** It has the largest baseline error of all templates (MSE_B 1.72e-3) and the largest negative
  contribution: −34.69% of the net gain, or 47.9% of the gross negative.

## 3. Template×α cells (n = 6 each, 96 cells)

* 64 of the 96 cells have net Δ > 0 and 32 have net Δ < 0.
* Gross positive is +0.296751 and gross negative −0.134444, in G units; the net is 0.162307.
* The top 5 positive cells hold 48.51% of the net gain. All 96 cells are in the CSV.

**Top 5 positive and top 5 negative cells:**

| cell | MSE_B | MSE_A | mean Δ | contribution | net share | share of own gross sign |
|---|---|---|---|---|---|---|
| N4\|armistice, α = 1.5 | 1.0942e-3 | 4.2862e-4 | +6.6553e-4 | +0.020179 | +12.43% | 6.80% of gross + |
| N3\|investigation, α = 1.5 | 9.0162e-4 | 2.9959e-4 | +6.0203e-4 | +0.018254 | +11.25% | 6.15% |
| N3\|governorship, α = 1.5 | 9.3585e-4 | 4.7839e-4 | +4.5746e-4 | +0.013871 | +8.55% | 4.67% |
| N2\|shortage, α = 1.5 | 1.4231e-3 | 9.7960e-4 | +4.4354e-4 | +0.013448 | +8.29% | 4.53% |
| N1\|apprenticeship, α = 1.5 | 5.5932e-4 | 1.3091e-4 | +4.2841e-4 | +0.012990 | +8.00% | 4.38% |
| N3\|upheaval, α = 1.0 | 1.8527e-3 | 2.6543e-3 | −8.0164e-4 | −0.024306 | −14.98% | 18.08% of gross − |
| N3\|upheaval, α = 0.5 | 9.2461e-4 | 1.6138e-3 | −6.8917e-4 | −0.020896 | −12.87% | 15.54% |
| N3\|upheaval, α = 1.5 | 2.3746e-3 | 2.7407e-3 | −3.6617e-4 | −0.011103 | −6.84% | 8.26% |
| N4\|commission, α = 1.5 | 3.0563e-4 | 5.9483e-4 | −2.8920e-4 | −0.008769 | −5.40% | 6.52% |
| N3\|negotiation, α = 0.5 | 3.5410e-5 | 3.1702e-4 | −2.8161e-4 | −0.008539 | −5.26% | 6.35% |

## 4. Rows (recipient × α; 576 rows)

* 326 rows have Δ > 0 and 250 have Δ < 0.
* The gross positive is 265.77% of the net gain and the gross negative is −165.77%.

| row | template | Y | pred_B | pred_A | Δ | net share |
|---|---|---|---|---|---|---|
| tes-0148, α = 1.5 | N4\|armistice | 0.1680 | 0.0977 | 0.1275 | +3.293e-3 | +10.25% |
| tes-0100, α = 1.5 | N3\|feud | 0.1729 | 0.1170 | 0.1504 | +2.615e-3 | +8.14% |
| tes-0106, α = 1.5 | N3\|governorship | 0.2015 | 0.1338 | 0.1565 | +2.560e-3 | +7.97% |
| tes-0085, α = 1.5 | N2\|translation | 0.1863 | 0.1302 | 0.1604 | +2.472e-3 | +7.70% |
| tes-0110, α = 1.5 | N3\|investigation | 0.1139 | 0.1687 | 0.1378 | +2.428e-3 | +7.56% |
| tes-0118, α = 0.5 | N3\|negotiation | 0.0589 | 0.0510 | 0.0170 | −1.694e-3 | −5.27% |
| tes-0150, α = 1.5 | N4\|commission | 0.1099 | 0.0794 | 0.0560 | −1.973e-3 | −6.14% |
| tes-0140, α = 1.5 | N3\|upheaval | 0.2411 | 0.1360 | 0.1228 | −2.940e-3 | −9.15% |
| tes-0140, α = 0.5 | N3\|upheaval | 0.0928 | 0.0466 | 0.0158 | −3.798e-3 | −11.83% |
| tes-0140, α = 1.0 | N3\|upheaval | 0.1891 | 0.0990 | 0.0738 | −5.166e-3 | −16.08% |

* **Recipient tes-0140** (N3|upheaval) supplies the three most negative rows. Together they are −37.06% of the net
  gain, which is more than the whole N3|upheaval template (−34.69%). The template's other five recipients are net
  positive.

## 5. What the data say (arithmetic only)

* **Concentration by α.** The positive pooled G comes mostly from α = 1.5 (net share 97.4%), where the baseline
  error is largest. At α = 0.5 the net contribution is negative. The pooled G weights the α slices by their B-MSE;
  it is not their simple mean.
* **Mixed sign across templates.** The gain is spread unevenly: 23 positive and 9 negative templates. The top 5
  hold 81.03% of the net and 47.01% of the gross positive. A few clusters and rows carry large shares of either sign. The largest single drag is
  one recipient (tes-0140) at all three α.
* **What the gain measures.** The gain is in *predicting* the size of the reduced model's intervention error, i.e.
  the norm of the full-logit error vector. It is not a reduction of that error, and not a prediction of its
  direction or of any circuit.
* **Taylor secondary.** The equivalence or superiority of D_Taylor relative to D_exact was not tested.
* **Not established.** None of this explains why the slices differ, and it adds no interval or threshold. A net
  share is neither a subgroup G nor an equivalence test. This is heterogeneity, not a mechanism.
* The registered decision stays **INCONCLUSIVE**. This is a limited propagation audit of how well the error norm
  can be predicted, with no error-reduction, WeightWatcher/RG or novelty claim.
