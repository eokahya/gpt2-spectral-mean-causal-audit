# Follow-up summary: limited propagation audit — final (closed)

**Status: FINAL, closed.**
* The registered result is **INCONCLUSIVE**.
* Codex has completed independent raw-data audits of both the result and the post-hoc contribution arithmetic.
* GPT Pro, Fable Extra and Antigravity each reviewed the packets and recorded closure, with no remaining objection
  (§7).
* This is a limited propagation audit of how well the error *norm* can be predicted. It is not an
  error-reduction, mechanism, WeightWatcher/RG or novelty result.
2026-10-30. Protocol `followup-v1` (`followup/PROTOCOL_FOLLOWUP.md`; decision record `FOLLOWUP_DESIGN.md`).

## 1. Kısa sonuç (sade dil)

**Soru.** Bir PCA192 + ortalama (bias) indirgemesinde, retained-only müdahale `x' = x_r + α P(x_a − x_r)` yapılıyor.
Bunun final-token logit hatasını tahmin ederken, yerel hatanın aşağı akış (downstream) duyarlılıklarıyla
ağırlıklandırılması (D) işe yarıyor mu? Karşılaştırma, güçlü bir baseline'a (B, 14 özellik) göre yapıldı. Önceden
kayıtlı hedef: test MSE'sinde en az %10 kazanç, tek taraflı %95 alt sınırla gösterilmeli.

**Sonuç: INCONCLUSIVE (kararsız).**
* Nokta tahmini G = 0.162: D eklenince, hata normunu *tahmin etmenin* test MSE'si yaklaşık %16 düştü. Modelin
  kendi hatası azalmadı.
* Tek taraflı %95 alt sınır 0.0008. Kayıtlı 0.10 eşiğinin altında kaldığı için **GO yok**.
* Tek taraflı %95 üst sınır 0.328. Bu da 0.10'un üstünde olduğu için hedef **dışlanamıyor**.
* Bu, etkinin olmadığı anlamına gelmez; olduğu anlamına da gelmez.
* Taylor ikincil modeli (G = 0.132) betimseldir.
* α'ya göre nokta kazançları −0.26 / 0.11 / 0.27 arasında dağınık; bunlar betimseldir ve kurtarma kriteri değildir.

**Kapsam.**
* Tek model, katman, rank ve görev; 8 sabit probe.
* Mekanizma, yenilik, WW/RG veya sıkıştırma iddiası yok.
* Üç reviewer da (GPT Pro, Fable Extra, Antigravity) kalan bir itiraz olmadan bu sürümün kapatılmasını kayda
  geçirdi; model, örnek ya da probe sayısı artırılmayacak.
  * %10 minimum kazanç ne doğrulandı ne dışlandı.
  * Kapatma bir kaynak kararı, matematiksel bir imkânsızlık değil.
  * Ham verinin bağımsız denetimini Codex ve Claude yaptı. Reviewer'lar paketleri gördü, ham dosyaları
    denetlemedi.
* Kazanç, hata *normunun tahmininde*. Modelin müdahale hatasını azaltmıyor; hatanın yönünü ya da devreyi de
  tahmin etmiyor.
* Post-hoc betimsel katkı dökümü (`notes/followup_posthoc_loss_contributions.md`):
  * net kazancın %97'si α = 1.5'ten geliyor; α = 0.5'in katkısı negatif;
  * template'lerin 23'ü pozitif, 9'u negatif; ilk 5 template net kazancın %81.03'ünü, brüt pozitifin
    %47.01'ini taşıyor;
  * tek bir alıcı (tes-0140) net kazancın −%37'sini taşıyor.

  Bu döküm kararı değiştirmez. Codex bağımsız olarak doğruladı. Gösterdiği şey heterojenlik, mekanizma değil.

## 2. Registered result (test: 192 recipients × 3 α = 576 rows, 32 template clusters, 4 frames)

The statistics are fixed-weight cell MSEs (72 cells, uniform weights), compared with a paired template-within-frame
cluster bootstrap (B = 10,000, seed 20261104). Every bootstrap denominator is valid: 0 draws are at or below the
floor of 1e-12.

| quantity | value |
|---|---|
| MSE(B), baseline (14 features) | 3.4355e-4 |
| MSE(A) = B + D_exact (primary) | 2.8779e-4 |
| MSE(A_in) = B + D_Taylor (secondary) | 2.9822e-4 |
| MSE(fit-mean constant) | 1.44859e-3 |
| **Primary G = 1 − MSE(A)/MSE(B)** | **0.16230704** |
| one-sided 95% lower bound (5% quantile) | **0.00082324** |
| one-sided 95% upper bound (95% quantile) | **0.32804070** |
| MSE(B) − MSE(A), registered two-sided 95% | 5.576e-5 [−1.030e-5, 1.155e-4] |
| **Decision** | **INCONCLUSIVE**: the lower bound does not exceed 0.10 and the upper bound does not rule it out |
| Secondary G_Taylor (descriptive) | 0.13193702; lower −0.04574878, upper 0.31218287 |

* The lower and upper bounds are **each a one-sided 95% bound**. They are not a two-sided 95% interval.
* The registered 0.10 threshold is the only decision rule. The fact that the point value or the lower bound is
  above 0 is not a registered criterion.
* No G_Taylor/G ratio is computed (removed before registration).

**Descriptive (pre-registered outputs; not decision or rescue criteria; no causal explanation of the differences
is established):**

| slice | MSE(A) | MSE(B) | point G |
|---|---|---|---|
| α = 0.5 | 1.409e-4 | 1.115e-4 | −0.263920 |
| α = 1.0 | 2.733e-4 | 3.070e-4 | 0.109820 |
| α = 1.5 (extrapolation) | 4.492e-4 | 6.122e-4 | 0.266268 |
| frame N1 | 1.274e-4 | 2.261e-4 | 0.437 |
| frame N2 | 3.495e-4 | 4.186e-4 | 0.165 |
| frame N3 | 5.049e-4 | 5.200e-4 | 0.029 |
| frame N4 | 1.693e-4 | 2.096e-4 | 0.192 |

**Interpretation limits (corrections Codex required of the reviewers' first readings):**
* **Fit-mean reference.** B reduces test MSE by 76.28% relative to the **fit-mean predictor** (A by 80.13%). This is
  a fit-mean-referenced MSE reduction. It is not a test-mean R² and not "explained variance".
* **No ablation.** B is a combined set of 14 features: input/preactivation norms and their product, local b norms,
  ‖δb‖, KL, Jnorm, B_iso, ‖D1‖ and the shuffled D. No ablation was run, so the baseline's performance cannot be
  attributed to "norms" or to "isotropic sensitivity".
* **Y scale.** Y mean = 0.081 is the response error divided by s. Here s = 151.148 is the RMS of the original
  α = 1 R-response norm over fit recipients, a common scale denominator. Y is not a per-row 8% relative error.
* **Unestablished explanations.** No explanation of the α heterogeneity (probe noise, overshoot, attenuation) is
  established. The claims of "guaranteed attenuation" and "about 220 clusters needed / impossible to power" were
  rejected as unestablished.

## 3. Development stages (fit / validation)

* **Fit** (96 recipients, 288 rows):
  * s = 151.14832312;
  * fit mean of Y = 0.0764608;
  * descriptive 4+4 probe-half reliability: Pearson 0.7362, mean ratio 0.9747.
* **Validation** (48 recipients, 144 rows): fixed-weight MSE by λ ∈ {0.1, 1, 10}.

  | model | λ = 0.1 | λ = 1 | λ = 10 |
  |---|---|---|---|
  | B | 3.986e-4 | 3.870e-4 | 3.723e-4 |
  | A | 2.192e-4 | 2.100e-4 | 2.052e-4 |
  | A_in | 2.310e-4 | 2.217e-4 | 2.159e-4 |

  * The fit-mean predictor's validation MSE is 1.4366e-3.
  * The selected λ is 10 for all three models; this is the upper end of the registered grid.
  * The gate passed: A and B both beat the fit-mean predictor.
* **Freeze:** s, the λ selections, the coefficients, the standardization and the fit-mean value.

## 4. Order, locks and timestamps (UTC, 2026-10-30)

| step | time / run | note |
|---|---|---|
| prepare (manifests, tokenizer only) | 19:14:11 `20261030T191411Z_followup_prepare` | outcome-blind |
| final smoke (old Day-1 calibration only) | 19:21:34 `20261030T192134Z_followup_smoke` | 10/10 gates; bound to final source |
| **registration** | **19:22:13.065** `followup/registration/FOLLOWUP_REGISTRATION.json` | before any new outcome |
| fit, failed attempt | 19:22:17 `20261030T192217Z_followup_fit` | sandboxed, MPS unavailable; failed before model load, no outputs; kept |
| fit | 19:22:29 `20261030T192229Z_followup_fit` | unsandboxed MPS |
| validate | 19:22:42 `20261030T192242Z_followup_validate` | gate pass |
| freeze | 19:22:51.196 `FOLLOWUP_FROZEN.json` | |
| predict-test (no perturbed forward) | 19:23:01 `20261030T192301Z_followup_predict_test` | |
| **test prediction lock** | **19:23:06.100** `FOLLOWUP_TEST_PREDICTIONS_LOCK.json` | before evaluation |
| evaluate-test (labels) | 19:23:18.741 `20261030T192318Z_followup_evaluate_test` | lock hashes verified first |
| evaluated marker | 19:23:23.097 `FOLLOWUP_TEST_EVALUATED.json` | write-once |
| report | 19:25:12 `20261030T192512Z_followup_report` | |

* There was no code, config or protocol change after registration.
* **Bound hashes:**
  * follow-up source tree `c39af8e403bdfdbe…`;
  * scp dependency `f04e0ece5f30f5a5…`, the Day-3 tree, unchanged;
  * registration sha256 `f4b24ef9…`;
  * freeze sha256 `73d4c262…`;
  * results.json sha256 `024cded5…`.
* The historical 33 files and the Day-3 freeze are unchanged (Codex audit).

## 5. Authorship, verification and evidence

**Authorship.**
* Design: the real app debate (GPT Pro, Fable Extra, Antigravity) with Codex coordination (`notes/followup_debate.md`).
* All follow-up code and tests: Claude Code Opus 5.5 High (`followup/`, `run_followup.py`, `tests_followup/`).
* Execution from registration onwards: Codex.

**Tests:** 91 distinct tests passed, 2 skipped, across the two recorded logs.
* 27 follow-up tests passed with the final config/source in the registration test log
  `followup/logs/pytest_followup_20261030T192123Z.log`.
* Of the 66 historical tests, 64 passed. The 2 skips are the Day-3 guard tests, which skip by design because the
  registrations exist. Their regression log is `followup/logs/pytest_followup_20261030T191837Z.log`.

**Smoke.** All 10 numerical gates pass on old calibration only. For example:
* f32 FD max rel 1.245e-3;
* f64 FD about 6.9e-8;
* MPS vs f64 VJP 6.0e-5;
* local δb rel 9.82e-5;
* self-patch, earlier positions and same-shape rerun exactly 0.

The descriptive exact-JVP ratios over 8 probes were 0.986–1.485 on 4 old-calibration examples. That is a
conditional spread, not a general noise level, and there was no probe increase.

**Independent audit.**
* `notes/codex_followup_result_audit.json`: an independent NumPy reconstruction from the raw CSVs and frozen
  coefficients, with a separately implemented balanced-template bootstrap. It reproduced:
  * the fit scale and all validation MSEs and λ selections;
  * the frozen test predictions (max difference 0);
  * the test MSEs, G and both bounds (within 1e-12);
  * the hashes and temporal order.
* Claude, read-only, no new analysis: recomputed from `test_predictions.csv`, `test_labels.csv` and the frozen s.
  * MSE(A), MSE(B), MSE(A_in), the fit-mean MSE, G, G_Taylor, the 76.28% fit-mean-referenced reduction and the Y
    mean all agree with `results.json` to every printed digit.
  * The lock file hashes match, and the lock (19:23:06) precedes the evaluation start (19:23:18).

## 6. Scope and limitations

* **Setting:** one model (GPT-2 small), one layer (9), one artifact (Day-3 PCA192 + bias, last token), one task
  (greater-than years), 4 test frames and 32 template clusters.
* **Probes:** the result is conditional on the 8 fixed probes. Their uncertainty is not covered by the template
  bootstrap.
* **α = 1.5:** an extrapolation/stress condition, not a natural activation.
* **Intervals:** nominal, approximate and conditional, with fixed frames.
* **Not input-only:** the secondary A_in is not input-only, because B contains exact local errors. Input-only
  prediction was not tested.
* **Not claimed:** no mechanism identification, circuit preservation, novelty, compression, or WeightWatcher/RG
  falsification or support.
* **Meaning of INCONCLUSIVE:** it is neither proof of the absence of an effect nor evidence that the target gain
  exists.

## 7. Independent audits, three-model closure and status

**Independent raw-data audits** (Codex and Claude only):
* `notes/codex_followup_result_audit.json`: the registered result (§5).
* `notes/codex_followup_contribution_audit.json`: Codex independently verified the post-hoc contribution
  arithmetic. That covers all 96 template×α cells, the α B-MSE weights, the net shares, the 23/9 template signs,
  the 64/32 cell signs, and the top-5 template shares: 81.03% of the net, 47.01% of the gross positive.
* Claude computed the decomposition (`notes/followup_posthoc_loss_contributions.md`) and reproduced the registered
  numbers read-only.
* The reviewers received result packets. They did not audit the raw files, and no raw-data verification is
  attributed to them.

**Post-hoc contribution check** (requested by GPT Pro; descriptive, from the locked raw CSVs only; no CI,
threshold, leave-out or causal claim; the primary decision is unchanged):
* The contributions reproduce G = 0.16230704 exactly.
* α = 1.5 holds 97.44% of the net gain, with B-MSE weight 0.594. The α = 0.5 share is −17.59%. The simple mean of
  the α gains would be 0.0374.
* Templates: 23 positive, 9 negative. The top 5 hold **81.03% of the net** and **47.01% of the gross positive**.
  N3|upheaval holds −34.69% of the net.
* Template×α cells: 64 positive, 32 negative. Gross +0.2968 vs −0.1344 in G units.
* Rows: 326 positive, 250 negative. One recipient, tes-0140, holds −37.06% of the net over its three α rows.
* This is heterogeneity in this test set's loss accounting, not a mechanism.

**Three-model closure** (the actual app conversations, relayed by Codex; paraphrased; all three received the
contribution numbers):
* **GPT Pro**:
  * It accepted that its own proposal did not meet the GO condition, and has no remaining objection.
  * INCONCLUSIVE stands. The 10% minimum gain was neither confirmed nor ruled out.
  * The positive point gain is in *predicting* the intervention-error norm. It is not a reduction of the model's
    error, and not a direction or circuit prediction.
  * The equivalence or superiority of Taylor vs exact was not tested.
  * The pooled G is not the simple mean of the α gains: groups with high B-MSE carry more weight.
  * The bootstrap uses 32 template clusters as its sampling units, not 576 independent examples.
  * Its reading of the decomposition (23/32 positive templates, top-5 net 81% and gross positive 47%, α = 1.5 net
    97.44%) is heterogeneity, not mechanism. It recommends closing this version.
* **Fable Extra**:
  * Closing is appropriate; it asked that its withdrawals stay on record.
  * Withdrawn after Codex's challenge:
    * the norm-only reading ("76% explained variance");
    * guaranteed attenuation;
    * "about 220 clusters needed / impossible to power".
  * Also challenged by Codex and not adopted here: a "no 95% support for any improvement" reading, which conflated
    the one-sided and two-sided statements, and probe-noise or overshoot explanations for the α heterogeneity.
  * A later net-share remark by Fable was rejected by Codex and is not adopted: a net share is neither a subgroup G
    nor an equivalence test.
* **Antigravity**:
  * No real remaining objection; close with a technical note after the descriptive decomposition.
  * Its earlier norm-only / "76% explained variance" reading was withdrawn after Codex's challenge.
  * Probe-noise or overshoot explanations for the α heterogeneity were challenged and are not adopted here.

**Status: CLOSED.** This version of the propagation audit is closed.
* It does not support an error-reduction, mechanism, novelty or WeightWatcher/RG claim.
* No new experiment, refit, probe increase or scope expansion is planned under this protocol.
* Any further work would need a new protocol and fresh splits.
* Stopping is a research-resource judgment, not a statistical impossibility.

## 8. Paths

* Design and history: `FOLLOWUP_DESIGN.md`, `notes/followup_debate.md`.
* Protocol and config: `followup/PROTOCOL_FOLLOWUP.md`, `followup/config/followup_v1.json`.
* Implementation and audit notes:
  * `notes/claude_followup_implementation.md`;
  * `notes/codex_followup_pre_registration_audit.json`;
  * `notes/codex_followup_result_audit.json`;
  * `notes/codex_followup_contribution_audit.json`.
* Post-hoc contribution decomposition: `notes/followup_posthoc_loss_contributions.md`, `.csv`, `_rows.csv` and the
  script `.py`.
* Locks: `followup/registration/` (registration, freeze, test prediction lock, evaluated marker).
* Runs: `followup/runs/`. The test results are in `20261030T192318Z_followup_evaluate_test/results.json`, the
  rendered report in `20261030T192512Z_followup_report/REPORT.md`, and the manifests in `followup/manifests/`.
