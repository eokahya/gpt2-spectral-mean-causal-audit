# Spektral bileşen ortalaması nedensel denetimi (GPT-2 small, blok 9 MLP)

[English version / İngilizce sürüm](README.md)

Düşük ranklı bir ağırlık kesmesinde atılan kısım, kalibre edilmiş ortalamasıyla (ya da kalibre edilmiş bir Gauss
beklenen değeriyle) değiştirildiğinde bir transformer'ın davranışının ne olduğunu inceleyen, önceden kayıtlı
(pre-registered) bir nedensel denetim.

Projenin protokolündeki çalışma başlığı: *When Can Spectral Components Be Replaced by Their Average? A Causal Audit
of Transformer MLPs.*

## Kısa sonuç

**Ana hipotez desteklenmedi.** Ana deneyin kilitli, dışarıda tutulmuş (held-out) testinde şu kurulum kullanıldı:
GPT-2 small, blok 9, MLP giriş matrisi SVD ile rank 192'ye kesilmiş, yalnız son token. Burada atılan kısmı kalibre
edilmiş bir **Gauss beklenen değeriyle** değiştirmek, modelin yıl değiştirme müdahalesine verdiği yanıtı düz bir
kalibre edilmiş **ortalamayla** değiştirmekten daha iyi değil, daha *kötü* yaptı:

* kayıtlı birincil istatistik P1 = RMS hata (Gauss) − RMS hata (ortalama) = **+0.0268, %95 GA [0.0235, 0.0302]**;
  bu, ortalama ile değiştirmenin hatasının yaklaşık %2.1'i;
* iki SVD varyantının toplam hatası da büyüktü (RMS yaklaşık 1.27–1.30; özgün yanıtın RMS'i 1.758).

Projenin nihai ortak kararı, Gauss/SVD'ye özgü tez için **NO-GO** oldu.

Kalibrasyondan sonra yapılan ikincil bir seçim daha iyi sonuç verdi. Kalibrasyon aktivasyonlarına uydurulmuş bir
**PCA** tabanı ve aynı rankta düz ortalama ile değiştirmede skaler yanıt hatası küçüktü (RMS 0.0684). Tek yönlü üst
sınırı (0.0780), önceden kayıtlı tolerans ε = 0.0897'nin altında kaldı. Bu dar, skaler bir sonuçtur. Altta yatan yanıt
*vektörlerinin* (MLP çıktısı ve logitler) yaklaşık %14–15'i yine de yeniden üretilemedi. Ayrıca PCA ancak kalibrasyon
verisi görüldükten sonra seçildi.

Sonraki iki takip çalışması bu tabloyu değiştirmedi. Biri **INCONCLUSIVE (kararsız)** bitti; diğeri yalnız geliştirme
aşamasında bir ölçümdü ve o kol kapatıldı.

Bu sonuç, spektral bileşenleri ortalama ya da Gauss ile değiştirmenin hiçbir zaman işe yaramayacağını **kanıtlamaz**.
Sınırlı bir sonuçtur: tek model, tek katman, tek rank, tek sentetik görev, held-out bölünme başına iki prompt
çerçevesi ve tek tür müdahale.

## Problem, gündelik dille

Büyük ağırlık matrisleri çoğu zaman yalnız en güçlü yönleri tutularak sıkıştırılır. Buna düşük ranklı ya da
"spektral" kesme denir ve genellikle SVD ile yapılır. Tutulan yönlerin dışında kalan her şey atılır. Yaygın bir onarım
yöntemi, atılan kısmı bir kalibrasyon metninde ölçülmüş **ortalama** değeriyle değiştirmektir; böylece katman
sistematik olarak kaymaz.

**Oyuncak örnek.** Bir MLP'nin tek bir gizli nöronunu düşünün. Girdisi bir toplamdır, `a = z + q`:

* `z`, tuttuğumuz yönlerden gelir;
* `q`, attığımız yönlerden gelir.

Nöronun çıktısı `gelu(a)`'dır. Kalibrasyon promptlarında bu nöron için `q`'nun ortalamasının 0.3, yayılımının
(standart sapma) 1.0 olduğunu varsayalım. Aday değiştirmeler şunlar:

| ad | indirgenmiş nöronun hesapladığı | fikir |
|---|---|---|
| `zero` | `gelu(z)` | düz kesme; `q`'yu unut |
| `bias` ("ortalama") | `gelu(z + 0.3)` | `q`'nun ortalamasını geri koy |
| `gauss` | standart normal `ε` üzerinden `gelu(z + 0.3 + 1.0·ε)` ortalaması | `q` Gauss gürültüsü olsaydı ortalama *çıktıyı* geri koy |
| `gauss_wo` | `gauss` gibi, ama ortalama ve yayılım yalnız ağırlıklardan | kalibrasyon verisi yok |

`gelu` eğri olduğu için Gauss sürümü ortalama sürümüyle aynı değildir. Umut, gerçek nöronla ortalamada daha iyi
örtüşmesiydi.

**Nedensel denetim neyi kontrol eder?** Modelin sıradan girdilerdeki çıktısını tutturmak (örneğin düşük KL
ıraksaması), indirgenmiş modelin *değişikliklere* özgün model gibi tepki verdiğini göstermez. Denetim, özgün modelin
mekanizmasının tepki verdiği bilinen bir görevi kullanır: GPT-2'nin şu tür promptlardaki "büyüktür" (greater-than)
davranışı:

> The drought continued from the year 1724 through the year 17

Özgün model olasılığın çoğunu 24'ten sonraki iki basamaklı yıllara verir. Şimdi başlangıç yılı farklı bir *donör*
prompt alalım ("… from the year 1737 through the year 17"). Donörün MLP girdisini son tokende alıcıya
kopyaladığımızda özgün modelin yıl tercihleri kayar. Skaler `T`, bu kaymayı donör-eksi-alıcı "büyüktür" örüntüsü
üzerine izdüşürerek ölçer.

Denetim **aynı fiziksel değişikliği** hem özgün modele hem her indirgenmiş modele uygular ve iki kaymayı karşılaştırır.
İyi bir değiştirme, özgün modelin yalnız müdahalesiz çıktısını değil, yanıtını da yeniden üretmelidir.

## Sınanan hipotez

Kayıtlı birincil soru (Gün 1 protokolü; Gün 3'e kadar değişmedi): SVD rank k = 192, blok 9, son-token kapsamında,
kalibre edilmiş Gauss değiştirmesi özgün modelin doğal yıl değiştirme yanıtını kalibre edilmiş ortalama
değiştirmesinden **farklı bir RMS hatayla** mı yeniden üretiyor? Yön önceden varsayılmadı. Altta yatan tez, atılan
SVD bileşeninin Gauss olarak ele alınmasının daha iyi ve mekanizmayı koruyan bir telafi olacağıydı.

## Yöntem ne yapıyor?

**Sade dille.** Blok 9 MLP'sinin giriş ağırlık matrisini en güçlü 192 yönüne indir. Kesilen kısmı, kalibrasyon
promptlarından bir kez öğrenilmiş sabit bir düzeltmeyle değiştir. Sonra, sona kadar mühürlü tutulmuş yeni promptlarda,
indirgenmiş modelin bir yıl değişimine tam model gibi tepki verip vermediğine bak.

**Kesin sürüm** (bkz. `PROTOCOL.md`, `PROTOCOL_DAY2.md` ve `PROTOCOL_DAY3.md`):

* **Model ve konum:** `openai-community/gpt2`, revizyon `607a30d783dfa663caf39e06633721c8d4cfcd7e`; sıfır tabanlı
  blok 9; MLP giriş izdüşümü `W_in = Wc^T` (HF `Conv1D` ağırlığı `[768, 3072]`). `W_out` hiç değiştirilmez.
* **Ayrıştırma:** girdi altuzayı izdüşürücüleri `P_R` (tutulan, rank k) ve `P_B = I − P_R` (atılan). Ön aktivasyon
  `a = z + q`; burada `z = W_in P_R x + b` ve `q = W_in P_B x`.
* **Telafiler:** `zero`, `bias` (`g(z+μ)`), `gauss` (`E g(z+μ+σε)`; olasılıkçı Gauss–Hermite, Gün 1'de 64, Gün 2'den
  itibaren 128 düğüm), `gauss_wo` (yalnız ağırlıklardan momentler). Momentler kalibrasyondan sonra sabitlenir.
* **Tabanlar:** `W_in`'in SVD'si (ana); kalibrasyon aktivasyonlarının PCA'sı (Gün 3'ten itibaren ana; kalibrasyondan
  sonra seçildi); beyazlatılmış SVD (duyarlılık); 5 sabit rastgele taban (boyut kontrolü).
* **Kapsamlar:** `last_token` (indirgenmiş ağırlıklar yalnız son pozisyonda; birincil) ve `full_layer` (tüm
  pozisyonlar).
* **Müdahale:** son tokende, `S ∈ {R, B, RB}` için `x' = x_rec + P_S(x_don − x_rec)`. Artık (residual) yol ve diğer
  tüm pozisyonlar olduğu gibi kalır.
* **Skor:** 100 iki basamaklı yıl tokeni J üzerinde `T = <Δlogits_J, v_rd> / ||v_rd||²`; burada
  `v_rd = centre(1[j > d] − 1[j > r])`. Hata, aynı fiziksel değişiklik için `e_T = T_indirgenmiş − T_özgün`.
* **İndirgenmiş model:** yoğun (dense) işlevsel bir rank-k denetimi (yoğun `P_R` ile özgün ağırlıklar). Bir sıkıştırma
  yöntemi **değildir**; hız ya da bellek iddiası yoktur.

## Neyle karşılaştırıldı? Karar kuralları

* **Temel çizgiler:** `gauss` ve `gauss_wo`, `zero` (düz kesme) ve `bias` (kalibre ortalama) ile karşılaştırıldı.
  Eşit boyutlu rastgele tabanlar, koordinat maskeleri (yalnız Gün 2), beyazlatılmış SVD ve PCA geometrik kontrol olarak
  kullanıldı.
* **Donör kontrolleri:** `same_example` (özdeşlik; tam sıfır), `same_year` (yıl değişmeden bağlam değişimi) ve
  `wrong_donor` (farklı bir yıl). Gün 2'de norm ve ön-aktivasyon eşlemeli sürümler eklendi.
* **Pozitif kontrol kapısı** (her bölünmede, indirgenmiş model hakkında herhangi bir iddiadan önce): ortalama PD > 0;
  ortalama `T_RB` (doğal) GA alt sınırı > 0; `T_doğal − T_same_year` GA alt sınırı > 0.
* **Önceden kayıtlı Gün 3 kuralları** (`PROTOCOL_DAY3.md`; testten önce `registration/DAY3_REGISTRATION.json` ve
  `registration/DAY3_FROZEN_*.json` içinde donduruldu):
  * **P1 (birincil):** `RMS(e_T^gauss) − RMS(e_T^bias)`; SVD, k = 192, son token, doğal donör, S = RB. İki yönlü %95
    tabakalı template-küme bootstrap'i (B = 10.000). Sonuç yalnızca bir yön işaretidir: GA > 0 ise Gauss daha kötü,
    GA < 0 ise Gauss daha iyi. Eşdeğerlik iddiası yoktur.
  * **S1 (ikincil):** `RMS(e^{pca,bias}) − RMS(e^{svd,bias})`; tek yönlü %97.5 üst sınır < 0 ise desteklenir.
  * **S2 (ikincil):** `RMS(e^{pca,bias})`; tek yönlü %97.5 üst sınır < ε ise tolerans içindedir.
    ε = 0.05 × kalibrasyondaki özgün RB RMS (1.7947) = **0.08973707820478917**. ε, kalibrasyon hataları görüldükten
    sonra önerildi.
  * Bonferroni yalnız {S1, S2} içinde uygulanır. P1'i de kapsayan küresel bir hata oranı güvencesi yoktur.
  * KL eşlemeli karşıtlıklar (Q1–Q3) validasyonda seçildi; testte betimseldir.

## Sonuçlar

### Ana deney: Gün 3 held-out testi (dondurulmuş, önceden kayıtlı)

Test bölünmesi: F4/F5 çerçeveleri, 192 çift, 18 template kümesi. Validasyon bölünmesi (F2/F3) yalnız dondurulmuş KL
seçimleri için kullanıldı; karşılaştırma için gösteriliyor. Her test çalıştırma dizininde, kullanılan kodun,
yapılandırmanın ve protokolün tam kopyası olan bir `snapshot/` bulunur.

| nicelik | validasyon (F2/F3) | **test (F4/F5)** | kaynak alan (`results.json`) |
|---|---|---|---|
| pozitif kapı | geçerli; PD 0.904; T_RB 1.776 [1.692, 1.866] | geçerli; PD 0.905; T_RB 1.706 [1.647, 1.765] | `positive_control.pd_mean_ci`, `.T_RB_natural_mean_ci` |
| bu bölünmedeki özgün RB RMS | 1.886 | 1.758 | `analysis.orig_RB_rms_this_split` |
| **P1** RMS gauss − RMS bias (SVD k192 LT) | +0.0416 [0.0376, 0.0457] | **+0.0268 [0.0235, 0.0302]** → Gauss daha kötü | `analysis.P1.diff_cluster_ci` |
| RMS e_T gauss / bias (SVD k192 LT) | 1.312 / 1.271 | 1.301 / 1.275 | `analysis.P1.rms_gauss`, `.rms_bias` |
| **S1** PCA-bias − SVD-bias (üst %97.5) | −1.197 (−1.123) | **−1.206 (−1.136)** → desteklendi | `analysis.S1.diff_cluster_upper` |
| **S2** RMS PCA-bias (üst %97.5), ε 0.0897'ye karşı | 0.0740 (0.0790) | **0.0684 (0.0780)** → tolerans içinde | `analysis.S2.rms_cluster_upper` |

Kaynaklar: `runs/20261030T164723Z_day3_test/results.json` (test) ve
`runs/20261030T164225Z_day3_validate/results.json` (validasyon).

Testte betimsel bağlam (k = 192, son token; karar ölçütü değil):

| nicelik | değer | kaynak |
|---|---|---|
| müdahalesiz KL(özgün‖indirgenmiş): SVD bias / SVD gauss / PCA bias | 0.142 / 0.132 / 0.0022 | test `results.json` → `unintervened[].kl_mean` |
| özgün yanıtın bölünmesi T_R / T_B: SVD | 0.396 / 1.263 | test `results.json` → `controls[]` (natural, native) |
| özgün yanıtın bölünmesi T_R / T_B: PCA | 1.683 / 0.016 | aynı |
| PCA-bias vektör hata oranı: yerel MLP çıktısı / merkezlenmiş logitler | 0.138 / 0.153 | `notes/codex_day3_vector_ratio_audit.json` → `results.test.last_token` |

Nasıl okunmalı:

* Gauss'un müdahalesiz KL'si ortalamadan *düşüktü* (0.132'ye karşı 0.142), ama yanıt hatası *daha yüksekti*. Müdahalesiz
  çıktıya daha iyi uyum, daha iyi nedensel yanıt anlamına gelmedi.
* SVD'de doğal yanıtın çoğu atılan altuzaydan geçiyor (yaklaşık 1.7'nin 1.263'ü T_B). Sabit bir telafi bunu yapısal
  olarak taşıyamaz: indirgenmiş modelin saf-B yamasına yanıtı tam olarak sıfırdır. Büyük SVD hatasına bu baskındır.
* PCA'da yanıtın neredeyse tamamı tutulan altuzaydan geçiyor (T_B 0.016); hatanın küçük olmasının nedeni budur.
* S2'nin geçmesi tüm vektörün, devrenin korunması ya da çift başına bir güvence **değildir**. Son-token kapsamında yerel
  yanıt vektörünün yaklaşık %14'ü, logit yanıt vektörünün yaklaşık %15'i yeniden üretilmiyor. Çerçeve bazında
  (betimsel), validasyon çerçevesi F3 ε'u aştı ve test çerçevesi F4 dar bir payla geçti (`DAY3_SUMMARY.md` §2).

P1'in aynı yönü Gün 1'deki keşif amaçlı kalibrasyon ölçümünde de görüldü: +0.035 [0.031, 0.038]
(`runs/20261030T153707Z_pilot/results.json` → `primary_candidate_exploratory`, last_token, k = 192). Bu kalibrasyon
verisidir, ana sonuç değildir.

### Keşif amaçlı ön aşamalar (yalnız kalibrasyon bölünmesi; doğrulayıcı değil)

* **Gün 1** (`SUMMARY.md`, `runs/20261030T153707Z_pilot/`): tümü kalibrasyon çiftlerinde ilk ölçümler. Atılan SVD
  altuzayı yıl yanıtının büyük bir kısmını taşıyor.
* **Gün 2** (`DAY2_SUMMARY.md`, `runs/20261030T160653Z_day2_run/`): geometrik kontroller. k = 192'de SVD bölünmesi
  T_R 0.429 / T_B 1.240 verirken PCA T_R 1.724 / T_B 0.0005 veriyor (`results.json` → `controls[]`). PCA'nın
  kalibrasyondan sonra Gün 3'e ana taban olarak eklenmesinin nedeni budur.

### Takip çalışması 1: sınırlı yayılım denetimi (ayrı protokol; takip olarak etiketli)

Soru: Gün 3'ün PCA192 + ortalama modeli için, yerel hatayı aşağı akış duyarlılıklarıyla ağırlıklandırmak (özellik
`D`), logit düzeyindeki hatanın *büyüklüğünü tahmin etmeye* güçlü bir 14 özellikli temel çizgi `B`'den daha çok
yardım ediyor mu? Bu, modelin hatasını azaltmak değil, hata normunu tahmin etmekle ilgilidir. Yeni N0–N9 çerçeveleri
ve yeni isimler kullanıldı. Test kümesi: 192 alıcı × 3 müdahale gücü α = 576 satır, 32 template kümesi, 4 çerçeve.

Önceden kayıtlı kural: `G = 1 − MSE(A)/MSE(B)` için tek yönlü %95 alt sınır > 0.10 ise GO; tek yönlü %95 üst sınır
≤ 0.10 ise hedef dışlanır; aksi halde INCONCLUSIVE.

| nicelik | değer | kaynak alan |
|---|---|---|
| MSE(B), temel çizgi | 3.4355e-4 | `primary.mse_B` |
| MSE(A) = B + D_exact | 2.8779e-4 | `primary.mse_A` |
| **G** | **0.1623** | `primary.G` |
| tek yönlü %95 alt / üst sınır | 0.00082 / 0.3280 | `primary.G_lower_one_sided`, `.G_upper_one_sided` |
| **karar** | **INCONCLUSIVE** | `primary.decision` |
| ikincil G_Taylor (betimsel) | 0.1319 (alt −0.0457, üst 0.3122) | `secondary_descriptive` |

Kaynak: `followup/runs/20261030T192318Z_followup_evaluate_test/results.json`. Özet: `FOLLOWUP_SUMMARY.md`.

İki sınır ayrı ayrı tek yönlü sınırlardır; iki yönlü bir aralık değildir. α başına nokta kazançları −0.26 (α = 0.5)
ile 0.27 (α = 1.5) arasında dağıldı; bunlar yalnız betimseldir. %10'luk hedef kazanç ne doğrulandı ne dışlandı.

### Takip çalışması 2: atılan durum yanıt tabanı kapısı (yalnız geliştirme; takip olarak etiketli)

Soru: alıcı bağlamı, tutulan durum ve tutulan müdahale sabitken, yalnız atılan bileşen aynı yıldaki 8 donör arasında
değiştirildiğinde yanıt ne kadar değişiyor? Cevap, atılan durumu göremeyen her tahminci için normalize edilmiş bir
hata tabanı `L`'dir. Önceden belirlenen hedef L0 = 0.05 idi. Bu, 48 grup üzerinde bir **geliştirme** ölçümüydü;
doğrulayıcı bir test değildi.

| nicelik | değer | kaynak |
|---|---|---|
| s² (yanıt ölçeği) | 1.0934 | `discarded_response_floor_gate_v3/runs/20261031T105820Z_gate_v3_develop/development.json` → `s2` |
| **L** | **0.0396** (L² = 0.0015673) | aynı → `Lhat`, `Lhat2` |
| betimsel bootstrap alt / üst (her biri tek yönlü) | 0.0382 / 0.0409 | aynı → `dev_bootstrap_descriptive` |
| planlama çıktısı (uygulanmadı) | POWER_PLAN_CANDIDATE, N = 192 | `discarded_response_floor_gate_v3/runs/20261031T105841Z_gate_v3_power/power.json` |

Bu sonlu tasarımdaki taban hedefin altında kaldı. Kol, doğrulayıcı bir çalıştırma yapılmadan kapatıldı
(`notes/discarded_response_floor_gate_v3_results_2026-10-31.txt`, Türkçe). Bu deney renormalizasyon grubu (RG) ya da
WeightWatcher fikirlerini sınamadı.

### Yerini başkasına bırakmış ya da geçersiz çalıştırmalar (kayıt olarak tutuldu, sonuç değil)

* `discarded_response_floor_gate/` **v1**: bir donör-isim örtüşme (aliasing) sorunu nedeniyle, hiçbir sonuçtan önce
  yerini başka tasarıma bıraktı.
* `discarded_response_floor_gate/` **v2** geliştirme çalıştırması `20261031T103329Z_gate_develop`:
  **EVALUATION_INVALID**. 48 özdeşlik kontrolünün 15'i sabit 1e-4 eşiğini aştı (en çok 2.136e-4); kaynağı MPS float32
  padding/düzen duyarlılığı olarak belirlendi. v3 bunun teknik yeniden ölçümüdür.
* `runs/20261030T153332Z_quadcheck` (başarısız: MPS float64 dönüşümü) ve `runs/20261030T153344Z_quadcheck`
  (32 düğüm yakınsamadı).
* `followup/runs/20261030T192217Z_followup_fit` (model yüklenmeden başarısız: sandbox içinde MPS yoktu) ve daha önceki
  takip smoke çalıştırması `20261030T191925Z_followup_smoke`.

## Proje neden sürdürülmedi?

* Kayıtlı birincil sonuç held-out veride tezin aleyhine çıktı. Aynı yön kalibrasyon, validasyon ve testte görüldü.
* Olumlu PCA sonucu dar (tek görev, bölünme başına iki çerçeve, skaler ölçüt, kalibrasyondan sonra seçilmiş taban).
  Açıklaması da büyük ölçüde geometrik: PCA altuzayı yanıtı doğrudan taşıyor.
* Takip 1 INCONCLUSIVE çıktı. Gözden geçirenler ve koordinatör onu büyütmek yerine kapattı. `FOLLOWUP_SUMMARY.md`'ye
  göre durmak istatistiksel bir imkânsızlık değil, bir araştırma kaynağı kararıdır.
* Takip 2, geliştirme tasarımında hedefin altında bir taban buldu ve kol kapatıldı. Yeniden açmak için yeni,
  gerekçelendirilmiş bir hipotez ve yeni bir kayıt gerekir.

## Hangi kanıt hangi türden?

**(a) Bilimsel deneyler** (yukarıdaki sonuçlar): Gün 1 pilotu ve Gün 2 çalıştırması (keşif amaçlı, yalnız
kalibrasyon); Gün 3 validasyonu ve testi (önceden kayıtlı; ana sonuç test); takip 1 (önceden kayıtlı, ayrı
bölünmeler); takip 2 v3 (yalnız geliştirme).

**(b) Mühendislik kontrolleri.** Bunlar kodun söylediğini yaptığını doğrular; hipotezi sınamaz.

* Birim testleri: `tests/` (66 geçti, `logs/pytest_day3_20261030T164003Z.log`), `tests_followup/` (27 geçti,
  `followup/logs/pytest_followup_20261030T192123Z.log`) ve `discarded_response_floor_gate_v3/tests/` (48 geçti,
  `discarded_response_floor_gate_v3/logs/pytest_gate_v3_20261031T105522Z.log`). İlk test çalıştırmasında 5 hata vardı
  (`logs/pytest_20261030T153120Z.log`); kayıt olarak tutuldu.
* Smoke, kuadratür ve benchmark çalıştırmaları: `*_smoke`, `*_quadcheck`, `*_bench` ve `*_calibrate` dizinleri.
* Sayısal tanılar: `notes/gate_numerical_diagnostic/`.
* SHA-256 bağlı kayıt ve dondurma dosyaları: `registration/`, `followup/registration/` ve kapı kayıt klasörleri.

**(c) Literatür ve yargı yorumları.** Her aşamada yapay zekâ modelleri (GPT Pro, Antigravity/Gemini, Claude/Fable)
sonuçları gözden geçirdi; Codex denetledi. Bu kayıtlar yargıdır, ölçüm değildir. Codex'in sayıları ham dosyalardan
bağımsız olarak yeniden hesapladığı denetimler `notes/codex_*.json` içindedir. Dahil edilen inceleme ve karar notları:
`notes/three_model_review_day1.md`, `notes/claude_*.md`, `notes/day3_posthoc_diagnostic.md` (post-hoc, betimsel) ve
`notes/followup_posthoc_loss_contributions.md` (post-hoc, betimsel). Bazı inceleme kayıtları özel sohbet bağlantıları
içerdiği için **dahil edilmedi** (bkz. "Köken ve lisans").

## Sınırlılıklar

* Tek model (GPT-2 small), tek katman (blok 9), tek matris (`W_in`), tek çapa rank (192) ve tek sentetik görev
  (büyüktür yılları).
* Held-out bölünme başına yalnız 2 çerçeve; küme yapısı zayıf. Aralıklar nominal, yaklaşık ve uydurulmuş taban ile
  momentlere koşulludur.
* ε, kalibrasyon hataları bilinerek seçildi. PCA kalibrasyondan sonra seçildi.
* `full_layer` kapsamında, kapsam ile moment havuzu karışıktır (farklı kalibrasyon havuzları).
* Müdahale yalnız son tokendeki MLP girdisine yama yapar. Uç nokta yeniden kurulumu PCA sonucu için güçlü bir açıklama
  adayıdır, kanıtlanmış bir mekanizma değildir.
* Genel metin ya da görev dışı değerlendirme, sıkıştırma ya da hız ölçümü yapılmadı.
* Takip 1, belirsizliği bootstrap'e dahil edilmeyen 8 sabit rastgele probe'a dayanır. α = 1.5 bir dışdeğerlemedir.
* Takip 2, doğal aktivasyon manifoldu üzerinde olması gerekmeyen sentetik hibrit girdiler kullandı. Kör değildi
  (v2 çıktıları görülmüştü).

## Depo haritası

| yol | içerik |
|---|---|
| `PROTOCOL.md`, `PROTOCOL_AMENDMENT_v1.1.md` | Gün 1 kayıtlı protokolü ve değişikliği |
| `PROTOCOL_DAY2.md`, `PROTOCOL_DAY3.md` | Gün 2 (geometrik kontroller) ve Gün 3 (held-out) protokolleri |
| `SUMMARY.md`, `DAY2_SUMMARY.md`, `DAY3_SUMMARY.md` | aşama raporları; `DAY3_SUMMARY.md` nihai ana rapor |
| `FOLLOWUP_DESIGN.md`, `FOLLOWUP_SUMMARY.md` | takip 1 tasarım kaydı ve nihai rapor |
| `docs/PROJECT_NOTES.md` | projenin özgün README dosyası (değiştirilmedi) |
| `config/` | tüm parametreler (`pilot_v1.json`, `day2_v1.json`, `day3_v1.json`) |
| `registration/` | SHA-256 kayıtları ve Gün 3 dondurma dosyaları |
| `manifests/`, `manifests_day3/`, `metadata/` | prompt bölünmeleri, çiftler, mühürlü held-out promptlar, devam maskesi |
| `calibration_day3/calib_index.json` | dondurulmuş kalibrasyon artefaktının dizini ve hash'leri (artefaktın kendisi dahil değil) |
| `scp/`, `run.py`, `run_day2.py`, `run_day3.py` | ana kod ve komut satırı arayüzleri |
| `tests/`, `pytest.ini` | ana birim testleri |
| `runs/<utc>_<aşama>/` | her çalıştırma için bir dizin: `run_manifest.json`, `log.txt`, sonuçlar, `snapshot/` |
| `followup/`, `run_followup.py`, `tests_followup/` | takip 1 kodu, protokolü, yapılandırması, manifestleri, kaydı, çalıştırmaları |
| `discarded_response_floor_gate/` | takip 2 v1/v2 (yerini başkasına bırakmış / geçersiz) |
| `discarded_response_floor_gate_v3/` | takip 2 v3 kodu, protokolü, çalıştırmaları |
| `notes/` | denetimler, post-hoc tanılar, dahil edilen inceleme notları |
| `logs/` | pytest ve stdout günlükleri |
| `env/`, `requirements.txt`, `requirements.lock.txt` | kaydedilmiş ortam |

## Yeniden üretme

**Kaydedilmiş ortam** (`env/hardware_and_versions.txt`, `requirements.txt`, `requirements.lock.txt`): Apple M2 Max,
32 GB, macOS 26.6.2; Python 3.13.0; torch 2.15.0 (MPS); transformers 5.22.1; numpy 2.5.4; scipy 1.18.1; tokenizers
0.23.3; safetensors 0.8.0; huggingface_hub 2.4.0; pytest 9.2.0. Sabitlenmiş tam küme `requirements.lock.txt`
içindedir.

**Model indirmesi (tek dış bağımlılık):** `openai-community/gpt2`, revizyon
`607a30d783dfa663caf39e06633721c8d4cfcd7e` (ağırlıklar ve tokenizer). `scp/__init__.py`,
`HF_HOME=<depo>/.cache/hf` ve `HF_HUB_OFFLINE=1` ayarlar; bu yüzden modelin önceden bir kez indirilmesi gerekir. Kod
MPS bekler ve başka bir cihaza sessizce geçmeyi reddeder.

**Proje belgelerinde kayıtlı komutlar** (`docs/PROJECT_NOTES.md`, `SUMMARY.md` §7, `DAY3_SUMMARY.md` §7,
`run_followup.py` docstring'i ve `DEV_PROTOCOL_v3.md` §V3). Depo kökünden çalıştırın:

```bash
python3.13 -m venv .venv
PIP_CACHE_DIR=$PWD/.cache/pip .venv/bin/pip install -r requirements.lock.txt
HF_HOME=$PWD/.cache/hf .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('openai-community/gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e')"
.venv/bin/python -m pytest -v
.venv/bin/python run.py splits      # aynı manifestleri yeniden üretir (tohumlu; hash'ler kayıtla eşleşmeli)
.venv/bin/python run.py quadcheck
.venv/bin/python run.py smoke
.venv/bin/python run.py bench
.venv/bin/python run.py pilot
.venv/bin/python -m scp.report runs/<pilot_run_id>
.venv/bin/python run_day3.py smoke
.venv/bin/python -c "import scp; from scp.common import load_config; from scp import day3; day3.verify_frozen(load_config('config/day3_v1.json')); print('ok')"
.venv/bin/python run_followup.py <aşama>   # prepare -> smoke -> register -> fit -> validate -> freeze -> predict-test -> evaluate-test -> report
.venv/bin/python -m pytest discarded_response_floor_gate_v3/tests -v -p no:cacheprovider
```

**Yollar.** Kod dosyaları depo köküne göre çözer; bu yüzden bir klon herhangi bir yerde durabilir. Proje belgeleri
`cd /Users/emre/Documents/Codex/ML_Projects/spectral_causal_pilot` ile başlar ve
`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13` kullanır; ikisini de kendi klon yolunuz ve Python
3.13 ile değiştirin. Dondurulmuş kanıt dosyaları (`runs/*/run_manifest.json`, `runs/*/log.txt`, takip ve kapı
çalıştırma manifestleri, bazı günlükler, notlar ve özetler) özgün makinenin mutlak yollarını (`/Users/emre/...`)
içerir. Kayıtlara hash ile bağlı oldukları için kanıt olarak **değiştirilmeden** tutuldular. Bunları düzenlemeyin;
yeni çalıştırmalar yeni dizinlere yazar.

**Karşılaşacağınız korumalar.** Aşamalar kayıtların, dondurmaların, manifestlerin ve bir kez yazılan işaret
dosyalarının üzerine yazmayı reddeder. Hash'e bağlı herhangi bir dosya değiştiyse test aşamaları reddeder. Gün 3
validasyon ve testi, takip 1 ve takip 2 v3, **dahil edilmeyen** `calibration_day3/calib_artifact.pt` dosyasına
(117 MB) ihtiyaç duyar. `run_day3.py calibrate` bu dosyayı kalibrasyon havuzundan yeniden kurar, ama sonraki aşamalar
SHA-256'sını `calibration_day3/calib_index.json` ve kayıtlarla karşılaştırır. Başka donanım ya da yazılımda yeniden
kurulan dosya bit düzeyinde aynı olmayabilir; o zaman bu aşamalar reddeder.

**Yalnız dahil edilen dosyalardan yeniden hesaplanabilenler (model yok, yeniden çalıştırma yok):**

| raporlanan sayı | yeniden hesaplama kaynağı |
|---|---|
| Gün 3 testi P1, S1, S2 nokta değerleri; yöntem başına RMS e_T; S'ye göre betimsel ayrıştırma | `runs/20261030T164723Z_day3_test/reduced_rows.csv` (sütunlar `basis, method, scope, k, S, template, e_T, T_orig`). Yukarıdaki P1/S2 nokta değerleri bu dosyadan yeniden türetildi ve `results.json` ile birebir eşleşti. Bootstrap GA'ları için `PROTOCOL_DAY3.md` §5'teki tabakalı template bootstrap'i gerekir (`scp/day3.py` içinde uygulanmış). |
| Gün 3 testi müdahalesiz KL ve PD; dondurulmuş çiftlerin testteki KL farkları | `runs/20261030T164723Z_day3_test/unperturbed_rows.csv` |
| Gün 3 pozitif kapısı (test ve validasyon) | `runs/*_day3_{test,validate}/gate_rows.csv` |
| Gün 1 pozitif kontrolü (doğal / aynı yıl / yanlış donör için PD ve T) | `runs/20261030T153643Z_smoke/smoke_per_pair.npz` ve `runs/20261030T154234Z_smoke/smoke_per_pair.npz` (her biri 192 değerli dört float64 dizi) |
| Takip 1 MSE(A), MSE(B), G ve α / çerçeve başına değerler | `followup/runs/20261030T192301Z_followup_predict_test/test_predictions.csv` + `followup/runs/20261030T192318Z_followup_evaluate_test/test_labels.csv` + `followup/registration/FOLLOWUP_FROZEN.json` içindeki dondurulmuş `s` + `followup/manifests/test_recipients.csv`. G = 0.16230704 bunlardan yeniden türetildi. |
| Takip 1 post-hoc katkı payları | `notes/followup_posthoc_loss_contributions_rows.csv`, `.csv`, `.py` |
| Takip 2 v3 L, L², s² | `.../20261031T105820Z_gate_v3_develop/groups_U.csv` (L = 0.039589 yeniden türetildi) ve ham J-logitleri `raw_J_logits.pt` (model çıktılarından oluşan bir torch tensör dosyası, ağırlık değil; açmak için torch gerekir) |
| Takip 2 v3 güç tablosu | `.../20261031T105841Z_gate_v3_power/power_rows.csv`, `power.json` |

**Dahil edilen dosyalardan yeniden hesaplanamayanlar:**

* Gün 1 ve Gün 2 çift başına sayıları: satır düzeyindeki CSV'ler (her biri 13–24 MB) boyut nedeniyle çıkarıldı. Bu
  aşamaların `results.json` ve `summary_tables.md` dosyaları dahildir.
* Gün 3 validasyonunun çift başına satırları ve Gün 3 testinin `control_rows.csv` dosyası (geometrik kontrol
  tabloları ve `notes/day3_posthoc_diagnostic.md`'nin bir kısmı için gerekli): boyut nedeniyle çıkarıldı. Toplamlar
  `results.json` içindedir.
* Takip 1 probe VJP tensörleri ve takip 2 aktivasyon tensörleri: çıkarıldı. Yukarıda raporlanan sayılar için gerekli
  değiller.

## Kaynaklar

Projenin kendi protokolünde (`PROTOCOL.md`, `PROTOCOL_AMENDMENT_v1.1.md`) atıf yapılanlar, orada verildiği biçimde:

* Greater-than devresi, Hanna ve ark. 2023: arXiv:2305.00586. Görev biçimi ve PD tanımı buradan alındı; yeniden üretim
  iddiası yoktur.
* LASER: arXiv:2312.13558
* Beyond Components: arXiv:2511.20273 (protokol, bu çalışmanın Ek B.2'sinin zaten sınıf-ortalaması değişimleri
  yaptığını not eder)
* SVD-LLM: arXiv:2403.07378
* Optimal Ablation: arXiv:2409.09951

Protokoller ayrıca WeightWatcher'dan (spektral kesim `k*`, yapılmadı olarak listelenmiş) ve Wilson RG'den (açıkça
sınanmadı) atıf vermeden söz eder.

## Köken ve lisans

* Yerel bir proje klasöründen (`spectral_causal_pilot`, 2026-10-30 ile 2026-10-31 tarihli) kopyalandı. Klasör bir git
  deposu değildi. Kod ve testler, kaydedilen çalıştırmalarda kullanılan dondurulmuş sürümlerdir.
* Proje belgelerine göre takip kodu ve testleri bir yapay zekâ kodlama ajanı (Claude Code, Opus 5.5 High) tarafından
  yazıldı; Codex koordinatör ve bağımsız denetçi olarak çalıştı; sonuçlar yapay zekâ modelleri (GPT Pro,
  Antigravity/Gemini, Claude/Fable) tarafından gözden geçirildi.
* Bu açık kopyaya dahil edilmeyenler:
  * Python sanal ortamı, Hugging Face ve pip önbellekleri, `__pycache__`;
  * küçük bir çıktı-logit dosyası dışında torch tensör dosyaları (`*.pt`);
  * yukarıda listelenen büyük satır düzeyi CSV'ler;
  * özel sohbet oturumu bağlantıları içeren ya da sohbet promptlarının veya dökümlerinin birebir kopyası olan inceleme
    ve koordinasyon notları: `notes/three_model_review_day2.md`, `notes/three_model_review_day3.md`,
    `notes/three_model_review_day3_prevalidation.md`, `notes/followup_debate.md`,
    `notes/research_direction_rg_2026-10-31.md`, `notes/gpt_pro_*`, `notes/discarded_gate_coordination_20261031.json`,
    `notes/gate_v3_actual_review_record_20261031.json` ve
    `notes/gate_v3_technical_remeasurement_instruction_20261031.txt`.

  Dahil edilen bazı belgeler hâlâ bu yollara atıf yapar. Bu notlardaki kararlar `DAY3_SUMMARY.md`,
  `FOLLOWUP_SUMMARY.md`, `discarded_response_floor_gate_v3/DEV_PROTOCOL_v3.md` ve
  `notes/discarded_response_floor_gate_v3_results_2026-10-31.txt` içinde özetlenmiştir. Bu çıkarmalar nedeniyle, tüm
  korunan dosya kümesini kapsayan hash doğrulama komutları (örneğin takip 2'deki `verify-protected`; onun
  `protected_manifest.json` dosyası `calib_artifact.pt`, `*.pt` tensörleri ve satır düzeyi CSV'leri listeler) bu
  kopyada eksik dosya bildirir.
* Lisans dosyası yoktur; aksi belirtilmedikçe tüm haklar depo sahibine aittir. Yeniden kullanmadan önce sahibiyle
  iletişime geçin.
