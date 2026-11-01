# Claude — koşullu GO / itiraz (v1.1, 2026-10-30)

**Dar yorum için koşullu GO:** "Bu kalibrasyon pilotunda, MLP9 W_in'in atılan B alt uzayı yıl-değişimi
yanıtının büyük bir kısmını taşıyor (B, R'yi yalnız k=96/192'de aşıyor). Atılan girdiye bağlı olmayan sabit moment
telafileriyle indirgenmiş modelin saf-B müdahalesine yanıtı yapısal olarak sıfır (çıktı yine z'ye bağlı).
Yöntemler arası RB farkları ikincil. Bu, doğal RB için evrensel, indirgenemez bir hata alt sınırı değil: R'deki
aşırı yanıt bu kaybı dengeleyebilir." *(2026-10-30, Codex'in dil düzeltmesiyle güncellendi.)*

**Koşullar / itirazlar:**
1. RB sıralaması kısmen aritmetik bir artefakt. Her çiftte `e_RB = e_R − (T_B^orig + I^orig)` sağlanıyor
   (en büyük kalıntı 1.9e-5). Bu yüzden R'de aşırı yanıt veren bias, B kaybını kısmen iptal edip RB'de daha
   iyi görünüyor. Bu bir özdeşlik, hipotez değil. Gauss'un bias'tan "kötü" olduğu yorumu yapılmamalı.
2. R'de gauss ≤ bias sonucu ikincil bir okuma. Gauss eğim/kazanç mekanizması ölçülmedi; kanıt diye sunulmamalı.
3. Karar (üç reviewer ile ortak, 2026-10-30): RB birincil aday (uçtan uca geri-kazanım auditi), R ikincil
   (işaretli hatasıyla telafi sadakati). ε ve spesifik confirmatory rank henüz dondurulmadı; validasyondan önce
   dondurulmalı.
   Bu karar test sonucuna göre verilmemeli.
4. Özgüllük iddiası yok: norm-eşli wrong-donor, PCA, rastgele alt uzay, whitened SVD ve resample henüz yok.
5. 2 frame / 18 küme: küme CI'ları frame'ler arası belirsizliği temsil etmiyor.
6. Reduction yalnız işlevsel bir audit; sıkıştırma iddiası yok.

Yeni deney başlatılmadı.
