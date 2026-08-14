# ASU_SLAYER — bağımsız denetim raporu

Kapsam: `ASU_SLAYER/` (policy, scoring, board, search, parallel, benchmark)
ve `submission/` (contract, fetch, validate) — 2.529 satırın tamamı okundu,
motor eşitliği doğrulandı, dört ayrı test koşusu yapıldı. Motor: bu kopyanın
`monopoly_game_engine`'i kaynak repo ile **birebir aynı** (env/state/actions/
constants/agents_fixed); `ASU_FROZEN_TEACHER` politika matematiği aynı
(core.py yalnızca sonradan eklenen hızlandırma altyapısından yoksun, spec.py
ve hash birebir).

Tüm koşular: `PYTHONHASHSEED=0`, seat = seed % 4, `max_steps=20000` (oyunlar
motorun 200-tur kuralıyla doğal biter), Wilson %95 CI.

## 1. Ölçüm sonuçları

| Koşu | n | Sonuç | Not |
|---|---|---|---|
| slayer-v1 vs fixed-b/d/e (güçlü saha), seeds 962000–963999 | 2000 | **50.55%** [48.36, 52.74] | illegal aksiyon: 0 |
| — aynı seed'lerde Candidate D baseline (38.85%) ile eşleştirilmiş | 2000 çift | **+11.70pp** [+9.30, +14.10], McNemar p=4.3e-21 | |
| — aynı seed'lerde arm D (41.80%) ile eşleştirilmiş | 2000 çift | **+8.75pp** [+6.30, +11.15], p=4.3e-12 | |
| slayer-v1 vs 3× asu-value-v1, seeds 966000+ | 120 | **15.00%** [9.7, 22.5] | parite %25 — ALTINDA |
| slayer-rollout-v1 vs güçlü saha | 155 | 43.2% [35.7, 51.1] | aynı seed'lerde greedy 46.5%, discordant +17/−22 — arama kazanç ÜRETMİYOR |
| determinizm (aynı seed iki koşu) | 1 | byte-aynı | RNG'ye dokunmuyor |

Güçlü sahada slayer-v1 gerçek ve büyük bir sıçrama: NW-exact tez bu skor
kuralında çalışıyor (full_group %43.6, ortalama NW 17.599; kayıpların %46'sı
iflas, kalanı NW geriliği).

## 2. KRİTİK — README'nin ana iddiası bu ölçümde tutmadı

README "Why it beats the frozen teacher" diyor ve `TRAINING_RESULTS.md`'deki
bir marja atıf yapıyor; fakat **bu kopyanın TRAINING_RESULTS.md'sinde slayer
geçmiyor** ve repo'da `slayer_results/` yok — iddianın kanıtı commit'li değil.

Benim ölçümüm (3× asu-value-v1, n=120 seat-dengeli, doğal bitiş): slayer-v1
**15.00% [9.7, 22.5] — paritenin (%25) anlamlı derecede altında**, %85 iflas,
%100 eliminasyon. Üstelik oyunların %64'ünde tekel kurmayı BAŞARIYOR —
tekeli kuruyor, sonra kira baskısı altında batıyor: kaybettiren şey grup
edinememek değil, aşırı-yayılma. Mekanizma net: ASU sahasında oyunlar kısa ve rant
baskılı; slayer'ın "her dolar NW'ye" agresif harcaması + `expected_game_length
= 45` kalibrasyonlu rezervi bu rejimde yetersiz nakit bırakıyor. NW-exact tez
**kapak (round-200 NW hükmü) rejiminde** kazanıyor, **eliminasyon rejiminde**
kaybediyor — ve ASU'ya karşı oyunlar eliminasyon rejiminde.

Öneri: iddiayı ya ölçümle belgeleyin (config sweep'in kazanan konfigürasyonu
+ committed report.json) ya da README'yi "fixed sahada güçlü, ASU sahasında
doğrulanmadı" olarak düzeltin.

## 3. Doğrulanmış bug'lar

### 3a. Rezerv kapısı sıfır maliyetli takasları da bloke ediyor (repro'lu)
`_affordable(env, cost, reserve)` = `cost ≤ cash AND cash − cost ≥ reserve`.
`cost=0` olan işlemler (bedava deed takası) için bile `cash ≥ reserve` şartı
aranıyor. Sentetik repro (seed 970000 tabanlı): iki orange'ı olan slayer'a
rakip, **tekelini tamamlayacak deed'i bedavaya** teklif ediyor —
`cash=100` → ACCEPT, `cash=0` → **DECLINE**. Aynı kapı `_investments`
içindeki takas önerilerini de keser: 40 oyunluk sayımda cash < reserve
yüzünden **641 kez** pozitif-kazançlı takas önerisi bastırıldı.
Düzeltme: `cost == 0` (veya `cash_requested == 0`) yolunda rezerv şartını
atlamak; kabul tarafında `_incoming_trade_action`'daki
`self._affordable(float(offer.cash_requested), reserve)` çağrısı da aynı
düzeltmeyi ister.

### 3b. Ölü mekanizma: `active_liquidation` default kapalı
`_raise_cash_action` docstring'i "kararların yarısı rezervin altında geçiyor;
harcamayı reddetmek kurtarmaz, sadece nakit toplamak kurtarır" diye
gerekçelendiriyor — ama `SlayerConfig.active_liquidation = False` ve
`CONFIG_GRID` bu bayrağı hiç süpürmüyor. Yani savunulan kurtarma mekanizması
**hiç çalışmıyor** (kendi ölçümüm fixed sahada kararların %3.6'sı rezerv
altında; ASU sahasında iflas %87 — bayrağın asıl işe yarayacağı yer orası).
Açıksa da hafif churn var: bir oyunda 8 mortgage / 10 unmortgage gözledim —
histerezis (ör. rezervin 1.2×'i altında mortgage, 0.8× üstünde unmortgage) yok.

### 3c. Belgelenen sonuç dosyası eksik
README `TRAINING_RESULTS.md` marjına atıf yapıyor; dosyada slayer yok
(bkz. §2). Doküman-kod tutarsızlığı.

## 4. Tasarım zayıflıkları — geliştirilebilir noktalar

1. **Rezerv ufku yanlış rejime kalibre** (`expected_game_length=45.0`).
   Fixed sahada doğal oyunlar 130–170+ tur sürüyor (kayıplarda p50 adım
   6.712). Tur 45'ten sonra ufuk `min_horizon=4`e sabitleniyor → tur-başına
   %91.5 kantil, kalan ~100+ tur boyunca kümülatif iflas riskini sistematik
   az fiyatlıyor. Güçlü sahadaki kayıpların %46'sı iflas. Ufku "kalan tur
   tahmini"ne (ör. 200 − round, saha-koşullu) bağlamak doğal düzeltme.
2. **Öneri spam'i**: 40 oyunda 14.493 takas önerisi (≈362/oyun), kabul 8
   (%0.06). README cash-tekliflerini tam da bu gerekçeyle kaldırmış; aynı
   mantık swap önerilerine uygulanmamış. Karar bütçesinin büyük payı boşa
   gidiyor (sonuca etkisi nötr olsa da OOT slotunu meşgul ediyor ve rakip
   cevap döngüsüne adım harcatıyor). Aynı çifte tekrar teklif için cool-down
   yeterli olur.
3. **Kıskançlık (envy) kuralı**: `mine > theirs` şartı, bize devasa artı olan
   ama rakibe daha çok kazandıran her teklifi reddediyor (40 oyunda kabul
   184/8.383 = %2.2). Sıfır toplamlı olmayan bir oyunda mutlak-kazanç yerine
   göreli-kazanç maksimizasyonu; 4 oyunculu oyunda "theirs" tek rakip değil —
   üçüncü tarafın zararı hesaba girmiyor. En azından `mine > 0 ve theirs`
   payını denial ağırlığıyla tartmak denenebilir.
4. **Rollout araması maliyetine değmiyor**: `SlayerRolloutV1` karar başına
   ≤24 `deepcopy(env)` + 44'er adım — oyun başına ~3-4 dakika (greedy'nin
   ~200×'i). n=155 aynı-seed ölçümünde greedy 72/155 kazanırken rollout
   67/155 (discordant +17/−22): ~200× hesaba karşılık sıfır, muhtemelen
   negatif katkı. `_fast_copy_env` benzeri hafif klon + ASU_FROZEN_TEACHER'daki
   hazır altyapı varken deepcopy pahalı. Ayrıca:
   - rollout rakipleri hep SlayerV1 (self-play bias; fixed/ASU rakip
     modellenmiyor),
   - `SearchConfig.seed=0` sabit → tüm oyun boyunca aynı 6 zar dizisi
     (aday-karşılaştırma için CRN doğru, ama karar-arası korelasyon
     sistematik önyargı üretir; karara `env.round` tuzu katmak yeter),
   - `depth=44` ≈ 3-4 tur — "üç tur sonrasını görememe" motivasyonunun tam
     sınırında.
5. **`development_outlook` asimetrisi**: bloklu gruptaki deed alışta 0.25×
   iskonto edilirken `disposal_loss` iskontosuz — takas değerlemesi alış/veriş
   tarafında farklı ölçek kullanıyor. NW-exact tezle tutarlı ama "stratejik
   ölülük" alışta var satışta yok; bloklu deed'leri elden çıkarma isteksizliği
   üretir.
6. **Config doğrulama eksik**: `target_survival`/`min_horizon`/`build_reserve_
   fraction` doğrulanıyor; `auction_value_fraction=-3`, `denial_fraction=99`,
   `trade_margin=-1`, `reserve_floor=-500` sessizce kabul (sweep'te yanlış
   grid sessiz çöp üretir).
7. **`trade_margin` işlevsiz**: öneri sıralamasında tüm adayları aynı çarpanla
   çarpıyor (sıra değişmez), eşiği yok — ölü knob.
8. **Vergi/kira dışı nakit şokları rezervde yok**: income/luxury tax (200/100)
   `rent_quantile`'a girmiyor. Motor vergiyi nakde clamp'lediği için iflas
   riski yaratmaz ama rezervi sessizce deler; kantil hesabına eklemek ucuz.
9. **Jail eşiği mutlak dolar** (`jail_exposure_threshold=95.0`): oyun
   ilerledikçe beklenen rant 95'i hep aşar → geç oyunda "hep hapiste kal"a
   çöker; erken oyunda hep çık. Board gelişmişliğine oranlamak (ör. medyan
   rantın katı) daha sağlam.

## 5. Edge case'ler

- **`_auction_action`, `square is None`**: PASS dönüyor — güvenli. Ama PASS
  legal listede her zaman var mı kontrol edilmiyor (motor auction menüsünde
  PASS'ı hep verir — bugün güvenli, motor değişirse kırılgan).
- **`USE_GOOJ_CARD` dalı ölü kod**: bu rulesette kart destesi yok, GOOJ hiç
  oluşmaz (PPO_PLUS_RULES). Zararsız.
- **`_debt_action` yalnız POST_ROLL'da**: motor rescue menüsünü sadece
  post_roll'da ürettiği için bugün doğru; faz makinesi değişirse sessizce
  `min(legal)`e düşer (o da `DO_NOTHING`/`END_TURN` olabilir).
- **`_investments` tek aksiyon döndürür**: tur içinde çok adımlı planlama yok
  (motor zaten karar-başına-aksiyon istiyor; sıralı greedy yeterli). Sorun
  değil, not.
- **`rent_quantile(turns>1)`**: `_landings` çok-tur dağılımı turlar arası
  MUTLAK kira değişimini (rakibin bu arada ev dikmesi) göremez — 1 turda
  kullanıldığı için bugün etkisiz.
- **`benchmark.py evaluate_lineup` bağımlılığı**: kaynak repo'daki
  `evaluate.py`'a iki policy ID kaydı gerektiriyor; bu kopyada mevcut, ama
  paket tek başına taşındığında `benchmark.py` kırılır (README bunu belirtir).

## 6. `submission/` harness'ı — ayrı ciddiyette bir açık

**`env` enjeksiyonu canlı ortamı veriyor ve mutasyona karşı korumasız.**
`SubmissionAgent.choose_action` girişimciye `env`'in kendisini geçiriyor;
RNG'yi geri sarıyor ve illegal aksiyonu yakalıyor ama `env.players[i].cash`,
`prop.owner`, `pending_trades` mutasyonunu **hiçbir şey denetlemiyor**. Tek
satırlık bir submission (`env.players[opp].cash = 0` veya kendi deed'lerine
`owner=me` yazan) smoke testte yakalanmaz — validate yalnız
"oyun bitti mi, illegal döndü mü, RNG oynadı mı"ya bakıyor.
Öneri: karar öncesi/sonrası hafif bir durum özeti (cash vektörü + ownership
vektörü + house vektörü hash'i) karşılaştırıp mutasyonda diskalifiye; veya
`ASU_FROZEN_TEACHER`'daki `_fast_copy_env` ile klon geçirmek.
İkincil notlar: (a) süre limiti yalnız focus koltukta ölçülüyor — doğru;
(b) `smoke_test` 2 seed × 4 koltuk fixed-a/b/c'ye karşı — güçlü sahayı hiç
görmüyor; (c) `directory_size` fetch sonrası `.git` DAHİL 100MB kontrolü ile
checkout sonrası `.git` HARİÇ kontrolü farklı tabanlara bakıyor (fetch'te
pack büyükse erken red — kabul edilebilir ama bilinçli olmalı).

## 7. İyi yanlar (hakkı teslim)

- Kod kalitesi yüksek: her sabitin gerekçesi yazılı, frozen dataclass'lar,
  RNG hijyeni (`preserve_global_rng`) doğru, LRU'lu zar matematiği
  (`board.py`) motorun doubles/jail zincirini birebir modelliyor.
- 2.000 oyunda **sıfır illegal aksiyon**, tam determinizm.
- NW-exact tez güçlü sahada gerçek: +11.7pp, bizim en iyi müdahalemizin
  (+2.95pp) çok ötesinde. `net_worth()`'ün 2.5×/5.0× çarpanlarını hedef
  fonksiyon yapmak bu benchmark'ın en iyi bilinen istismarı.
- Ledger'lı, resume'lu, seat-dengeli ölçüm altyapısı (parallel.py) doğru
  tasarlanmış; tuning/holdout seed ayrımı ve "sweep seçimini holdout'ta
  raporlama" disiplini örnek alınacak düzeyde.

## 8. Önceliklendirilmiş aksiyon listesi

1. ASU-sahası performansını ölç ve README iddiasını kanıtla ya da düzelt (§2)
2. Sıfır-maliyet işlemlerde rezerv kapısını kaldır (§3a — tek satır, repro'lu)
3. Rezerv ufkunu doğal-bitiş rejimine kalibre et (§4.1 — iflasların %46'sı)
4. `active_liquidation`'ı ASU sahasında aç/test et (§3b)
5. Öneri spam'ine cool-down (§4.2)
6. Submission harness'ına env-mutasyon denetimi (§6 — turnuva bütünlüğü)

## 9. Uygulanan sertleştirme (bu PR) ve A/B doğrulaması

Uygulanan düzeltmeler:
- **3a**: `_affordable` artık `cost <= 0` işlemlerde rezerv şartı aramıyor
  (bedava takas önerisi + kabulü serbest; repro artık ACCEPT veriyor).
- **3b**: `active_liquidation` default **açık**; histerezisli —
  `liquidation_trigger=0.85` altında nakit topla, unmortgage için
  `unmortgage_headroom=1.25` üstü şart (ping-pong kapalı).
- **4.1**: rezerv ufku artık "kalan beklenen uzunluk":
  `min(expected_game_length, 200 − round)` — tur-45 sonrası `min_horizon`'a
  çakılma kalktı; kantil oyun sonuna dek ~0.992'de sabit, gerçek sonda gevşiyor.
- **4.2**: takas önerisine `proposal_cooldown_rounds=10` — aynı öneri 10 tur
  içinde tekrarlanmıyor.
- **4.4**: rollout seed'leri karar bağlamıyla tuzlanıyor (aday-içi CRN korunarak
  karar-arası zar korelasyonu kırıldı).
- **4.6/4.7**: kalan tüm config alanlarına doğrulama; `trade_margin > 0` şartı.
- **4.8**: income/luxury tax `rent_quantile` dağılımına girdi.
- **6**: `submission/contract.py`'a `_env_fingerprint` tabanlı
  **EnvironmentMutationError** — `env` enjekte edilen submission'ların cash/
  ownership/ev/trade/faz mutasyonu artık karar başına yakalanıyor (birim test:
  rakibin nakdini sıfırlayan ajan yakalandı, dürüst ajan geçti).

A/B (orijinal vs sertleştirilmiş, aynı seed'ler, eşleştirilmiş):

| Saha | Orijinal | Sertleştirilmiş | Eşleştirilmiş Δ | p |
|---|---|---|---|---|
| fixed-b/d/e (n=2000) | 50.55% | 49.45% | −1.10pp [−3.00, +0.85] | 0.28 (fark yok) |
| 3× asu-value-v1 (n=120) | 15.00% | **22.50%** | **+7.50pp** [+1.67, +14.17] | **0.035** |

Teacher sahasında iflas %85 → %77.5, full_group %64 → %74. Kalan parite açığı
(≈2.5pp) rezerv mekanizmasının ötesinde bir sorun — muhtemelen erken-oyun
aşırı-yayılma; README'deki iddia ölçülen gerçeğe çekildi.
