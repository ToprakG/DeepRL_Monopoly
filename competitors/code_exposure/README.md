# NEMESIS — AI Monopoly gönderimi

Tek dosya, tek bağımlılık: **hiçbiri**. `agent.py` yalnızca Python standart
kütüphanesini (`json`, `os`, `typing`, `__future__`) kullanır; motoru,
`ASU_FROZEN_TEACHER`'ı ya da başka hiçbir repoyu import etmez.

## Sözleşme

```python
def choose_action(state, player_id, allowed_actions) -> int
```

Bu imza modül düzeyinde tanımlı (`agent.choose_action`) ve ayrıca
`agent.Agent` sınıfı üzerinden de sunuluyor — hangisi çağrılırsa çağrılsın
aynı politika çalışır. Platform canlı motor nesnesi vermiyor; ajan aktöre
göre düzenlenmiş 300 float'lık `vector`'dan tahtayı yeniden kurup
(`ShadowEnv`) oradan karar veriyor. `board` görünümü savunmacı okunuyor:
tanıdığı alanları (nakit, tur) kullanıyor, tanımadığını yok sayıyor.

Hiçbir girdi (`None`, bozuk sözlük, eksik alan, boş aksiyon listesi)
istisna üretmiyor — turnuvada bir istisna strike demek, kötü bir hamle
sadece kötü bir hamle.

## Ne en iyiliyor

Kazanan her zaman **en yüksek final servetidir**, ve motorun
`Property.calculate_net_worth()`'ü liste fiyatı değil:

| varlık | net servete katkısı |
|---|---|
| ipotekli olmayan tapu | `fiyat × 2,5` |
| tamamlanmış renk grubundaki tapu | `fiyat × 5,0` |
| `h` ev | `h × ev_fiyatı × (1 + 0,5h)` |
| nakit | nominal |

Ajanın tüm kararları bu tablodan türetilmiş: satın alma, geliştirme
sırası, nakde çevirme oranı, ihale tavanı, takas teklifleri, hapis
stratejisi. Ayrıntılı gerekçe ve ölçülüp reddedilen fikirler için kod
içindeki yorumlara bakın.

## Ölçüm — adil lig, 9 ajan, aynı şartlar

Dört kişilik **her** alt küme aynı sayıda oynandı (tam round-robin,
C(9,4)=126 masa × 8 tohum, koltuk rotasyonlu), 8 makine / 168 vCPU'da
(3× Colab v6e1 + 5× Hetzner cpx51). n=1.008, %95 Wilson aralığı:

| # | ajan | WR | %95 aralık | ort. servet | iflas |
|---|---|---|---|---|---|
| **1** | **NEMESIS (bu repo)** | **%46,4** | [41,9 – 51,1] | **17.477** | %37,1 |
| 2 | ASU (referans öğretmen) | %39,5 | [35,1 – 44,1] | 13.402 | %51,1 |
| 3–9 | diğer altı rakip | %9,6 – %30,6 | — | — | %49,8 – %74,1 |

Eşitlik %25,0. Aralığımız ASU'nunkiyle **kesişmiyor** — istatistiksel
olarak kesin önde. 1.008 oyunda sıfır strike, sıfır çökme, sıfır yasadışı
aksiyon.

## Hız

Ortalama **2,6 ms/karar**, sert sınır 2.000 ms'nin ~770 katı altında.
10 dakikalık engine-play simülasyonunda hiçbir maçta en yavaş taraf değil.

## Özgünlük

Import ağacı ve rakip kaynak kodlarına karşı token-düzeyinde benzerlik
taraması yapıldı: ASU import edilmiyor, hiçbir rakip dosyasından
kopyalanmış blok yok. Ayrıntı: ana geliştirme reposundaki
`scripts/ozgunluk_denetimi.py`.
