# Baseline optimizasyonu — 6 Kasım 2026

## Sonuç ve sınırlar

Florence-2 → DINOv3 → EfficientLoFTR → RANSAC hattı korunmuştur. Aynı dört video anında, ikişer tekrarlı karşılaştırmada ortalama işlem süresi **8.837 s → 1.197 s (7.38×)** olmuştur. Yeni deney profili hedefin bulunduğu 264 ve 267. saniyelerde geometrik doğrulanmış FOUND üretmiştir. **Her kare <1 s ve tüm hedeflerde doğru tespit sağlanmış değildir.** Özellikle 23–25. saniyelerde Florence hedefi aday olarak üretmediğinden kaçırma sürmektedir. Birkaç başarılı kare, instance kimliğinin veya genel doğruluğun kanıtı değildir.

Çalışmakta olan CPU deneyi kontrollü iptal edilmiştir: `20261106_152952_static_dinov3-vitb16-pretrain-lvd1689m_36d5e739_Deneme1`, 409 işlenmiş kare, manifest `cancelled`. Girdi ve eski sonuçlar korunmuştur.

## Deney tasarımı

- Apple M4, 16 GiB birleşik bellek; PyTorch 2.9.1 / Transformers 4.57.6. Yerel cache, ücretli inference API yok.
- Kaynak video yaklaşık 301 s, 1920×1080. İlk 44 saniye, önceki uzun çalışmanın işleyebildiği bölüm; tüm video uzunluğu değil.
- Aynı `input_video.mp4` ve `ref_01.jpg`; kaynak SHA256 ve model revision'ları JSON sonuçlarında.
- Statik, bağımsız kareler; referans bankası her probda sıfırlanır. Dinamik bellek veya tam video doğruluk değerlendirmesi değildir.
- Aynı-kare ana karşılaştırma: 16, 40, 264, 267 s × 2 tekrar; tek süreç/ortak kilit, offline. Model yükleme ve warmup ayrı.
- İşlem süreleri senkronize aşama ölçümleridir. Model yükleme, ilk referans encoding, rastgele seek/decode, kaynak hashing ve kayıt/telemetri ek yükünün tamamını içermez. `probe_wall_seconds`, `seek_decode_ms`, `load_seconds`, `warmup_seconds` ayrıca kaydedilir. Dolayısıyla aşağıdaki <1 s sonuçları uçtan uca süre garantisi değildir.
- 17 zaman noktalı ek tarama ve 20 zaman noktalı küçük-model deneyi aynı videodandır. `heldout.json` dosya adı bağımsız veri kümesi anlamına gelmez. Kalibrasyon ve değerlendirme aynı sahneden olduğundan overfitting riski vardır.
- Ham video, kare ve crop kopyalanmadı. Sonuç dosyaları yalnız metadata, kutular, skorlar, ölçümler içerir. Ground truth okunmadı; precision/recall/mAP hesaplanmadı.

## Ölçümler

| Profil | Prob | Ortalama | Maksimum | FOUND | >1 s |
|---|---:|---:|---:|---:|---:|
| CPU özgün, beam=3, cache kapalı | 8 | 8837 ms | 20053 ms | 0 | 8 |
| MPS large, cache + beam=1 + yalnız Florence fp16, eski karar eşikleri | 8 | 1130 ms | 1462 ms | 0 | 4 |
| MPS large, aynı hız ayarları + deneysel eşikler/full affine | 8 | 1197 ms | 1448 ms | 4 | 5 |
| MPS large, ek zaman noktaları | 17 | 1058 ms | 1445 ms | 2 | 10 |
| MPS base, deneysel eşikler | 40 | 429 ms | 904 ms | 4 | 0 |
| MPS base, beam=3 | 18 | 505 ms | 1110 ms | 0 | 2 |
| MPS large, ilk bölüm 23/24/25 s | 3 | 956 ms | 1011 ms | 0 | 1 |

FOUND sütunu tekrarları içerir; doğruluk oranı değildir. Base ile large satırlarının zaman noktaları tamamen aynı değildir; genel hız/başarı oranı olarak karşılaştırılmamalıdır. Ana üç satır eşleştirilmiş karşılaştırmadır.

## Darboğaz ve başarısızlığın ayrıştırılması

### Florence

CPU eşleştirilmiş deneyinde sürenin %96.10'u Florence'a gider. 16. saniyede 26 aday; yaklaşık 20 s işlem vardır. Optimized large profilinde 2–12 aday, Florence payı %82.09. Fazla kutu, otoregresif üretim token'larını ve ardından crop embedding maliyetini artırır. NMS/max-candidates sonrası kutu azaltmak, **önceden harcanmış decoder maliyetini geri almaz**. Beam=1 ve KV cache, bu nedenle öncelikli iyileştirmedir. İncelenen üretimlerde 256-token sınırına çarpılmadı; başka yoğun sahnelerde bu garanti değildir, bayrak kaydedilir.

24. saniyede hedef görüntüde olsa da optimized large'ın üç kutusu sağ taraftaki yapılara gider. Hedefin adayı yoktur. Küçük model, iki örtüşen kare bölgesi ve 90° döndürme denemeleri de bu örneklerde sorunu çözmedi. Bu yöntemler varsayılan hatta eklenmedi. Aday sayısını körlemesine kısmak tespit kaybını ağırlaştırabilir.

### DINO ve geometri

264–268 s çevresinde large model hedef adayını üretebilir. Doğru adaya ait gözlenen cosine yaklaşık .36–.42 iken eski `.70` eşiği geometrik doğrulamaya ulaşmasını engeller. Yalnız MPS'ye geçiş tespit sorununu çözmez. İlk 44 saniyelik eski koşuda tüm kararların `no_visual_match` olması, geometri performansı hakkında kanıt değildir: geometri hiç çalışmamıştır.

Deneysel profil `.30` cosine, confidence ≥.40 LoFTR eşleşmeleri, full affine RANSAC, ≥12 inlier, ≥.20 oran, ≥.03 coverage, ≤3 px medyan hata kullanır. Kısmi affine'e göre anisotropik görünüm değişimine izin verir; yansıma, aşırı ölçek/anisotropi, yetersiz uzamsal destek reddedilir. Parametreler bu videodan seçilmiştir; yanlış pozitif olasılığının arttığı varsayılmalı ve bağımsız veride ölçülmelidir. Eşiği tek başına düşürüp FOUND ilan edilmez.

Düşük benzerlikte geometrinin denenebilmesi için `geometry_min_cosine` ayrı parametredir. Bu yalnız tanılama amacıyla detection eşiğinden düşük olabilir; geometrinin geçmesi detection eşiğini bypass etmez. Dinamik bellek kabul eşikleri gevşetilmemiştir; bu deneylerde statik mod kullanılır.

## Uygulanan değişiklikler

1. Florence legacy KV-cache uyumluluğu sabit Transformers sürümü ve model sınıfı için instance-local adaptörle; upstream dosyalar değiştirilmeden. Beam/cache/Florence precision konfigüre edilebilir.
2. MPS'de yalnız Florence fp16. DINOv3 ve EfficientLoFTR fp32. Tüm hattı fp16 yapmak DINO'da sonlu olmayan embedding ürettiği için benimsenmedi.
3. Orijinal 768×768 Florence processor korunur. 512/384 deneyleri kutu koordinat tutarlılığı ve aday kaybı nedeniyle reddedildi.
4. Geometri modeli ve eşleşme confidence filtresi ayrı konfigürasyon alanları; eski partial-affine varsayılanı korunur.
5. Tüm aday kutuları/crop sınırları, skorlar, generation tanılaması, geometri ret nedenleri ve maliyetleri kaydedilir; reddedilen adaylar arayüzde görüntülenebilir.
6. Arayüzde açık cihaz ve kare sınırı; ilerlemede aşama süreleri ve aday sayısı. `auto` hâlâ CUDA yoksa CPU seçer; Mac için `mps` açık seçilmelidir.
7. RAM, MPS tensor/driver, sistem kullanılabilir RAM ve swap telemetrisi. Desteklenmeyen metrikler null; sıfır GPU maliyeti gibi gösterilmez.
8. Aynı gerçek pipeline'ı kullanan bütçeli, yeniden çalıştırılabilir benchmark scripti ve regresyon testleri.

## Bellek, ısınma ve maliyet

Eşleştirilmiş CPU deneyinde örneklenen süreç RSS maksimumu yaklaşık 5.76 GiB; optimized large MPS'de RSS 1.28 GiB, Metal driver tahsisi 4.99 GiB. Bunlar farklı ölçümlerdir: Apple birleşik belleğinde RSS ve Metal tahsisleri **toplanarak toplam RAM tüketimi hesaplanamaz**. MPS base deneyinde RSS yaklaşık 1.22 GiB, driver 3.92 GiB. Değerler kare sınırlarında örneklenmiştir; anlık gerçek tepe daha yüksek olabilir. Sistem swap kullanımı süreç dışı uygulamaları ve önceki yükü de içerir, yalnız bu projeye mal edilemez.

Donanım hedefli prototipleme için kullanılabilir; mevcut büyük-model profiliyle 1 s en-kötü-süre şartı doğrulanmadı. GPU daha kısa hesaplama sağladı, fakat daha soğuk çalışacağı gösterilmedi. Güç/ısı sensörü ölçümü yapılmadı; watt, joule, elektrik faturası ve termal kazanç uydurulmadı. Küçük model hız hedefini seçili prob hesaplarında karşıladı, fakat aday recall kaybı yüzünden ana öneri yapılmadı.

Deneyler yerel modellere dayanır; ücretli model API çağrısı yoktur. Bu tur indirilen ek base ağırlığı yaklaşık 425 MiB, kullanıcı cache'indedir. Codex kredi tüketimi bu uygulamanın profiler'ından okunamaz; kesin yüzde veya parasal tasarruf çıkarılamaz. Uzun video yeniden işlenmedi; sınırlı kareler ve yeniden kullanılan cache ile çalışma bütçesi sınırlandı.

## Kullanım ve sonraki kontrollü deney

`configs/experiments/mps-calibrated.yaml` ölçülen large profildir; yalnız deneysel kullanım için. `mps-fast.yaml` eski karar eşiklerini koruyarak hız etkisini ayırır. `cpu-original.yaml` eski davranışı tekrar üretir. Base profilleri karşılaştırma içindir, başarı önerisi değildir. Varsayılan karar eşikleri sessizce değiştirilmez.

```bash
cd ~/Desktop/uav-reference-matching
source .venv/bin/activate
uav-refmatch ui --config configs/experiments/mps-calibrated.yaml
```

Yeni uzun koşudan önce 24 s ve 262–269 s gibi pozitif anlar ile benzer araç içeren negatif anları ayrı bir değerlendirme setinde işaretleyin. Önce **proposal recall@IoU**, ardından referansın adaylar içindeki sırası, geometri geçişi ve yanlış pozitifleri ölçün. Etiketler inference'a verilmemeli. Gerçek yarışma metriği uçtan uca ise seek/decoder, kayıt, warmup ve cihaz senkronizasyon kapsamını sabitleyin; p50/p95/p99 ve maksimumu ardışık video üzerinde ölçün.

Öncelikli kalan sorun 24 s aday recall'ıdır. Florence görev/decoding seçimi ve referans görüş farkını azaltan temsil/crop deneyleri, ayrı küçük ablation setinde sınanmalıdır. İşe yaramadığı ölçülen tiling/rotasyon veya eşikleri sürekli gevşetme ana hatta taşınmamalıdır. Daha geniş değerlendirme olmadan tüm video üzerinde tekrar uzun koşu veya yeni büyük model indirme önerilmez.

## Kanıtlar ve doğrulama

`docs/experiments/2026-11-06/` altında paired, heldout, base, base-beam3 ve first44 JSON'ları; çözülmüş konfigürasyonlar, revision'lar, kare bazlı sonuçlar ve kaynak hash'leri bulunur. İlk ek tarama geliştirme sırasında kaydedildiğinden sonraki harness'in tüm source-hash alanlarını içermeyebilir; eşleştirilmiş ana deney kod hash'lerini içerir. JSON içindeki geçici çalışma yolu ve dirty=true, o sıradaki geliştirme ağacını dürüstçe gösterir.

74 test geçti (gerçek model testleri dahil); cache adaptörü izolasyonu, affine/reflection, confidence filtresi, eşiklerin bypass edilmemesi, aday kayıtları ve eski nedensel bellek davranışı kontrol edildi. Sonrasında yalnız deney verileri ve bu dokümantasyon eklendi.

Resmî kaynaklar: [Florence-2 model kartı](https://huggingface.co/microsoft/Florence-2-large-ft), [DINOv3](https://github.com/facebookresearch/dinov3), [EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR), [PyTorch MPS](https://docs.pytorch.org/docs/stable/mps.html), [OpenCV affine estimation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). Bunlar mimari/API kaynaklarıdır; yukarıdaki süreler bu bilgisayardaki yerel deneylerdir.
