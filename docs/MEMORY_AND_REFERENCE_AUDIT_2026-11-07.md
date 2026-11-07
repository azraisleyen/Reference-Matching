# Dinamik bellek ve farklı referans denetimi

## Mevcut deneylerin karşılaştırması

| Deney / gerçek mod | Referans | Kare | FOUND | Benzerlik reddi | Geometri reddi | Ortalama süre |
|---|---|---:|---:|---:|---:|---:|
| 071705 / static | ref_01, biçerdöver | 2258 | 61 | 1893 | 302 | 955.98 ms |
| 084957 / dynamic | ref_01, biçerdöver | 2258 | 61 | 1893 | 302 | 955.37 ms |
| 095511 / **static** | ref_03, futbol kalesi | 707 | 0 | 557 | 150 | 1095.11 ms |

İlk iki deney aynı video/referans hash'lerini ve mod dışındaki aynı ayarları kullanıyor; kare bazında durum ve bbox sonuçları tamamen aynı. İki karede aday yok. Son deneyin adında Dynamic yazması gerçek modu değiştirmiyor: manifest static, durum cancelled, son PTS 94.161 s. Diğerleri videonun 300.934 s noktasına kadar tamamlanmış. Eşit ilk 707 karede ref_01 için 35, ref_03 için 0 FOUND var; **hedefler farklı olduğundan bu sayılar doğrudan referans fotoğrafı kalite/recall kıyası değildir**. Etiketli ground truth bulunmadığından doğru/yanlış tespit yüzdesi verilmez.

## Dinamik bellek neden hiçbir fark yaratmadı?

Bellek algoritması çağrılıyor, ancak hiç yeni kayıt kabul etmiyor: ACCEPTED=0, maksimum aktif referans=1 (yalnız R0). Başarılı tespitlerde en yüksek cosine **0.6202**; bellek kabul eşiği **0.80**, anchor eşiği **0.75**. İki eşik de gözlenen başarılı tespitlerin üzerindedir. Dolayısıyla belleğin farklı görünüşlerden yararlanması mümkün olmamış. Sadece modu dynamic yapmak eşikleri kalibre etmez.

Tüm 2258 olay eski logda REJECTED_LOW_CONFIDENCE olarak görünüyordu; bunların çoğu zaten NOT_FOUND. Yeni telemetri `failed_gate` ile tespit yokluğu, cosine/anchor eşiği, geometri ve sınırda kesilme nedenlerini ayırır. Özet artık kabul sayısını, maksimum aktif referansı, başarılı tespitlerdeki en yüksek cosine'ı ve eşikleri gösterir. Dynamic olup kabul sıfırsa açık açıklama üretir.

Eski çalışmalarda görüntü bankası iki modda da 19,427,265 byte. Bellek güncellemesinin ortalama ek süresi static 0.0005 ms, dynamic 0.0048 ms; tüm kare süresindeki 0.6 ms fark pratikte ölçüm değişkenliği düzeyinde. **Bu koşular, çalışan/genişlemiş dinamik belleğin maliyetini ölçmüyor.**

## Kontrollü, nedensel bellek deneyi

Florence tekrar çalıştırılmadan aynı kayıtlı model kutuları kullanıldı. DINOv3, EfficientLoFTR, RANSAC ve bellek güncellemeleri gerçek modellerle yeniden hesaplandı. Seçili aralıklar: 24–25, 40–41, 68.4–75, 83–85 ve 262–270 s; toplam **141 kare**. Orijinal kare indeksleri korunur; atlanan aralıklar ardışık onay gibi sayılmaz. Bellek kronolojik ilerler, gelecekteki görüntüler geçmiş kararlarda kullanılmaz. Video/referans SHA256 ve her çözülen karenin PTS'si doğrulanır.

İlk araç sürümü atlanan video aralıklarını da sırayla decode ettiği için bütçeye ulaştı; kısmi sonuç son karşılaştırma olarak kullanılmadı. Araç boşluklara seek edecek şekilde düzeltildi ve iki mod aynı 141 karede tamamlandı.

Deneysel bellek kabulü: cosine/anchor ≥.50, inlier ≥20, oran ≥.25; kalite, ardışık onay, cooldown ve duplicate kontrolleri korunur. Ek korumalar: yalnız **R0'a geometrik doğrulanmış** tespit bellek adayı olabilir; sınırda kesilmiş kutular tespit olarak kalabilir fakat belleğe alınmaz. Kapasite dört ek kayıtla sınırlıdır. Bunlar ref_01 üzerinde denenmiş eşiklerdir; evrensel ayarlar değildir.

| Ölçülen ilk kalibrasyon | Static | Dynamic |
|---|---:|---:|
| FOUND | 60 | 88 |
| Yeni kabul | 0 | 1 |
| Maksimum aktif referans | 1 | 2 |
| LoFTR çağrısı | 264 | 293 |
| Downstream ortalama süre | 317.69 ms | 259.13 ms |
| Banka görüntü byte tepe | 19,427,265 | 19,774,065 |

Bir kayıt 69.603 s'de kabul edildi; sonraki karelerde kullanıldı. 31 yeni FOUND ve 3 kayıp, net +28 elde edildi. Yeni tespit örnekleri 72.806 s, 267.200 s ve hedef kısmen görünürken 269.870 s'de görsel olarak kontrol edildi; kutular biçerdöveri kapsıyor. Bu kontrol tüm tespitlerin instance doğruluğu için tam etiketleme değildir.

Ek görüntü bankası yalnız **346,800 byte (~0.35 MB)**, embedding farkı 3072 byte. Çağrı sayısı artmasına rağmen ölçülen downstream ortalaması artmadı; küçük bellek crop'larıyla eşleşme, büyük R0 ile eşleşmeden daha az iş içerebilir. Bu yorum tek tekrarlı kısa ölçümden gelir. Florence, tam video decode ve tüm uçtan uca maliyet dahil değildir; bu değerler 955 ms tam-video ortalamasıyla doğrudan karşılaştırılmaz. RSS/Metal tahsisleri birleşik bellekte toplanmaz. Tam video, ek negatifler ve drift denemesi olmadan genel maliyet/başarı garantisi verilmez.

### Bulunan bütçe paylaşımı sorunu

262.295, 262.429 ve 263.363 s'de static R0 +45° ile geçiyordu. Dynamic modda önce yeni referans ve R0 normal deneniyor; kalan üçüncü çağrı yeni referansın -45° denemesine harcanıyor, R0 +45° hiç denenmiyordu. Bu bir bellek faydasından bağımsız **doğrulama bütçesi paylaşımı** sorunudur.

Yeni deneysel `mps-dynamic-adaptive.yaml` uygun R0 çiftine yer ayırır, R0'ın iki yönelim tekrarına öncelik verir ve toplam LoFTR sınırını **3'ten 4'e** çıkarır. Bu değişiklik bedelsiz değildir: kare başına azami çağrı sınırı bir artırılmıştır. Son kontrol aşağıdadır.

## Ref_03 neden daha zor?

`ref_03.jpg` aynı biçerdöverin farklı açısı değil, yakın çekim bir futbol kalesidir. İnce beyaz çizgiler, geniş çimen arka planı ve farklı bakış açısı nedeniyle başka bir eşleşme problemidir. LoFTR yerel desteğinin nesnenin kendisinden mi, benzer zeminden mi geldiği önemlidir; yalnız sınıf benzerliği belirli fiziksel nesneyi doğrulamaz.

Otomatik açıklama `A soccer goal post on a grassy field` idi. Eski metin kısaltması konum fiilleriyle çalıştığından `on a grassy field` bölümünü kesmiyordu. Florence hem kale hem bütün saha kutuları üretebiliyordu. 24.024 s'de yalnız büyük saha kutusu; 23.490 ve 84.017 s'de doğru küçük kale kutusu da görülüyor. Dolayısıyla **hem aday hatası hem geometrik reddetme** var.

150 geometri-reddedilmiş karede toplam 450 doğrulama denemesi: 277 insufficient_matches, 102 invalid_scale_or_transform, 71 insufficient_inliers. Bunlar kare değil deneme sayılarıdır. Doğru aday bulunduğunda bile küçük kale crop'unda birkaç güvenilir eşleşme kalabiliyor. Örneğin yeni dar kutu 24.024 s'de yaklaşık 77×94 px; 84.017 s'de 136×120 px. Referans ise çok daha büyük ve eğik bakışlıdır.

### Küçük, ayrı ablation

Dört zaman noktasında (16.016, 23.490, 24.024, 84.017):

- Eski açıklama/ölçek: 0 FOUND.
- `caption_subject_v2`: yer belirten “on/in/near/beside/against a/an/the …” kısmını kaldırır, hedef sınıfı sabit değildir. Doğru dar kale kutularını iyileştirir; tek başına 0 FOUND.
- Aynı odaklama + `loftr_resize_to_max: true`: küçük görüntüyü LoFTR'ın sınırlı 640 uzun kenarına büyütür, eşleşmeleri özgün piksele geri çevirir. 84.017 s'de **53 eşleşme, 23 inlier, .434 oran, 1.094 px medyan hata**, geometri geçti. Diğer iki pozitif görünüş geçmedi; negatif örnek FOUND olmadı.

Upsampling yeni görüntü ayrıntısı yaratmaz; mevcut çizgilerin ağ girişindeki ölçeğini değiştirir. Daha fazla compute getirebilir ve tek olumlu örnek yeterli genelleme kanıtı değildir. Bu nedenle `mps-small-object-probe.yaml` **deneysel ayrı profil**, mevcut biçerdöver profilinin varsayılanı değildir. Geometri eşiği düşürülmedi. `caption_subject_v2` de basit dil kuralıdır; başka referanslarda yanlış kısaltma ihtimali vardır, üretilen metin manifest'te incelenebilir.

## Sınırlar ve sonraki gerekli doğrulama

- Erken 24 s civarı biçerdöver görünüşü, ilk bellek kabulünden öncedir; sonraki belleğin geçmişteki hatayı düzeltmesi nedensel olarak mümkün değildir.
- Bazı karelerde yalnız birkaç inlier veya bozuk dönüşüm vardır; min-inlier/coverage eşiklerini düşürmek benzer çimen/başka kale gibi yanlış eşleşmeleri de geçirebilir. Bu yapılmadı.
- Aynı sınıftan diğer nesnelerle zor negatifler ve ayrı videolar gerekir. Başka bir referansta başarının değişmesi beklenebilir; tek ortak eşik tüm nesne ölçeklerini/dokularını garanti etmez.
- Mevcut testler görüntü kalitesi ve görünüş çeşitliliği etkisini gösterir; bütün kullanıcı referansları veya tüm gelecek videolar için sonuç çıkarılmaz.

Kanıtlar `docs/experiments/2026-11-07-memory/` altında; yeni araç `scripts/benchmark_memory_replay.py`. Kaynak görüntüler/kareler repoya kopyalanmadı. Eski deney çıktıları değiştirilmedi. Yeni rapor alanları yeni koşularda veya açık `uav-refmatch report` çağrısında üretilir.

## Son kontrol — düzeltilmiş bütçe politikası

Aktif kullanıcı deneyi açık onayla kontrollü iptal edildi: `20261107_114134_static_dinov3-vitb16-pretrain-lvd1689m_55521242`, 1140 kare, manifest cancelled. Ardından GPU kilidi alınarak son test tamamlandı.

Aynı 141 karede **static 60 → düzeltilmiş dynamic 91 FOUND**. 31 kazanım var, static tespitlerden kayıp yok; önceki üç regresyon da giderildi. Bir yeni referans kabul edildi, azami aktif referans iki. Bu sonuç tam video accuracy/recall değil, kayıtlı adaylar üzerinde nedensel pencere deneyidir. Tam video için 91 sonucu öngörülmemeli.

Son dynamic downstream ortalaması 364.96 ms, toplam LoFTR çağrısı 315, görüntü bankası tepe 19,774,065 byte. Static karşılıkları 317.69 ms / 264 çağrı / 19,427,265 byte. Florence tekrar çalıştırılmadığından bunlar uçtan uca ölçüm değildir; tek tekrarlı sıralı ölçümlerde termal/cache değişkenliği bulunur.

## Devam oturumu: durum denetimi ve tamamlayıcı kontrol

Başlangıçta masaüstündeki main hâlâ `d1fdc19` üzerindeydi. Son 141-kare deneyi gerçekten tamamlanmıştı; geliştirmeler `codex/audit-dynamic-memory` çalışma ağacında commit/merge edilmeden kalmıştı. Veri/kod kaybı yoktu. Bu sonuç yeniden hesaplanmadı. Eksik test kapanışı ve entegrasyon tamamlandı.

Ek nedensel kontrol: 68.4–75 s ile bellek oluşturuldu; ardından 100–101, 150–151, 210–211 s aralıkları ve 262–264 s sonraki hedef görünüşü kullanıldı. Aynı **89 karede static 45 / dynamic 60 FOUND**, 15 kazanım, önceki tespitlerden kayıp yok. Bellek kabulünden sonraki üç negatif aralıkta toplam **24 kare, iki modda da 0 FOUND**. Downstream ortalamaları **242.37 / 250.07 ms**; ek maliyet yaklaşık **7.70 ms (%3.2)**. Önceki 141-kare karşılaştırmasında ek maliyet **47.27 ms (%14.9)** idi. Kare karışımı, sıcaklık ve cache farkları nedeniyle tek sayı genelleştirilmemeli. Bu, ayrı video veya benzer sınıftan zor negatiflerle drift kanıtı değildir.

Bütçe karşılaştırması notu: 141-kare static kaydında geometriye uygun her karede yalnız bir aday vardı; iki aday üretilen beş karede ikisi de .30 cosine altında kaldı. Dolayısıyla bu seçili static örneklerde dört çağrı izni ek bir R0 denemesi yaratmaz (normal + iki dönüş toplam üç). Dynamic yeni referansla, ek çağrı ise R0'ın kaybolan dönüş hakkını koruyarak fayda sağladı. Daha fazla uygun adaylı sahnelerde ayrıca adil bütçe ablation'ı gerekir.

Kale için yeni zaman noktaları 23.890, 25.025, 84.418 ve 84.685 s'dir. Odaklama + upsampling profili son iki örnekte FOUND, ilk ikide NOT_FOUND üretti. Önceden tamamlanmış ref_03 koşusunda FOUND hiç yoktu. Yeni dört örnek için süreler yaklaşık 1.32–1.76 s; farklı örneklerin değişik bakış açısı/ölçeği genellenmiş doğruluk veya sabit hız iddiasına dönüştürülmedi. İnce/az dokulu erken görünümler hâlâ açık sorundur.

Önerilen kullanım: `mps-dynamic-adaptive.yaml` ref_01 üzerinde ölçülmüş, sınırları belirtilen bellek profili; `mps-small-object-probe.yaml` kale gibi küçük nesnelerde ayrıca değerlendirilmesi gereken statik deney profilidir. İkincisi bütün nesneler için varsayılan değildir. İkisini birleştiren yeni bir profil test edilmedi. Varsayılanlar ve eski deney profilleri korunmuştur.

Son doğrulama: **90 birim/regresyon testi ve ayrıca 3 gerçek model testi geçti**. Ruff ve diff kontrolleri temiz. Gerçek Florence testinde upstream kaynaklı tek escape-sequence deprecation uyarısı görüldü.
