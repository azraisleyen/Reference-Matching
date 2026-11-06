# UAV referans eşleştirme: sonuç, performans ve iyileştirme araştırması

Tarih: 6 Kasım 2026. Kapsam: Masaüstündeki beş gerçek UAV run'ının bütün tahmin, zaman, kaynak ve bellek kayıtları; mevcut adaptörler ve karar kodu; iki sentetik kontrol raporu; resmî teknik kaynaklar. Bu incelemede model indirilmedi, inference çalıştırılmadı, video/kırpım kopyalanmadı, eşikler veya çalışan pipeline değiştirilmedi. Araştırma çıktıları mevcut yazılabilir Documents/ChatGPT çalışma alanına kaydedildi.

**Karar: Mevcut ayarla tam video tekrar çalıştırılmamalı. Öncelik, aday kutularını görünür kılmak ve benzerlik kapısındaki başarısızlığı teşhis etmek; ardından Florence üretimini ve MPS yolunu küçük deneylerle iyileştirmek. Daha büyük DINO modeli, daha fazla referans kapasitesi veya tüm videoda daha düşük eşik şu an gerekçelendirilmiş çözümler değil.**

## 1. Ölçülen sonuçlar

Ana run: `20261106_114512_dynamic_dinov3-vitb16-pretrain-lvd1689m_a946925a_Deneme`.

| Ölçüm | Sonuç |
|---|---:|
| Run durumu | cancelled |
| Gerçek duvar süresi | 2721,63 saniye = 45 dakika 21,63 saniye |
| İşlenen kare | 331 |
| Son kaynak zaman damgası | 44,0106 saniye |
| Kaynak video | 301,034 saniye; 1920×1080; yaklaşık 29,97 FPS |
| FOUND / NOT_FOUND / ERROR | 0 / 331 / 0 |
| Ret nedeni | 331 × `no_visual_match` |
| Toplam aday / ortalama aday | 3064 / 9,257 |
| Kare başına aday aralığı | 1–28 |
| En iyi aday skorlarının ortalaması / medyanı | 0,1590 / 0,1646 |
| En iyi aday skorlarının p95 / maksimumu | 0,2181 / 0,2498 |
| DINO kabul eşiği | 0,70 |
| Inference sırasında geometrik çift | 0 |
| Yeni dinamik referans | 0 |
| Aktif referans / pending | Daima 1 / 0 |
| Gerçek cihaz | CPU, fp32 |

331 kare bütün videoyu temsil etmez. Yaklaşık ilk 44 saniye işlendi; kalan bölüm hakkında sonuç yok. Ekran görüntüsünde hedefe benzeyen sarı biçerdöver görünür ve referans görüntüsü de biçerdöverdir; fakat tüm kareler için bağımsız hedef-var/yok ve kutu değerlendirmesi yapılmadı. Bu yüzden `0 FOUND`, “ölçülmüş recall %0” diye raporlanamaz. Sorun belirgindir, akademik doğruluk metriği henüz yoktur.

**Eşik-geçiş sonucu kesindir:** Bütün adaylar 0,70 altında. EfficientLoFTR/RANSAC henüz adaylar üzerinde sınanmamış. Servis başlangıcında warmup için görüntünün kendisiyle eşleştirme yapılması, bu sıfır çift sonucuyla çelişmez.

Masaüstündeki diğer dört run da incelendi:

| Run kısa adı | Mod | Kare | Duvar süresi | Sonuç |
|---|---|---:|---:|---|
| 110018 validation-real | dynamic | 3 | 26,31 sn | 3 NOT_FOUND |
| 110128 validation-real | static | 3 | 19,96 sn | 3 NOT_FOUND |
| 110507 validation-final | dynamic | 3 | 21,06 sn | 3 NOT_FOUND |
| 110531 validation-final | static | 3 | 20,10 sn | 3 NOT_FOUND |

İlk iki eski kayıtta eşik altı cosine alanı boş; bunlardan skor dağılımı türetilemez. Son iki kısa run'ın ortalama kare süresi 3,54 ve 3,51 saniye. Bunlar yalnız ilk üç kareyi kapsadığı için uzun video tahmini için yetersizdi.

`positive_control_validation.json` ve `memory_view_validation.json` içindeki sentetik kontrollerde static/dynamic başına üç FOUND var; ortalama süreler yaklaşık 4,97–5,34 sn/kare. Döndürülmüş sentetik kontrolde PENDING → ACCEPTED → sonraki karede kullanım doğrulanmış. Bu, yazılımın olumlu karar üretebildiğini ve nedenselliği gösterir; gerçek 3B bakış açısı değişimi, hedefi kaçırma ve yanlış nesne kabulü hakkında başarı kanıtı değildir. Önceki 66 geçen test de doğruluk benchmark'ı değildir.

## 2. Süre: dar boğaz Florence

| Aşama | Ortalama ms/kare | Toplam kare süresindeki pay |
|---|---:|---:|
| Decode | 38,44 | %0,470 |
| Florence proposal ve crop hazırlama | 7829,88 | **%95,760** |
| DINOv3 | 307,87 | %3,765 |
| Retrieval | 0,116 | %0,0014 |
| EfficientLoFTR / RANSAC | 0 / 0 | %0 |
| Bellek güncelleme | 0,0051 | İhmal edilebilir |
| Toplam | **8176,58** | %100 |

Toplamın medyanı 6,653 sn, p95'i 15,568 sn, maksimumu 114,530 sn. 118. karede Florence tek başına 113,635 sn sürmüş; bu tek aykırı değeri saklamadım. Sıcaklık, swap ve işlemci frekansı kaydı olmadığı için sebebi belirlenemiyor.

Toplam kare süreleri 2706,45 sn; duvar süresi 2721,63 sn. Model yükleme 8,54 sn, warmup 5,33 sn ve ilk referans kodlama 0,076 sn. Model yükleme ana maliyet değil. Kare toplamı resource örnekleme, manifest yazma ve ilerleme bildiriminin hepsini kapsamaz; duvar süresi de ayrıca tutulmalıdır.

| İşlenmiş kare bloğu | Ortalama sn/kare | Ortalama aday |
|---|---:|---:|
| 0–49 | 3,84 | 2,36 |
| 50–99 | 6,03 | 5,84 |
| 100–149 | 14,23 | 16,16 |
| 150–199 | 5,23 | 5,34 |
| 200–249 | 7,99 | 9,74 |
| 250–299 | 10,86 | 14,28 |
| 300–330 | 9,58 | 12,19 |

Aday sayısı ile proposal süresi Pearson korelasyonu bütün kayıtlarda 0,612; tek >30 sn proposal aykırısı dışarıda bırakıldığında 0,979. İkinci sayı hassasiyet incelemesidir, ana sonuçtan aykırı değer çıkarılmamıştır. Korelasyon neden kanıtı değildir; fakat kutu sayısıyla büyüyen üretim yükü çok güçlü bir adaydır. Süreler sürekli artmıyor: 100–149 sonrasında tekrar düşüyor. “Kesin thermal throttling” açıklaması kayıtlarla desteklenmiyor.

Kod bulguları: Florence `num_beams=3`, `use_cache=False`, `attn_implementation="eager"`, `max_new_tokens=1024` ile çalışıyor. KV cache kapalıyken autoregresif üretimde önceki token hesaplarının yeniden yapılması maliyet getirir. [Transformers 4.57.1 KV cache belgesi](https://huggingface.co/docs/transformers/v4.57.1/en/kv_cache) bunu açıklıyor. Proje belgeleri cache'in legacy uyumluluk nedeniyle kapatıldığını söylüyor; kontrolsüzce açmak çalışır bir düzeltme sayılmaz.

`max_proposals=30`, üretim bittikten sonra uygulanan sınırdır; Florence'ın token üretimini erken durdurmaz. 1024 token üst sınırını azaltmak ise her karede hızlanma sağlamaz: EOS zaten erken geliyorsa fark olmayabilir, sınır dolarsa doğru kutular kesilebilir. Üretilen token sayısı ve EOS/truncation durumu önce ölçülmeli.

Mevcut ortalama ile yaklaşık 2258 kare 5,13 saat hesaplama gerektirir; 12 sn/kare varsayımıyla 7,53 saat. İkisi de tam video ölçümü değildir. 7,5 FPS gerçek zaman hedefi kare başına 133 ms bütçe demektir; ortalama mevcut hat yaklaşık 61 kat daha yavaştır.

Yatırım önceliği için Amdahl hesabı: DINO süresi teorik olarak sıfırlansa dahi toplam hızlanma yaklaşık 1,04×. Florence 2× hızlanırsa yaklaşık 1,92× toplam hızlanma beklenir (diğer aşamalar sabit varsayımı). Dolayısıyla DINO'yu ConvNeXt Tiny'ye geçirmek ilk hız optimizasyonu olmamalı. Geometriye aday göndermeye başlayınca mevcut süre hesabına yeni maliyet eklenecektir.

## 3. Bellek, depolama ve maliyet

Worker RSS ortalaması **4,625 GiB**, gözlenen en yüksek örnek **5,290 GiB**, en düşük **2,391 GiB**. Bunlar kare sonu süreç RSS örnekleridir; bütün sistem, arayüz, sıkıştırılmış bellek, swap veya model çağrısı içindeki gerçek tepe değildir. Bu kayıtlardan “16 GB kesin yetmiyor” ya da “bellek sızıntısı var” sonucu çıkmaz. Sürekli artan RSS de görünmüyor.

Referans görseli RAM'de 19.427.265 bayt = 18,53 MiB, embedding 3072 bayt = 3 KiB. Yeni referans yok; bellek bankası bu deneyde yükün nedeni değil. Dinamik belleğin faydası ve büyüme maliyeti gerçek UAV deneyinde henüz ölçülemedi.

GPU alanları CPU run'ında sıfır. Ayrıca mevcut profiler MPS belleğini hiç ölçmüyor: MPS run'ında aynı alanların sıfır kalması GPU'nun bellek kullanmadığı anlamına gelmez. `torch.mps.current_allocated_memory()` ve `driver_allocated_memory()` eklenmeli; kare arası örneklerin gerçek tepe olmadığı yazılmalı. Apple birleşik belleğinde RSS ile GPU sayaçlarını doğrudan toplayıp toplam RAM diye sunmamak gerekir. [Kurulu PyTorch 2.9 sürümünün MPS belgesi](https://docs.pytorch.org/docs/2.9/mps.html).

Ana run dizini inceleme anında yaklaşık 150,85 MB; bunun yaklaşık 150,27 MB'ı sonradan üretilen anotasyonlu video, yaklaşık 0,58 MB'ı diğer çıktılar. Render, inference sonrası ayrı maliyettir; manifest render süresi/enerjisi tutmadığından toplam render maliyeti hesaplanamaz. Model cache yaklaşık 1,4 GiB Florence + 342 MiB DINO + 192 MiB EfficientLoFTR; kaynak kodları/sanal ortam bunların dışındadır. Gradio yükleme, kaynak videonun geçici kopyasını oluşturabilir; geliştirmede yerel dosya yolu alanı bu ek yüklemeyi önler.

Mevcut pipeline yerel PyTorch çalıştırıyor; kare başına ücretli model API çağrısı yok. Yerel inference'ın hesaplama maliyeti elektrik, cihaz zamanı ve depolamadır. Sohbetteki araştırma/geliştirme kullanım bütçesi ayrı bir kalemdir; bu run dosyaları hesap kredisi veya fatura bilgisi içermez, kalan kredi ve parasal tüketim buradan hesaplanamaz.

Enerji ölçümü yok. Doğru formül: `enerji_kWh = ortalama_priz_gücü_W × saat / 1000`; elektrik bedeli bunun kendi tarifenle çarpımıdır. CPU yüzdesinden watt, watt'tan tahminî bir TL bedeli üretmek doğru olmaz. Bulut için de `saatlik ücret × ölçülmüş çalışma süresi + depolama/aktarım` gerekir; ölçülmüş GPU süresi olmadan maliyet karşılaştırması yapılmamalı. Şu aşamada ücretli bulut veya yeni donanım satın almak gerekmiyor.

## 4. Hedef neden bulunamıyor?

**Kanıtlanan yakın neden:** DINO kapısı bütün adayları eliyor. **Henüz belirlenmeyen kök neden:** doğru hedef kutusu adaylar arasında mı; doğru crop düşük skor mu alıyor; yoksa hedef önerilmiyor mu?

Mevcut `predictions.jsonl`, aday indeksini ve cosine skorunu tutuyor; başarısız adayların bbox'larını tutmuyor. `bbox_xyxy` yalnız FOUND olduğunda doluyor. Bu nedenle 3064 adayın hangisinin biçerdöveri kapsadığını geçmiş kayıtlardan geri çıkarmak mümkün değil. Bu, ilk giderilmesi gereken gözlemlenebilirlik eksikliği. Kaydetmek için görüntü gerekmez: proposal ID, ham/temizlenmiş bbox, crop bounds, eleme nedeni ve sıralama metadata'sı yeterli. RAM'de çizilip gösterilen aday kutuları da teşhisi kolaylaştırır; ham frame/crop diske yazma kısıtı korunur.

Görüntü incelemesinden hipotezler: Referans yüksek çözünürlüklü eğik bakış, UAV görüntüsü tepeden ve çok daha küçük nesne görünümü içeriyor. Referansta yan yüzeyler; videoda üst yüzeyler baskın. DINO adaptörü resmi processor ile görüntüyü 224×224'e getirip yalnız CLS kullanıyor. Normalizasyon, RGB dönüşümü ve CLS indeksinde okunan kodda bariz bir hata görülmedi; bu, bütün model davranışını doğrulamaz. Florence processor girdisi 768×768; kaynakta mevcut detay azalıyor ve en-boy oranı değişiyor. Yanlış/geniş crop, arka plan baskınlığı, yön ve görünüş değişimi olası nedenler; hiçbiri mevcut loglarla tek başına kanıtlanmadı.

0,70 eşiği ilk prompt'taki kalibre edilmemiş başlangıç değeridir. Skorun yüzde olasılık anlamı yoktur. **0,70'i körlemesine 0,20 yapmak çözüm değildir:** hedef olmayan benzer bölgeleri de geçirir, geometri süresini artırır ve yanlış bellek ekleme riskini büyütür.

DINO'nun resmi deposunda patch düzeyinde dense/sparse eşleştirme örnekleri bulunuyor. Bu, CLS dışında yerel görünüm karşılaştırması için gerekçeli bir araştırma yolu; daha büyük ağırlık indirmeden mevcut ViT-B ile denenebilir. [DINOv3 resmi depo ve örnekler](https://github.com/facebookresearch/dinov3). Ancak baseline CLS davranışını sessizce değiştirmemeli: ayrı encoder temsili deneyi olarak kaydetmeli.

Geometri için ikinci bir olası engel var: `estimateAffinePartial2D`, öteleme/döndürme/tekdüze ölçekli sınırlı modeldir. 3B biçerdöverin eğik ve üstten görünümlerini tek dönüşümle açıklayamayabilir. Bu deneyde geometri çalışmadığı için mevcut başarısızlığın sebebi ilan edilemez. Homography de genel 3B yüzey değişimini sihirli biçimde çözmez. [OpenCV dönüşüm belgesi](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). İlk prompt kısmi affine modeli istediğinden ana baseline korunmalı.

Dinamik bellek ilk güvenilir eşleşme olmadan başlayamaz. Bu deneyde belleğe eklenmiş bir görünüm olmadığı için “dynamic başarısız, static daha iyi” sonucu da çıkarılamaz. Anchor eşikleri 0,75/0,80 de ileride ayrı kalibrasyon gerektirebilir; tespit eşiğini düşürmek bunları otomatik olarak anlamlı hale getirmez.

## 5. Öncelikli iyileştirme planı

| Sıra | İş | Başarı ölçütü / sınır |
|---|---|---|
| P0 | Aday bbox, crop sınırı, ham/temiz aday sayısı, DINO sırası, üretim token sayısı ve kesilme nedenini kaydet | Bir hedef karesinde hangi kapının neden elendiği yeniden inference gerektirmeden anlaşılmalı |
| P0 | MPS bellek, toplam duvar süresi, gerçek cihaz, sistem available RAM/swap ve örnekleme kapsamını kaydet | Ölçülmeyen alan 0 yerine null/unsupported; GPU belleği ve CPU RSS ayrı |
| P1 | İlk üç kare yerine farklı sahnelerden 6–10 karede proposal inceleme | Hedef önerilmediyse DINO eşiğiyle uğraşmayı bırak; doğru bbox geliyorsa retrieval incele |
| P1 | Aynı karelerde CPU–MPS fp32 karşılaştırması | Warmup ayrı; 3 model + geometri pozitif kontrol; finite çıktılar, skor/kutu farkları, medyan/p95 ve bellek |
| P1 | Florence cache uyumluluğunu onar/doğrula; sonra beam 3→1 deneyi | Sabit model revision; token/kutu tamlığı ve hedef kapsaması bozulmadan hız kazancı |
| P2 | Retrieval bütçesi kapısı ile nihai kabul eşiğini ayıran teşhis modu | En çok 1–3 çift; debug sonucu FOUND/bellek kabulü üretmesin; görsel ve geometrik kanıt ayrı incelensin |
| P2 | CLS, patch temsili ve sınırlı yön varyantı ablation'ı | Aynı kare/adaylar; model indirmeden; zor negatifleri de karşılaştır; varyantlar bellek güncellemesi sayılmasın |
| P3 | Florence base-ft veya yalnız teşhis amaçlı sınırlı tiling | Sadece aday üretimi sorunu doğrulanırsa; ayrı deney, proposal kapsaması ve maliyet birlikte |
| P4 | Ortak hat işe yaradıktan sonra static/dynamic kıyas | Aynı video örneklemesi/cihaz/encoder/geometri bütçesi, kronolojik akış, eşikler sabit |

MPS'yi ilk önce Florence ve DINO üzerinde denemek anlamlıdır. EfficientLoFTR MPS üzerinde desteklenmeyen işlemlere takılırsa açıkça kaydedilen hibrit cihaz seçimi değerlendirilebilir; gizli CPU fallback ile “GPU hızlandı” sonucu verilmemeli. fp16 şu an kod tarafından CPU/MPS için reddediliyor; yalnız YAML'a fp16 yazmak uygun optimizasyon değil. Daha düşük hassasiyet ancak ayrı nümerik ve uyumluluk testiyle ele alınmalı.

Masaüstündeki `configs/default.yaml` inceleme anında `device: mps` içeriyor; fakat analiz edilen run manifest'i `device: cpu` gösteriyor. Run kanıtı manifest'tir. Ayrıca `uav-refmatch ui` komutunun config argümanı verilmeden YAML'ı kendiliğinden okuduğu varsayılmamalı: CLI config'i yalnız `--config` üzerinden geçiriyor. Önceki “YAML'ı değiştir” önerim tek başına yeterli değildi. Sonraki kısa MPS deneyi açık `--config /Users/azra/Desktop/uav-reference-matching/configs/default.yaml` ile başlatılmalı ve gerçek cihaz manifest'ten doğrulanmalı. Açık UI eski config'i bellekte tutabilir; yeniden başlatılmadan etkilenmez.

Florence base-ft 0,23B, large-ft 0,77B parametredir; bu oran gerçek hız oranı değildir. Resmi model kartı `<REGION_PROPOSAL>` görevini destekler ama bu UAV hedefi için proposal garantisi sunmaz. [Microsoft Florence model kartı](https://huggingface.co/microsoft/Florence-2-large-ft). Beam ve cache değişikliklerini, model küçültmeden önce tek değişkenli denemek mevcut ağırlıkları yeniden kullanır.

EfficientLoFTR resmi uygulamada `full` ve `opt` seçenekleri, reparameterization ve farklı hassasiyetler sunuyor; mevcut kod reparameterization zaten yapıyor. [Resmi EfficientLoFTR deposu](https://github.com/zju3dv/EfficientLoFTR). Şu an geometri maliyeti sıfır olduğundan `opt`a geçmek mevcut darboğazı çözmez. Başlangıçta 1 çiftlik teşhis, ardından en çok 3 çift bütçesi yeterlidir.

## 6. Bütçeyi koruyan deney protokolü

İlk paket en fazla 10 farklı kare, yalnız mevcut LVD/large ağırlıkları ve static bellek: hedef açık, kısmi görünüm, küçük görünüm ve gerçekten hedef olmayan zor negatifler mümkün olduğu ölçüde seçilsin. İlk 44 saniyeden 0, 8, 16, 24, 32, 40 sn gibi aralıklı noktalar başlangıçtır; hedef var/yok statüsü görüntüden ayrıca incelenmeli. Gelecek karelerden referans üretip geçmişe taşımak yok.

Mevcut kayıtlarla eşik değişiminin sadece kaç adayı geçireceği ucuzca analiz edilebilir; doğru nesne olup olmadığı öğrenilemez. İlk yeni inference paketinde kutu metadata'sı ve bütün skorlar tutulunca tekrar Florence çalıştırmadan aynı log üzerinden eşik eleme dağılımları incelenebilir. Kalıcı crop/frame cache yapılmaz. Resmî static/dynamic deneyde ise nedensellik ve aynı ayar şartları korunur.

Deneyleri sırayla yap: (A) gözlemlenebilirlik ve mevcut CPU; (B) aynı girdiyle MPS; (C) uyumlu cache; (D) beam=1; (E) yalnız gerekli ise temsil/crop ablation. Bütün kombinasyonları tarama. Her koşula başlangıçta 10 dakika duvar süresi üst sınırı öneriyorum; bu sınır henüz koda eklenmiş değil. Model çağrısı içinde iptal gecikebilir, kontrollü süreç sonlandırması ayrıca tasarlanmalı. Bir yöntem açıkça hatalıysa sonraki koşullara/uzun videoya geçme.

Geçiş koşulu: önerilen kutu doğru hedefi kapsıyor mu, doğru aday ilk K içine giriyor mu, en azından temsilî hedef görüntülerinde geometri kanıtı üretiliyor mu, zor negatiflerde kontrolsüz kabul oluyor mu? Bunlar sağlanmadan tüm video ve dynamic kıyasına bütçe ayırma. 10 kare yalnız teşhis paketidir; akademik başarı oranı için yeterli örnek değildir.

Eski GT dosyası okunmadı ve inference'a bağlanmamalı. Bağımsız etiketli değerlendirme ileride kapsam olarak seçilirse yalnız değerlendirme/kalibrasyon katmanında kullanılmalı, test bölümü ayrı tutulmalı; karar veya bellek için koordinat kaynağı olmamalı. Mevcut GT'siz kapsamda precision/recall uydurulmaz.

Şimdilik yapılmaması gerekenler: SAT ViT-L indirmek; YOLO/SAM/RL/VLM eklemek; bellek kapasitesini büyütmek; videoyu 1 FPS'e düşürüp 7,5 FPS baseline sonucu diye sunmak; her kareyi tile etmek; bütün referans varyantlarını her kare yeniden encode etmek; ölçmeden cloud kiralamak. Seyrek Florence + tracker tasarımı daha sonra ayrı mimari deney olabilir; ilk baseline'daki her örnek karede Florence sözleşmesini değiştirir.

## 7. Tekrarlanabilirlik ve sınırlar

`RUN_AUDIT_2026-11-06.json`, beş run için bütün zaman dağılımlarını, kaynak istatistiklerini, aday dağılımlarını, bellek olay sayılarını, dosya boyutlarını ve beş temel kanıt dosyasının SHA256 değerlerini içerir. `scripts/audit_runs.py` yalnız standart Python kütüphanesi kullanır; model yüklemez. Tahmin/zaman/kaynak satır sayıları manifest ile ve kare indeksleri birbirleriyle doğrulandı. p95 yöntemi JSON içinde belirtilmiştir. Bellek örnekleri toplanıp “toplam tüketilen RAM” diye sunulmaz.

Yeniden üretme:

```bash
python3 /Users/azra/Documents/ChatGPT/uav-reference-matching/scripts/audit_runs.py \
  /Users/azra/Desktop/uav-reference-matching/runs \
  /Users/azra/Documents/ChatGPT/uav-reference-matching/docs/RUN_AUDIT_2026-11-06.json
```

Bu teslim araştırma ve teşhistir. MPS hızlanması, cache düzeltmesi, yeni eşik başarısı veya hedefin bulunması henüz ölçülmüş sonuç değildir. Mevcut uygulama entegrasyon testlerini geçmiş olsa da gerçek UAV hedefini bulma bakımından henüz başarılı baseline sayılmamalıdır.
