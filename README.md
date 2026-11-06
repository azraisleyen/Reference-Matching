# UAV Reference Object Matching

Tek bir başlangıç fotoğrafındaki **belirli fiziksel nesneyi** UAV videosunda aramak için araştırma baseline'ı. Aynı sınıftan başka bir nesnenin doğru olduğu varsayılmaz. Sistem her örneklenen kare için `FOUND`, `NOT_FOUND` veya `ERROR` üretir. Ground truth okunmaz; doğruluk oranları uydurulmaz.

Araştırma sorusu: güvenilir yeni görünümlerin dinamik referans bankasına eklenmesi, sabit referansa kıyasla sonuçları ve hesaplama/bellek maliyetini nasıl değiştirir? Dinamik yöntemin daha başarılı olduğu **varsayılmaz**. Başlangıç eşikleri kalibre edilmemiştir.

## Mimari ve çalışma akışı

```text
PyAV / gerçek PTS → 7.5 FPS örnekleme → Florence-2 REGION_PROPOSAL
 → bbox temizleme / crop → DINOv3 CLS / L2 → cosine retrieval
 → bütçeli EfficientLoFTR full outdoor → affine RANSAC
 → tek kare kararı → tahmin kaydı → bellek kabulü → maliyet / olay kayıtları
```

`pipeline.py` tek inference motorudur. Model adaptörleri `core/interfaces.py` protokollerini uygular; gerçek modeller dışında çalışma modu yoktur. Test doubles yalnız `tests/` içindedir. `service.py` model yaşam döngüsü, kilit, manifest, hata ve iptali yönetir. CLI ve ayrı süreç kullanan Gradio aynı servisi çağırır. Modeller her deneyde bir kez yüklenir; kareler arasında yeniden yüklenmez.

`src/uav_refmatch/` altında video, aday üretimi, özellikler, eşleştirme, geometri, karar, bellek, profiler ve çıktılar ayrıdır. Gereksiz servis, veritabanı, Docker, training pipeline, GT bağımlılığı yoktur. SAM, YOLO, VLM, RL veya öğrenilebilir füzyon içermez.

## Statik / dinamik karşılaştırma

- **Static:** yalnız değişmez `R0`; yeni embedding/görüntü kabulü yok.
- **Dynamic (varsayılan):** `R0` + en fazla 8 ek referans; ayrı görüntüler ve embedding'ler RAM'de tutulur.

İki mod aynı adayları, ön işlemeyi, eşikleri ve en fazla 3 geometrik eşleştirme bütçesini kullanır. SAT encoder seçimi ayrı deneydir. Karşılaştırmalarda aynı video, aynı revision, aynı konfigürasyon kullanılmalı; yalnız `mode` değişmelidir. GPU aritmetiğinde bit düzeyinde deterministik sonuç garanti edilmez; seed, kararlı sıralama ve RANSAC seed'i kaydedilir.

## Modeller ve lisanslar

| Bileşen | Baseline | Kaynak |
|---|---|---|
| Florence-2 | `microsoft/Florence-2-large-ft`, `<REGION_PROPOSAL>` | [Microsoft model kartı](https://huggingface.co/microsoft/Florence-2-large-ft) |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m`, CLS | [Meta LVD](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) |
| Alternatif encoder | `facebook/dinov3-vitl16-pretrain-sat493m` | [Meta SAT](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) |
| EfficientLoFTR | resmî `full`, outdoor checkpoint | [ZJU resmî depo](https://github.com/zju3dv/EfficientLoFTR) |

Florence için yalnız allowlist'teki Microsoft depolarında, değişmez commit ile `trust_remote_code=True` kullanılır. Model kimliği ve commit birlikte değiştirilmelidir; base-ft alternatifinin commit'i `f6c1a25888ffc1d945ee8a1a77ac833c7303d46e`'dir. Remote kod gerekçesi: orijinal Microsoft checkpoint'inin özel mimarisi; başka checkpoint'e sessiz geçiş yoktur. `use_cache=False`, eski Florence decoder cache API'sine bağımlılığı kaldırır; hız bedeli vardır.

DINOv3 her checkpoint'in **kendi resmî processor'ını** kullanır. LVD/SAT normalizasyonları karıştırılmaz; boyut model config'inden alınır, 768 sabitlenmez. Erişim reddinde rastgele model yüklenmez. EfficientLoFTR resmî kaynağındaki `reparameter()` uygulanır; dış tensor adları yalnız adaptörde kullanılır. Lisans ve erişim ayrıntıları: [docs/MODELS.md](docs/MODELS.md).

## Ortam ve donanım

Python 3.11 veya 3.12, FFmpeg/ffprobe gerekir. Yerel doğrulama ortamı Apple M4, 16 GB birleşik bellek ve Python 3.11'dir. NVIDIA/CUDA yoktur. `auto`, CUDA varsa CUDA, yoksa CPU seçer. MPS açıkça seçilebilir ancak gerçek model uyumluluğu ayrıca test edilmelidir. CUDA fp16 isteğe bağlıdır; CPU/MPS için fp32 kullanılır. Üç modelin birlikte yüklenmesi ve batch/crop boyutları bellek tüketimini etkiler; ölçülmüş bir VRAM alt sınırı veya gerçek zamanlı hız iddiası yoktur.

**7.5 FPS, giriş örnekleme sıklığıdır; saniyede 7.5 inference tamamlanacağı garantisi değildir.**

## Kurulum

macOS / Linux:

```bash
cd ~/Desktop/uav-reference-matching
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
uav-refmatch doctor
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uav-refmatch doctor
```

FFmpeg ve ffprobe PATH üzerinde olmalı. CUDA sistemlerinde PyTorch wheel'ini [resmî kurulum seçicisi](https://pytorch.org/get-started/locally/) ile platforma uygun kurun. macOS ortam kilidi `requirements-lock-macos-arm64.txt` bütün kurulu sürümleri kaydeder; bu dosya CUDA/Linux için taşınabilir wheel garantisi değildir. `pyproject.toml` doğrudan bağımlılıkları sabitler. Ortamı değiştirdikten sonra `pip check` ve testleri çalıştırın.

## Checkpoint kurulumu

DINO model sayfalarında koşulları hesabınızla kabul edin; erişim onayından sonra:

```bash
hf auth login
python scripts/check_model_access.py
```

Tokenı koda, YAML'a veya Git'e yazmayın. `.env.example` örnektir; `.env` otomatik yüklenmez. HF standart token/cache mekanizması kullanılır. Ağırlıklar `~/.cache/huggingface` altında kalır.

EfficientLoFTR kaynak ve ağırlıkları proje dışında tutulur:

```bash
mkdir -p ~/.cache/uav-refmatch
git clone https://github.com/zju3dv/EfficientLoFTR.git ~/.cache/uav-refmatch/EfficientLoFTR
git -C ~/.cache/uav-refmatch/EfficientLoFTR checkout 07e9c1401e71e9c556b1fda9d86d060af36ba176
export ELOFTR_REPO="$HOME/.cache/uav-refmatch/EfficientLoFTR"
export ELOFTR_CHECKPOINT="$HOME/.cache/uav-refmatch/eloftr_outdoor.ckpt"
```

`eloftr_outdoor.ckpt` dosyasını [resmî README'deki ağırlık klasöründen](https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q) indirin. Checkpoint SHA256 değeri her gerçek deneyde kaydedilir. Kaynak commit'i uyuşmazsa veya tracked dosyalar değişmişse uygulama açık hata verir. `torch.load(weights_only=True)` ve strict state loading kullanılır. Alternatif olarak `models.eloftr_repo` / `models.eloftr_checkpoint` YAML alanlarını belirtin.

Hugging Face Xet aktarımı ağınızda takılırsa `HF_HUB_DISABLE_XET=1` ile aynı dosyanın standart HTTP aktarımı denenebilir; bu farklı model kullanmaz.

## CLI

```bash
uav-refmatch run --video "/veri/input_video.mp4" --reference "/veri/ref_01.jpg" \
  --mode dynamic --config configs/experiments/dynamic.yaml

uav-refmatch run --video "/veri/input_video.mp4" --reference "/veri/ref_01.jpg" \
  --mode static --config configs/experiments/static.yaml

uav-refmatch report --run "runs/DENEY_KLASORU"
uav-refmatch render --run "runs/DENEY_KLASORU"
# Kaynak taşınmışsa aynı içeriğin yeni yolu:
uav-refmatch render --run "runs/DENEY_KLASORU" --video "/yeni/yol/video.mp4"
```

Dosya isimleri sabit değildir. Boşluklu yollar tırnak içine alınır. `Ctrl+C` kontrollü iptal ister; devam eden model çağrısı bitince kayıtlar kapatılır. Başlangıç/model yükleme hatası `failed`, kullanıcı iptali `cancelled` olarak kaydedilir. Kare inference hatası önce `ERROR` kaydı üretir, sonra deney sonlandırılır; işlenmemiş karelere tahmin uydurulmaz. Decoder hatası henüz görüntüsü alınamamış bir kareye tahmin uydurmaz; run başarısız olur.

## Gradio

```bash
uav-refmatch ui --config configs/default.yaml
# Alternatif:
python ui/app.py
```

[Yerel arayüz](http://127.0.0.1:7860) üzerinden video/referans yüklenebilir veya yerel yollar girilebilir. Dynamic/Static, LVD/SAT ve deney adı seçilir. Video seçimi yalnız hızlı ffprobe metadata okur; tam video decode ve hash bu aşamada yapılmaz. Başlatmadan önce parametreler görülebilir.

Model hesabı spawn worker sürecinde yapılır. Arayüz durum kuyruğunu okur; model hesabını tekrar etmez. Ortak dosya kilidi CLI ve UI arasında da tek aktif deneyi zorunlu kılar. İptal, model çağrısını ortasından kesmez.

Sonuç sekmesinde kayıtlı deney, işlenmiş kare ve gerektiğinde kaynak videonun yeni yolu seçilir. Tam SHA256 kontrolü sonrasında görüntü kaynak videodan tekrar decode edilir. Bbox yalnız kayıttan çizilir. Önizleme için decoder baştan tarar; uzun videolarda gecikme olabilir. `Video Oluştur` varsayılan olarak çalışmaz; açık kullanıcı eylemiyle üretir.

Arayüz `127.0.0.1`, `share=False`, `allowed_paths=[]` kullanır. Girdi dosyaları sonuç dosyası olarak web'e sunulmaz. Gradio yükleme ve görüntü önizlemeleri için işletim sistemi geçici dizininde dosya oluşturabilir; bunlar proje/run çıktısı değildir. Uygulama kendi oluşturduğu geçici dizini normal kapanışta temizler; kaynak dosyalar silinmez. Anormal süreç sonlandırmada OS geçici dizini kalabilir. Büyük/uzun deneylerde yerel dosya yolu tercih edin. Geçici upload kaybolursa önizleme için özgün dosyayı yeniden seçin.

## Zaman ve koordinat sözleşmesi

`TimestampSampler` ilk geçerli PTS'yi başlangıç alır. Her `origin + n / target_fps` deadline'ına ulaşan ilk gerçek decode karesini seçer. Büyük PTS boşluğunda aradaki deadline'lar atlanır; aynı kare tekrar edilmez. 30 FPS için 0,4,8…; 25 FPS için değişen aralıklar; VFR için doğrudan PTS kullanılır. Kaynak hız düşükse mevcut kareler kullanılır ve manifest bunu raporlar.

PTS eksik/geriye gidiyorsa varsayılan davranış açık hatadır. Yalnız kullanıcı CFR'yi bağımsız doğrulamışsa `verified_cfr_fps` ayarıyla kontrollü fallback açılabilir; kullanım sayısı manifest'e yazılır. `r_frame_rate == avg_frame_rate` tek başına CFR kanıtı sayılmaz.

`frame_index` işlenmiş sıra, `source_frame_index` sıralı decode indeksi; ikisi de sıfır tabanlıdır. `timestamp_sec` orijinal PTS eksenindedir. Tüm bbox'lar kodlanmış orijinal çözünürlükte `(x_min,y_min,x_max,y_max)` biçimindedir. Sağ/alt sınır crop için exclusive'dir. Rotation metadata saklanır; otomatik döndürme uygulanmaz. Önizleme ve sonuç videosu aynı kodlanmış yönelimde gösterilir; boyutlar ve kutular tutarlı kalır.

Adaylar Florence'ın verdiği sırada doğrulanır, clip edilir ve kararlı NMS uygulanır. Florence güven puanı sağlamadığından NMS'e uydurma confidence verilmez. %10 padding yalnız crop'u etkiler; raporlanan bbox özgün proposal'dır. En güçlü 5 adayın referans çiftleri cosine'a göre sıralanır; eşitlik aday sırası/referans ID ile kırılır. En fazla 3 çift doğrulanır; başarısız ilk çift sonraki çiftin denenmesini engellemez.

EfficientLoFTR gri görüntüleri en uzun kenar 640 olacak şekilde sınırlar ve boyutları 32 katına getirir. Üretilen koordinatlar ayrı x/y ölçekleriyle özgün crop boyutlarına geri çevrilir. RANSAC eşikleri **aday crop'un özgün piksel uzayında** uygulanır. Spatial coverage, inlier convex hull alanının görüntü alanına oranıdır; iki crop'un minimumu alınır. Affine modelin 3D görünüm değişimlerini tam açıklayamaması bir baseline sınırlamasıdır.

## Dinamik bellek politikası

1. Geçerli `FOUND`, cosine ≥ .80, geometri geçerli, en az 12 inlier ve .40 oran gerekir.
2. Yeni crop'un **R0 embedding'iyle** cosine'ı ≥ .75 olmalı. Bu anchor kontrolü ek LoFTR çağrısı yapmaz; mevcut doğrulanmış çiftin geometrisi kullanılır. Dolayısıyla bütçe dışı gizli geometri yoktur. Bu global anchor kontrolünün drift'i kesin önlediği iddia edilmez.
3. Crop en az 24 piksel ve Laplacian varyansı ≥ 10 olmalı. Bu kalite eşikleri deneysel ve ölçek bağımlıdır.
4. R0 dahil bankadaki herhangi bir embedding'e cosine ≥ .96 ise tekrar olarak reddedilir.
5. Kabulden sonra 8 işlenmiş kare cooldown uygulanır.
6. Tek, sınırlı `PENDING` kayıt tutulur. Ardışık iki uygun tespitin embedding cosine'ı ≥ .85 olmalı. Bbox IoU zorunluluğu yoktur; kamera hareketine izin verir. Kesinti/uygunsuz tespit pending'i temizler. İkinci karenin crop'u kabul edilir; gelecekteki bilgi geçmiş kararı değiştirmez.
7. En fazla 8 dinamik kayıt; dolunca en eski dinamik kayıt FIFO ile çıkarılır. Anchor daima sabittir.
8. Tahmin kaydı **önce** yazılır; kabul **sonra** yapılır. `active(frame_index)` yeni kaydı yalnız bir sonraki işlenmiş karede döndürür. Her deney yeni bellek oluşturur; bitişte reset yapılır.

Görüntü ve embedding tekrar hesaplanmaz; aday için hesaplanan embedding belleğe aktarılır. Pending dahil görüntü/embedding byte sayıları profiler'a dahildir. Kabul, ret, pending, eviction, reset olayları kanıtlarıyla kaydedilir. Anchor koşulunu geçmeyen ancak dinamik referansla doğrulanan bir eşleşme hâlâ `FOUND` olabilir; belleğe eklenmez.

## Deney çıktıları

UTC başlangıç zamanı, mod, encoder ve rastgele benzersiz sonek içeren `runs/` klasörü oluşturulur. Saat dilimi ISO UTC olarak açık kaydedilir.

| Dosya | İçerik |
|---|---|
| `run_manifest.json` | Başlangıç/bitiş, durum, kaynak adları/SHA256, modeller/revision, ortam, Git SHA, fallback, sayaç |
| `resolved_config.yaml` | Deney konfigürasyonu |
| `predictions.jsonl` | Her işlenmiş kare için tek tam tahmin; doğrulanan çiftlerin kanıtları |
| `detections.csv` | Düzleştirilmiş kayıt; dört ayrı bbox sütunu, bulunamayan karelerde boş |
| `memory_events.jsonl` | Anchor/pending/kabul/ret/FIFO/reset olayları |
| `frame_timings.csv` | Decode, proposal, DINO, retrieval, LoFTR, RANSAC, decision, memory, total |
| `resource_usage.csv` | Süreç RAM'i, CUDA allocated/reserved/peak, referans/pending sayısı, görüntü/embedding byte |
| `summary.json`, `summary.html` | Durum dağılımı, skor/süre/olay özeti; doğruluk metriği yok |
| `run.log` | JSON satırlı tanı kayıtları |
| `annotated_video.mp4` | Yalnız kullanıcı isterse oluşturulur |
| `render_manifest.json` | Render zaman/koordinat/politika bilgisi |

Ham video, referans, kareler veya bellek crop'ları proje/run içine kopyalanmaz. Kaynak dosya yolları rapora yazılmaz; yerel tekrar erişim için `~/.cache/uav-refmatch/sources/` altında yalnız dosya yolu içeren 0600 izinli özel kayıt tutulur. Sonuçları paylaşırken bu özel dizini paylaşmayın. Büyük video hash'i akış halinde, metadata önizlemesinden ayrı hesaplanır.

Sonuç videosu orijinal karelerin PTS zamanlamasını korur. **Yalnız örneklenmiş FOUND karelerine** kutu çizilir; örneklenmemiş kareler için konum bilinmediğinden bbox taşınmaz/interpole edilmez. Bu nedenle tam FPS videoda kutular kesikli görünebilir. Ses kopyalanmaz. Eksik/tekrarlı PTS render'da açık hatadır. Video tekrar model çalıştırmaz.

## Ölçüm ve bilimsel sınırlamalar

Model yükleme, warm-up ve başlangıç referansı encoding süreleri steady-state kare sürelerinden ayrıdır. `processing_ms` tahmin kaydı anına kadardır (decode dahil, bellek güncellemesi hariç). `total_frame_ms` bellek ve tahmin/olay yazımını içerir; timing/resource CSV yazımı ve UI progress callback'i bunun dışındadır. Decode süresi, seçilmeyen ara kareleri çözmenin maliyetini de içerir. CUDA/MPS senkronizasyonu doğru ölçüm için kullanılır ve hız üzerinde ek maliyet oluşturabilir. CUDA peak, steady-state öncesinde sıfırlanır; modellerin resident belleği dahildir. CPU RAM değeri tüm süreç RSS'idir; bankanın byte hesabı ayrıca verilir.

Ground truth olmadan precision/recall/F1/IoU doğruluğu veya yanlış bellek ekleme oranı hesaplanamaz. Eski `gt_eski(ref_01).json` veya başka etiket dosyaları kullanılmaz. FOUND sayısı doğruluk kanıtı değildir. Küçük/hedefsiz/dokusuz crop'lar, affine model, benzer nesneler, occlusion ve sert görünüm değişimleri başarısızlığa yol açabilir. Bellek drift'i ayrı anotasyonlarla ölçülmelidir.

## Testler

```bash
pytest -q
ruff check src tests ui scripts
ruff format --check src tests ui scripts
pip check
# Gerçek ağırlıklar/izinler hazırsa (testler otomatik geçmiş sayılmaz):
UAV_REAL_MODELS=1 pytest tests/test_integration.py -v -s
```

Testler sentetik küçük verileri yalnız OS geçici dizinine yazar. Sampling/CFR/VFR, clipping/NMS, embedding, retrieval, affine/coverage, karar, nedensellik, anchor, FIFO, duplicate, pending, reset, ERROR, kayıtlar, gerçek video decode/render, CLI ve UI yapısı sınanır. Sentetik test doubles gerçek model doğrulaması veya akademik performans deneyi değildir. Gerçek doğrulama durumu [docs/VALIDATION.md](docs/VALIDATION.md) içinde raporlanır; gereksinim eşlemesi [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) içindedir.

## Geliştirme ve gelecekteki araştırmalar

Yeni encoder/proposal/matcher aynı protokole bağlanabilir. Bellek kabulü ayrı policy'dir; eviction gelecekte ayrı politika ile değiştirilebilir. Yeni karşılaştırmada config, model commit ve veri hash'lerini koruyun. Evaluation, ileride gerçek anotasyonlarla inference'dan bağımsız eklenmelidir. Diversity-aware bellek ve farklı geometrik modeller gelecekte araştırılabilir; mevcut baseline'a uygulanmış değildir.

Git ham veriyi, modelleri, `.env`, cache, sanal ortam ve run çıktılarını dışlar. Remote repository veya push otomatik oluşturulmaz. Gerçek bütün-model doğrulaması bitmeden `baseline-dynamic-v1` başarı etiketi kullanılmaz.

Kısa kurulum doğrulaması için `run --max-frames 3` kullanılabilir. Sınır konfigürasyon/manifest içinde kaydedilir; böyle bir run tüm videonun işlendiği anlamına gelmez. Varsayılan `max_frames: null` bütün videodur.

Resmî EfficientLoFTR Lightning checkpoint'i, bir sözlük anahtarı olarak `ModelCheckpoint` sınıfını içerir. `weights_only=True` korunur; yalnız doğrulanmış resmî SHA256 (`0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878`) için kurulu Lightning sınıfı dar `safe_globals` kapsamına alınır. Başka checkpoint'lerde bu istisna uygulanmaz.

EfficientLoFTR için YAML/env belirtilmezse standart `~/.cache/uav-refmatch/EfficientLoFTR` ve `~/.cache/uav-refmatch/eloftr_outdoor.ckpt` yolları otomatik aranır (`XDG_CACHE_HOME` desteklenir). Bu makinedeki kurulum bu yolu kullanır. Dosya yoksa açık hata verilir. Resmî fine matching modülünün koşulsuz CUDA autocast davranışı, fp32 seçildiğinde kendi `validate` bayrağıyla kapatılır; eval/frozen inference korunur. Bunun tuttuğu ek ara matrisler yalnız çift çağrısının ömrü boyunca RAM'dedir.

Kısmi (`max_frames` ile sınırlı), iptal edilmiş veya başarısız run'ların sonuç videosu son işlenmiş kaynak karede biter; işlenmeyen video kuyruğu sonuç gibi sunulmaz. Tamamlanmış ve sınırsız run tüm kaynak süreyi korur. Bu seçim `render_manifest.json` içinde açıkça yazılır.

`retrieval_candidates`, her adayın en iyi referansını ve gerçek cosine skorunu eşik altında kalsa da saklar. `dino_cosine`, FOUND için kazanan skordur; NOT_FOUND ve aday varsa en güçlü adayın skorudur. Eşik altındaki skorlar geometri bütçesini veya kabul kapısını atlamaz. Geometri çalıştırılmadığında JSON ve CSV geometrik alanları null/boştur.
