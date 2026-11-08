# UAV Reference Object Matching

Bir referans fotoğrafındaki **belirli fiziksel nesneyi** UAV videosunda arayan Python uygulaması. Her örneklenen kare için `FOUND`, `NOT_FOUND` veya `ERROR` üretir. Komut satırı ve Gradio arayüzü aynı işlem hattını kullanır.

## İşlem hattı

```text
Video / PTS → kare örnekleme → Florence-2 aday kutuları
→ DINOv3 temsilleri / cosine benzerliği → EfficientLoFTR
→ RANSAC geometrik doğrulama → karar → referans belleği / kayıtlar
```

- **Static:** yalnız başlangıç referansı (`R0`) kullanılır.
- **Dynamic:** güvenilir yeni görünümler sınırlı bir referans belleğine eklenir; `R0` korunur.
- Aday üretimi, eşikler, döndürme denemeleri ve toplam geometrik doğrulama bütçesi YAML ile ayarlanır.
- Yeni bellek kaydı yalnız sonraki karelerde kullanılabilir.

## Gereksinimler

- Python **3.11 veya 3.12**, Git, FFmpeg ve ffprobe.
- Modelleri ilk kez indirmek için internet ve yeterli disk alanı.
- DINOv3 için Hugging Face hesabı ve kullanılacak modelin erişim onayı.
- CPU, NVIDIA CUDA veya açıkça seçilen Apple Silicon MPS cihazı.

`configs/default.yaml` cihaz olarak `mps` seçer. Apple Silicon dışındaki sistemlerde `models.device` değerini `cpu` veya `cuda` yapın. `auto`, CUDA varsa CUDA'yı; yoksa CPU'yu seçer. CPU kullanırken `models.florence_precision: fp32` seçin. CUDA kurulumunda [PyTorch kurulum seçicisini](https://pytorch.org/get-started/locally/) kullanın.

`target_fps` giriş örnekleme sıklığıdır; uygulamanın saniyede aynı sayıda kare işleyeceği anlamına gelmez. Bellek ihtiyacı model, görüntü boyutu ve eşleştirme bütçesine bağlıdır.

## Kurulum

### 1. Depoyu klonlayın

```bash
git clone https://github.com/azraisleyen/Reference-Matching.git
cd Reference-Matching
```

### 2. Python ortamını hazırlayın

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

FFmpeg ve ffprobe'u sistem paket yöneticinizle veya [FFmpeg indirme sayfasından](https://ffmpeg.org/download.html) kurup PATH'e ekleyin. Ortamı kontrol edin:

```bash
ffmpeg -version
ffprobe -version
uav-refmatch doctor
python -m pip check
```

`pyproject.toml` doğrudan bağımlılıkları sabitler. `requirements-lock-macos-arm64.txt`, macOS ARM64 ortamının bağımlılık kaydıdır; diğer platformlar için kurulum listesi olarak kullanılmamalıdır.

### 3. Model erişimini hazırlayın

| Bileşen | Model / kaynak |
|---|---|
| Florence-2 | [Microsoft Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft) |
| Varsayılan DINOv3 | [Meta ViT-B/16 LVD](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) |
| Alternatif DINOv3 | [Meta ViT-L/16 SAT](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) |
| EfficientLoFTR | [Resmî kaynak ve ağırlık bağlantıları](https://github.com/zju3dv/EfficientLoFTR) |

Kullanacağınız DINOv3 modelinin koşullarını model sayfasında kabul edin. Erişim onayından sonra:

```bash
hf auth login
python scripts/check_model_access.py
```

Erişim aracı alternatif modelleri de kontrol eder; kullanmadığınız SAT modelinin erişim hatası LVD kullanımını engellemez. Tokenı YAML'a veya Git'e yazmayın. `.env.example` ortam değişkenlerini açıklar; uygulama `.env` dosyasını otomatik yüklemez.

Florence ve DINOv3 dosyaları ilk kullanımda Hugging Face cache'ine indirilir. EfficientLoFTR kaynağını ve checkpoint'ini ayrıca hazırlayın:

```bash
git clone https://github.com/zju3dv/EfficientLoFTR.git /yerel/modeller/EfficientLoFTR
git -C /yerel/modeller/EfficientLoFTR checkout 07e9c1401e71e9c556b1fda9d86d060af36ba176
```

`/yerel/modeller` örnek yoldur; kendi klasörünüzle değiştirin. `eloftr_outdoor.ckpt` dosyasını [resmî ağırlık klasöründen](https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q) indirin. `configs/default.yaml` içindeki ilgili alanları düzenleyin:

```yaml
models:
  eloftr_repo: /yerel/modeller/EfficientLoFTR
  eloftr_checkpoint: /yerel/modeller/eloftr_outdoor.ckpt
```

Bu alanları mevcut `models` bölümüne uygulayın; bölümün diğer alanlarını silmeyin. Alternatif olarak `ELOFTR_REPO` ve `ELOFTR_CHECKPOINT` ortam değişkenlerini kullanabilirsiniz. Alanlar boşsa `~/.cache/uav-refmatch/EfficientLoFTR` ve `~/.cache/uav-refmatch/eloftr_outdoor.ckpt` aranır (`XDG_CACHE_HOME` desteklenir).

Kaynak ve checkpoint dosyaları repoya dahil değildir. Kaynak commit'i ve model revision'ları doğrulanır; eksik model yerine başka model kullanılmaz. Model lisansları ve erişim ayrıntıları: [docs/MODELS.md](docs/MODELS.md).

## Kullanım

### Gradio arayüzü

```bash
uav-refmatch ui --config configs/default.yaml
```

1. Terminalde gösterilen yerel adresi açın; varsayılan `http://127.0.0.1:7860`.
2. Video ve referans fotoğrafını seçin veya tam dosya yollarını girin.
3. Modu, cihazı ve çalışma adını kontrol edip başlatın.
4. Sonuç bölümünde kayıtları ve kareleri inceleyin; **Video Oluştur** ile etiketli videoyu üretin.

İptal, devam eden model çağrısı tamamlandığında uygulanır. Aynı anda tek model çalıştırması desteklenir. Büyük dosyalarda yerel dosya yolu kullanmak yükleme kopyası oluşturmayı önler.

### Komut satırı

```bash
uav-refmatch run --video "/veri/input_video.mp4" --reference "/veri/ref.jpg" \
  --config configs/default.yaml --mode dynamic --name "ucus01_dynamic"

# Sabit referansla çalıştırma:
uav-refmatch run --video "/veri/input_video.mp4" --reference "/veri/ref.jpg" \
  --config configs/default.yaml --mode static --name "ucus01_static"

# Kısa kurulum kontrolü için run komutuna --max-frames 3 ekleyin.
uav-refmatch report --run "runs/CALISMA_KLASORU"
uav-refmatch render --run "runs/CALISMA_KLASORU"

# Kaynak video taşındıysa aynı dosyanın yeni yolunu belirtin:
uav-refmatch render --run "runs/CALISMA_KLASORU" --video "/yeni/yol/video.mp4"
```

**Tek kullanım yapılandırması `configs/default.yaml` dosyasıdır.** CLI ve arayüz `--config` verilmediğinde çalışma dizinindeki bu dosyayı okur. Başka bir dizinden çalıştırırken `--config` ile tam yolu belirtin. Dosya bulunamazsa sessizce başka ayarlara geçilmez. YAML dosyaları birbirinden miras almaz; eksik alanlar şema varsayılanlarından tamamlanır. Arayüzde değiştirilen mod/cihaz gibi seçimler çalıştırmanın `resolved_config.yaml` dosyasında kaydedilir. YAML değişiklikleri için arayüzü yeniden başlatın.

### VS Code

Proje klasörünü açın, Python yorumlayıcısı olarak `.venv` ortamını seçin ve bütünleşik terminalde yukarıdaki komutları çalıştırın. Terminalin çalışma dizini repo kökü olmalıdır.

## Doğrulama ve bellek ayarları

- `retrieval.max_geometry_pairs`: normal eşleştirme ve döndürmelerin paylaştığı toplam LoFTR çağrı sınırı.
- `geometry.reserved_rotation_calls`: normal çiftlerin bütün bütçeyi tüketmesini önleyen döndürme payı. Yeni bir ek bütçe değildir.
- `geometry.rotation_retry_policy`: `cosine` en benzer doğrulanmamış çifte, `evidence` geometrik desteğe öncelik verir; `legacy` eski sıralamayı korur.
- `models.loftr_min_long_side`: yalnız küçük crop'ların uzun kenarı için alt çözünürlük sınırı; `0` büyütmeyi kapatır. Üst sınır `loftr_max_side` ile korunur.
- `memory.capacity`: R0 dışındaki ek kayıt kapasitesi; LoFTR çağrı sayısını artırmaz.
- `memory.use_verified_anchor`: kazanan adayın zaten hesaplanmış R0 geometrisini bellek kabulünde kullanır. Başka adayın kanıtı kullanılmaz; ek model çağrısı yapılmaz.

Geometrik kabul eşikleri ve belleğin benzerlik, kalite, tekrar ve ardışık onay kontrolleri korunur. Kayıtlardaki `anchor_geometry` bellek kararının R0 desteğini, `geometry.diagnostics` ise dönüşüm, ölçek, eşleşme sayıları ve başarısız koşulları açıklar. Ayarlar tüm nesnelerde aynı doğruluğu garanti etmez.

## Çıktılar ve video gösterimi

Her çalışma benzersiz bir `runs/` alt klasörüne yazılır. Çıktı kökü `--output-root` ile değiştirilebilir.

| Dosya | Açıklama |
|---|---|
| `run_manifest.json` | Durum, kaynak hash'leri, model sürümleri, ortam ve Git commit'i |
| `resolved_config.yaml` | Gerçekte kullanılan tüm ayarlar |
| `predictions.jsonl` | Kare kararları, adaylar, cosine değerleri ve geometri kanıtları |
| `detections.csv` | Düzleştirilmiş tespit kayıtları |
| `memory_events.jsonl` | Bellek kabul, ret, bekleme ve çıkarma olayları |
| `frame_timings.csv`, `resource_usage.csv` | Aşama süreleri ve kaynak kullanımı |
| `summary.json`, `summary.html` | Çalıştırma özeti |
| `run.log` | Tanı kayıtları |
| `annotated_video.mp4`, `render_manifest.json` | İsteğe bağlı video ve çizim/zamanlama bilgileri |

- **Yeşil:** yalnız `FOUND` olarak kabul edilen nesne kutusu.
- **Turuncu:** aday kutuları ve kayıtlı cosine değerleri; kabul edilmiş tespit anlamına gelmez.
- Aynı kutuda yeşil tespit çizgisi turuncu aday çizgisinin üzerine çizilir.
- Yalnız işlenmiş kaynak karelere çizim yapılır. Ara karelere kutu taşınmaz; bu yüzden kutular kesikli görünebilir.
- Video üretimi modelleri tekrar çalıştırmaz; kaynak PTS zamanlamasını korur, ses eklemez.
- Kısmi veya iptal edilmiş çalıştırmaların videosu son işlenmiş karede biter.
- Eski kayıtlar aday bilgisi içermiyorsa aday kutusu veya skor uydurulmaz.

Ham video, referans fotoğrafı ve model ağırlıkları sonuç klasörüne kopyalanmaz. Kaynak yollarının yerel erişim kaydı `~/.cache/uav-refmatch/sources/` altında tutulur. Kaynak taşındığında yeni yolun hash'i doğrulanır.

## Proje yapısı

```text
configs/default.yaml       Kullanım yapılandırması
src/uav_refmatch/
  core/                    Tipler, protokoller ve çalışma ortamı
  data/                    Video okuma ve kare örnekleme
  proposals/               Florence-2 ve kutu hazırlama
  features/                DINOv3 temsilleri
  matching/                Retrieval, EfficientLoFTR ve yönelim
  geometry/                RANSAC doğrulama
  decision/                Tespit kararı
  memory/                  Statik/dinamik referans belleği
  evaluation/              Süre ve kaynak ölçümü
  outputs/                 Kayıt, rapor ve video çizimi
  pipeline.py              Ortak işlem hattı
  service.py, worker.py     Yaşam döngüsü, süreç ve iptal yönetimi
  cli.py, ui.py             Komut satırı ve Gradio
scripts/                   Kurulum ve erişim kontrol araçları
tests/                     Uygulama birim ve entegrasyon testleri
docs/                      Model ve lisans bilgileri
ui/                        İsteğe bağlı Python arayüz giriş noktası
experiments/               Yalnız yerel deneyler; Git dışında
runs/                      Yerel uygulama çıktıları; Git dışında
```

`experiments/` klonlanan depoda bulunmaz. Yerel deney ayarları, ölçümler, analizler ve yardımcı araçlar burada tutulur; CI/CD kapsamına girmez. Uygulama testleri ise kaynak kodla birlikte korunur. Deneyler için `--output-root experiments/runs` kullanılabilir. Geçmiş bir uygulama sürümüne dönmek için yalnız YAML değil, ilgili Git commit'i de gereklidir.

## Geliştirme ve test

```bash
python -m pytest -q
ruff check src tests ui scripts
ruff format --check src tests ui scripts
python -m pip check
```

Gerçek model entegrasyon testleri varsayılan olarak atlanır. Modeller ve erişimler hazırsa macOS/Linux üzerinde açıkça çalıştırılabilir:

```bash
UAV_REAL_MODELS=1 python -m pytest tests/test_integration.py -v
```

Bu testler model bağlantılarını doğrular; nesne tespit doğruluğu deneyi değildir. `FOUND` sayısı tek başına precision/recall göstermez. Küçük, örtülmüş, az dokulu nesneler ve sert bakış açısı değişimleri başarısızlığa yol açabilir. İşlem hattı bu durumlarda doğrulanmamış bir tespit üretmek yerine `NOT_FOUND` döndürebilir.
