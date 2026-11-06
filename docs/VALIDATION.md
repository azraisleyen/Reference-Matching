# Doğrulama raporu — 6 Kasım 2026

## Teslim edilen uygulama

Ana proje: `~/Desktop/uav-reference-matching`. Sanal ortam burada kuruldu ve testler bu klasörden de çalıştırıldı. CLI ve yerel Gradio aynı inference servisini kullanır. Başlangıçtaki çalışma alanının boş Git deposu korunmuştur; teslim edilen Masaüstü deposunda anlamlı yerel commit'ler vardır. Remote oluşturulmadı, push yapılmadı.

## Ortam

- macOS arm64, Apple M4, 16 GB birleşik bellek; Python 3.11.18.
- PyTorch 2.9.1 / Torchvision 0.24.1 / Transformers 4.57.6.
- OpenCV headless 4.12.0.88 / PyAV 16.0.1 / NumPy 2.2.6.
- Gradio 6.0.2 / Pydantic 2.11.10 / FastAPI 0.119.1 / Starlette 0.48.0.
- FFmpeg/ffprobe mevcut. NVIDIA/CUDA yok; gerçek inference kontrolleri **CPU/fp32** ile yapıldı.
- Tam bağımlılık listesi `requirements-lock-macos-arm64.txt`; doğrudan bağımlılıklar `pyproject.toml` içinde sabit.
- Sistem seri numarası veya token kaydedilmez. MPS/CUDA inference doğrulaması yapılmadı.

## Test sonuçları

- **63 model gerektirmeyen test geçti.** Varsayılan suite'te 3 gerçek-model testi bilinçli atlanır.
- Son birleşik kontrolde **66/66 test geçti (63 çekirdek + 3 gerçek model)**.
- **3 gerçek-model entegrasyon testi ayrıca açılarak çalıştırıldı ve geçti** (`HF_HUB_OFFLINE=1 UAV_REAL_MODELS=1 pytest tests/test_integration.py -v -s`, önbellek hazırken 7,51 saniye). Atlanan testler bu sonuca dahil edilerek başarı sayılmamıştır; ayrı gerçek çalıştırma vardır.
- Ruff lint, Ruff format ve `pip check` başarılı.
- CFR/VFR gerçek PyAV video decode; kontrollü eksik PTS fallback; CFR/VFR sonuç video PTS korunumu; bbox çizimi; yanlış kaynak hash reddi; kısmi render sınırı doğrulandı.
- Statik/dinamik ortak pipeline, tek kare kaydı, ERROR ayrımı, iptal, süreç kilidi, anormal worker manifest kapatma, CSV/JSONL/HTML, CLI doctor ve UI JSON aktarımı test edildi.
- Anchor korunumu, kapasite, FIFO, duplicate, kalite/geometri kapıları, cooldown, pending, hareketli bbox, nedensellik ve reset testleri geçti.

## Gerçek model doğrulaması

| Model | Sonuç | Kanıt |
|---|---|---|
| Florence-2 large-ft | Resmî ağırlıkla `<REGION_PROPOSAL>` inference geçti; gerçek UAV karelerinde proposal üretti | `uav_validation.json`, gerçek run manifest'leri |
| DINOv3 LVD ViT-B/16 | Kullanıcının referansından `(1,768)` embedding, L2 norm `1.0` | `dinov3_validation.json` |
| EfficientLoFTR full/outdoor | Resmî örnek görüntü kimlik eşleştirmesinde 2.938 eşleşme/inlier; RANSAC kabul | `efficientloftr_validation.json` |
| DINOv3 SAT ViT-L/16 | Gated erişim/config doğrulandı; büyük ağırlık indirip inference testi yapılmadı | `check_model_access.py` ile tekrarlanabilir |

Florence large-ft commit: `4a12a2b54b7016a48a22037fbd62da90cd566f2a`.
DINO LVD commit: `5931719e67bbdb9737e363e781fb0c67687896bc`.
EfficientLoFTR commit: `07e9c1401e71e9c556b1fda9d86d060af36ba176`.
Outdoor checkpoint SHA256: `0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878`.

İlk DINO erişimi 401/GatedRepoError idi; kullanıcı girişinden sonra açıldı. Florence Xet aktarımı ilerlemedi; aynı resmî checkpoint standart HTTP ile tamamlandı. Bunlar başarı olarak gizlenmemiş kurulum engelleridir. İndirme süreleri inference hızı değildir.

## Gerçek UAV dosyalarıyla uçtan uca kontrol

Kullanıcının `input_video.mp4` ve `ref_01.jpg` dosyaları Downloads konumlarından okundu; projeye kopyalanmadı.

Video: 1920×1080, `30000/1001` FPS, 301,034 saniye, 9.022 kaynak kare. Referans: 3019×2145 piksel. Model/API/çıktı doğrulaması **yalnız ilk 3 örneklenen kareyle** sınırlandırıldı (`max_frames=3`). İşlenen kaynak indeksleri `0,4,8`; zamanlar `0`, `0.133466…`, `0.266933…` saniye.

- Dynamic: completed, 3 NOT_FOUND, 0 ERROR.
- Static: completed, 3 NOT_FOUND, 0 ERROR.
- Her karede bir proposal vardı; en güçlü cosine ortalaması yaklaşık `0.1142`, başlangıç eşiği `0.70` altında. Bu nedenle bütçe harcayıp LoFTR çalıştırılmadı; geometri süreleri sıfır, geometrik kanıtlar null.
- Dynamic bellek bu uygunsuz tespitlerden yeni referans eklemedi. Statik banka yalnız R0 kaldı.
- Her run için manifest, tam tahminler, CSV, bellek olayları, zaman/kaynak kayıtları, JSON/HTML özet ve log oluştu.
- `report` ve `render` CLI komutları gerçek kayıtlar üzerinde çalıştı. Anotasyonlu videolar son işlenen kaynak karede bitiyor; NOT_FOUND karelerinde kutu yok.
- Nihai run isimleri `real_run_paths.json`; sonuç kanıtı `uav_validation.json`. Run'lar `runs/` içinde ve Git dışında.
- Bu kısa kontrolde kare başına yaklaşık 3,5 saniye ölçüldü. Bu **tüm video veya genel hız benchmark'ı değildir**; kabul edilen geometri çiftleri yoktur ve CPU/fp32 kullanılmıştır.

## Sentetik pozitif kontrol (gerçek modeller)

Kullanıcının referans görüntüsünden yalnız OS geçici dizininde 12 karelik kısa video oluşturuldu. Üç gerçek model, aynı eşikler ve gerçek RANSAC çalıştırıldı; test doubles kullanılmadı. Görüntü/video girdileri işlem bitince geçici dizinle silindi.

- Static: 3 FOUND / 3 işlenmiş kare; bbox ve geometri kaydı var.
- Dynamic: 3 FOUND / 3 işlenmiş kare; aynı görünüm 3 kez `REJECTED_DUPLICATE` ile reddedildi.
- Her iki sonuç videosu kayıtlar okunarak üretildi, inference tekrar edilmedi.
- Kanıt: `positive_control_validation.json`.

**Bu kontrol bir UAV doğruluk deneyi değildir.** Aynı referansın videoda tekrar gösterilmesi, yalnız pozitif karar/bbox/geometri/render yolunun gerçek modellerle çalıştığını doğrular.

## Arayüz doğrulaması

Gradio gerçekten localhost üzerinde başlatıldı ve tarayıcıda incelendi. İlk 5.49.1 sürümünün JSON çıktı hatası 6.0.2 ile giderildi. Dynamic/Static, encoder seçimi, dosya/yol alanları, parametreler ve canlı durum/ilerleme göstergesi doğrulandı. Yerel yol metadata olayı her tuşta değil, alanı terk edince/Enter ile tetiklenir.

## Gerçek kalan araştırma işleri

- 301 saniyelik UAV videosunun tamamı henüz çalıştırılmadı. Tam run varsayılan olarak desteklenir; CPU üzerinde zaman alır.
- Gerçek UAV'de yeni görünüm kabulü, drift, kaybolma/yeniden görünme ve static/dynamic farkı uzun deneylerle ölçülmeli.
- Ground truth yok: precision, recall, F1, IoU doğruluğu ve yanlış bellek ekleme oranı hesaplanmadı.
- SAT, Florence-base, CUDA fp16 ve MPS yolları bu teslimde gerçek inference ile doğrulanmadı; ana doğrulama large-ft + LVD + full/outdoor + CPU/fp32 içindir.
- Eşikler optimal değildir; anote edilmiş bağımsız doğrulama kümesiyle kalibrasyon gerekir.

## Gerçek modellerle kontrollü yeni görünüm / bellek nedenselliği

Ek sentetik kontrol: aynı referans için 20° düzlem içi dönüş ve 0,85 ölçek uygulanmış video. Varsayılan eşikler, kapasite ve geometri bütçesi **değiştirilmedi**. Girdiler yalnız geçici dizindeydi; bu bir UAV benchmark'ı değildir.

- Static: 3 FOUND; tüm karelerde aktif banka 1 ve eşleşen referans R0.
- Dynamic: kare 0 `PENDING`; kare 1 `ACCEPTED(mem_0001)`; kare 2 yeni referansla `FOUND`.
- Tahmin anındaki aktif banka sayıları `1,1,2`. Dolayısıyla kabul edilen referans aynı karede kullanılmadı; yalnız sonraki karede kullanıldı.
- Üçüncü karede `REJECTED_COOLDOWN` kaydı var. Bitişte RESET kaydedildi.
- Her iki run'ın sonuç videosu yeniden inference yapmadan oluşturuldu.
- Kanıt: `memory_view_validation.json`.

Bu test, yalnız test doubles ile değil **üç gerçek modelin çıktılarıyla** pending → admission → sonraki karede retrieval zincirini doğrular. Gerçek UAV'de bellek başarısı/drift için uzun deney gereksinimini ortadan kaldırmaz.
