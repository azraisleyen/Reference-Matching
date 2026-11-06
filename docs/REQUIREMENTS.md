# Gereksinim izlenebilirliği

Kaynak: kullanıcının 0–23 bölümlü UAV Reference Object Matching master prompt'u. Bu tablo uygulama yerlerini gösterir; gerçek model doğrulama kanıtı ayrıca `VALIDATION.md` içindedir.

| Prompt bölümü | Uygulama / doğrulama |
|---|---|
| 0 Çalışma yöntemi | Ortam preflight, sabit bağımlılıklar, gerçek erişim kontrolleri, ayrı unit/integration testleri |
| 1 Araştırma / instance identity | Tek hedef, `NOT_FOUND`, ortak pipeline; başarı hipotez olarak ele alınır |
| 2 Kesin mimari | Florence → DINOv3 → cosine → resmî EfficientLoFTR → affine RANSAC → karar/bellek; yasak bileşenler yok |
| 3 Video / depolama | `data/video_reader.py`, `frame_sampler.py`; ffprobe; PTS; orijinal bbox; RAM crop; temp video testleri |
| 4 Florence | `proposals/florence2.py`, `boxes.py`; large-ft; sabit remote revision; NMS/clip/padding; confidence uydurulmaz |
| 5 DINOv3 | `features/dinov3.py`, `normalization.py`; resmî processor; CLS; L2; batch; model config dimension; R0 tek encoding |
| 6 Retrieval | `matching/retrieval.py`; max cosine; top5 aday; toplam 3 çift; kararlı tie breaking |
| 7 EfficientLoFTR | `matching/efficientloftr.py`; resmî full/outdoor; strict load; reparameter; gri/32 katı; koordinat geri ölçekleme |
| 8 Geometri | `geometry/verifier.py`; partial affine RANSAC, inlier/ratio/error/coverage/scale/degenerate kontrolleri |
| 9 Karar | `decision/engine.py`, `pipeline.py`; tek bbox veya null; ERROR ayrı; bütün işlenen kareler kayıtlı |
| 10 Bellek | `memory/`; ortak arayüz; frozen anchor; capacity8; ayrı admission; anchor cosine; pending1; cooldown; duplicate; FIFO; olaylar |
| 11 Nedensel sıra | Tahmin yazımı → memory.update → sonraki kare active filtresi; `test_pipeline`, `test_memory` |
| 12 Gradio | `ui.py`, `worker.py`; upload/yol, dynamic/static, LVD/SAT, metadata, ayrı süreç, iptal/kilit, kayıt inceleme/video üretimi |
| 13 Deney çıktıları | `outputs/recorder.py`, `reporting.py`, `evaluation/profiler.py`; JSONL/CSV/HTML, hash, config/revision/ortam, stage timing/RAM |
| 14 Mimari | `src/uav_refmatch` modülleri; protokoller/dataclass/Pydantic; model adaptörleri ayrı |
| 15 CLI | `cli.py`: run, doctor, render, report, ui |
| 16 Bağımlılıklar | `pyproject.toml`, platform kilidi; `doctor`; upstream API/kaynak doğrulaması |
| 17 Testler | CPU deterministic tests; gerçek opt-in entegrasyonlar; raporda açık sonuç ayrımı |
| 18 Git | `.gitignore`; yerel repo; remote/push yok; doğrulanmamış başarı etiketi yok |
| 19 Türkçe README | Kurulum, modeller/lisanslar, komutlar, protokoller, ölçüm/kısıtlar ve araştırma seçenekleri |
| 20 Aşama sırası | Ortam → tip/video → adaptör → karar/bellek → pipeline → UI → test/docs; son regresyon testleri |
| 21 Kalite / kapsam | Sahte model yok, GT yok, ham veri kayıt yok, sürüm/hash/olay kaydı; profiler overhead belgeli |
| 22 Teslim | VALIDATION ve nihai Türkçe rapor; gerçekleşen ve eksik işler ayrılır |
| 23 Öncelikler | Çalışan ortak çekirdek, nedensel bellek, static karşılaştırma, kare kayıtları, maliyet, UI ve tekrarlanabilirlik |

## Açık mühendislik seçimleri

- Anchor admission: R0 global cosine + kazanan çiftin geometrisi + zamansal doğrulama. Anchor'a ayrıca LoFTR çalıştırılmaz; geometri bütçesi aşılmaz.
- Pending kapasitesi 1; ardışık uygun kareler arasında embedding benzerliği; hareketli kamera için bbox IoU zorunlu değil.
- Tekrarlı filtre anchor dahil tüm bankayı karşılaştırır.
- Bütün geometrik bütçe sonuçları kaydedilir; yalnız kabul edilenler saklanmaz.
- Decode edilen görüntüler kodlanmış yönelimdedir; rotation metadata uygulanmaz, açıkça kaydedilir.
- Video renderer yalnız örneklenen kareye kutu çizer; aradaki karelere bilinmeyen konum tahmini taşımaz.
- CLI ile UI process lock paylaşır. Cancel güvenli model çağrısı sınırlarında işlenir.
- Warm-up için video karesi kullanılır; R0 embedding'i iki kez çıkarılmaz.
- Gated checkpoint veya yavaş/başarısız indirme hiçbir testte başarı olarak sayılmaz.
