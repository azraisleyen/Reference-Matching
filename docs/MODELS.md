# Model kaynakları, erişim ve lisanslar

Kontrol tarihi: 6 Kasım 2026. Bu belge üçüncü taraf lisanslarının yerine geçmez; kullanmadan/yayımlamadan önce bağlantıdaki lisans metnini okuyun.

- Florence-2 large-ft/base-ft: Microsoft model kartında MIT. Kaynak: https://huggingface.co/microsoft/Florence-2-large-ft . Orijinal checkpoint/processor ve sabit commit ile remote code. `REGION_PROPOSAL` yalnız dahili görev belirtecidir; kullanıcıdan kategori/metin alınmaz.
- DINOv3 LVD/SAT: Meta'nın özel **DINOv3 License** koşulları; model sayfasında gated erişim. Hugging Face hesabı, koşulların kabulü ve token gerekir. Erişim koşulları ayrı her model için kontrol edilir. LVD: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m ; SAT: https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m . Lisans: https://ai.meta.com/resources/models-and-libraries/dinov3/license/ . Her checkpoint'in kendi processor'ı kullanılır.
- EfficientLoFTR: resmî kaynak https://github.com/zju3dv/EfficientLoFTR ; lisans https://github.com/zju3dv/EfficientLoFTR/blob/main/LICENSE . Resmî README outdoor ağırlıklarını Google Drive üzerinden sunar. Bu projede kaynak repo `full_default_cfg` ve `reparameter` yolu kullanılır. Hugging Face dönüşümü resmî README'de ayrıca belirtilse de bu baseline sessizce o uygulamaya geçmez.

Resmî API kaynakları:

- https://huggingface.co/docs/transformers/model_doc/florence2
- https://huggingface.co/docs/transformers/model_doc/dinov3
- https://github.com/zju3dv/EfficientLoFTR/blob/main/README.md

Yeni native Transformers Florence örnekleri `florence-community` checkpoint'leri kullanır. Kullanıcının istediği `microsoft/Florence-2-large-ft` kimliğini korumak için kendi remote sınıfı kullanılır. `attn_implementation=eager` ve legacy cache kapatma, güncel Transformers ile kontrollü uyumluluk seçimleridir; gerçek test durumu VALIDATION belgesindedir.

Model erişim araçları token değerlerini yazdırmaz. `GatedRepoError` erişim iznini, HTTP/ağ hataları aktarımı, eksik EfficientLoFTR kaynağı/ckpt dosyası kurulum eksikliğini belirtir. Hiçbir hata sahte çıktı veya başka modelle örtülmez.
