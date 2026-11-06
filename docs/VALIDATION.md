# Doğrulama raporu

Bu dosya doğrulama tamamlandığında nihai sonuçlarla güncellenir. Henüz tamamlanmamış model testi başarı sayılmaz.

## Ortam

- macOS arm64, Apple M4, 16 GB birleşik bellek.
- Python 3.11.18; PyTorch 2.9.1; Torchvision 0.24.1; Transformers 4.57.6.
- OpenCV headless 4.12.0.88; PyAV 16.0.1; NumPy 2.2.6.
- Gradio 6.0.2; Pydantic 2.11.10; FastAPI 0.119.1; Starlette 0.48.0.
- FFmpeg/ffprobe mevcut. NVIDIA/CUDA yok; varsayılan inference CPU/fp32.
- Sistem donanım kimlikleri/seri numarası rapora dahil edilmez.

## Çekirdek testleri

59 test geçti, gerçek model gerektiren 3 test varsayılan CPU suite'inde atlandı (skip). Bu skip'ler gerçek model başarısı değildir. Ruff lint ve format kontrolleri, pip dependency kontrolü ayrıca çalıştırılır.

Gerçek FFmpeg/PyAV ile CFR ve VFR dosyaları oluşturulup decode edildi; CFR sonuç videosunda tüm kaynak PTS'leri korundu. Test verileri işletim sistemi geçici dizinindedir. Statik ve dinamik uçtan uca mantık testleri yalnız test doubles kullanır; model performansı olarak raporlanmaz.

## Arayüz

Yerel Gradio sunucusu gerçekten başlatıldı ve tarayıcıda incelendi. İlk 5.49.1 sürümünün JSON bileşeni `root=...` metni döndürüyordu; 6.0.2 ile giderildi. Durum, mod/encoder, dosya/yol girdileri ve ilerleme göstergesi kontrol edildi.

## Gerçek modeller

- DINOv3 LVD/SAT erişimi ilk kontrolde 401/GatedRepoError idi. Kullanıcının girişinden sonra iki modelin sabit revision config dosyalarına erişim başarıyla doğrulandı.
- Florence-2 large-ft resmî checkpoint indirmesi sürüyor; ilk Xet aktarımı veri ilerlemediği için durduruldu, aynı checkpoint HTTP yöntemiyle devam ediyor.
- EfficientLoFTR resmî full/outdoor checkpoint ile gerçek CPU inference doğrulandı. Resmî örnek görüntünün kendisiyle eşleşmesinde 2.938 eşleşme/inlier; RANSAC geçerli. Kanıt: `efficientloftr_validation.json`. Bu bir kimlik eşleştirme entegrasyon testi, UAV başarısı ölçümü değildir.
- Gerçek model inference ve tam hattın sonuçları indirme sonrasında güncellenecek.

## Bilimsel kapsam

Henüz UAV benchmark sonucu, doğruluk yüzdesi veya dinamik belleğin üstünlüğü iddiası yoktur. Kullanıcının gerçek UAV video/ref verisiyle ölçüm ayrı deney olarak yapılmalıdır. Eşikler deneysel başlangıç değerleridir.
