# Proje Kodu Teknik Raporu (Phase 1)

## 1. Repository Yapısı
- `train.py`: Model eğitiminin yapıldığı ve pipeline'ı başlatan ana giriş dosyası (entry point).
- `evaluate.py`: Önceden çıkarılmış özellik (feature) vektörleri kullanılarak mAP ve Rank metriklerinin hesaplandığı dosya.
- `extract_features.py` / `test.py`: Eğitim sonrasında modellerden özellikleri çıkararak, değerlendirmede kullanılacak `.mat` dosyasını ürettiği öngörülen (repository içinde bulunan) script'ler.
- `model.py`: `ft_net`, `ft_net_swin`, `ft_net_dense` vb. ağ mimarilerinin ve sonundaki `ClassBlock` (Sınıflandırma ve Embedding) yapısının tanımlandığı dosya.
- `dataset.py`: Veri seti sınıflarını (`ImageDataset`) ve mini-batch içindeki sınıf dağılımını kontrol etmek için tasarlanmış örnekleyiciyi (`BatchSampler`) içeren modül.
- `main.py`: Eğitilen modelden bağımsız, varsayılan ImageNet ResNet50 ile 5 adet görsel getiren çok basit bir cosine similarity (retrieval) demosu içeriyor.

## 2. Mimari Bileşenler
- **Backbone**: Birden fazla ağ mimarisi desteklenmektedir (ResNet, DenseNet, Swin, NAS, HRNet, EfficientNet). `--model resnet_ibn` varsayılan argümandır, bu nedenle asıl model ResNet50-IBN olarak kullanılmaktadır.
- **Loss Fonksiyonları**: `--label_smoothing` (varsayılan: 0) destekli `CrossEntropyLoss` ana loss'tur. Argümanlar aracılığıyla `pytorch_metric_learning` kütüphanesinden ArcFace, CircleLoss, CosFace, Contrastive, Instance, Triplet, Lifted ve Sphere loss eklenebilmektedir. Triplet loss kullanıldığında `MultiSimilarityMiner` ile hard-mining yapılmaktadır.
- **Embedding Dimension**: Classifier blokta `--linear_num` argümanı ile kontrol edilir ve varsayılan olarak **512**'dir.
- **Dataset Loader**: Pandas DataFrame üzerinden yolu (`path`) okunan, sınıf ve kimlik (`target_label`) sütunlarıyla etiket yönetimi yapan custom bir dataloader kurgusu mevcuttur.
- **Augmentation (Pre-processing)**: Resize (varsayılan: 224x224), Pad(10), RandomCrop(224x224), RandomHorizontalFlip ve standart ImageNet Normalize yapılıyor. `--erasing_p` ile RandomErasing ve `--color_jitter` ile parlaklık/kontrast bozulmaları (jitter) açılabilmektedir.
- **Optimizer & Scheduler**: Stochastic Gradient Descent (SGD). Classifier learning rate normal (`opt.lr`, def:0.05) iken backbone learning rate %10'u kadardır (0.1x). `--warm_epoch` ile kademeli warmup desteklenmektedir ve varsayılan olarak `StepLR` (10 epoch'ta 0.1x azaltma) scheduler vardır (Alternatif olarak CosineAnnealingLR kullanılabilir).
- **Batch Construction**: Validation aşamasında normal `DataLoader`, eğitim aşamasında ise aynı ID'den birden fazla resmi aynı batch'e koymayı amaçlayan (PK-sampling) bir `BatchSampler` ile `--samples_per_class` parametresi kullanılmaktadır.

## 3. Değerlendirme (Evaluation) Protokolü ve Metrikler
- **Protocol**: `evaluate.py`, daha önce `extract_features.py` vb. bir script tarafından kaydedilmiş olan `pytorch_result.mat` isimli dosyadan query ve gallery tensörlerini alır. İki tensör arasında matris çarpımı (`torch.mm`) ile cosine similarity / dot product hesaplayarak sıralama (ranking) yapar. Aynı kamera (`query_cam == gallery_cam`) ve aynı araç ID'sine (`query_label == gallery_label`) sahip görüntüler filtrelemeden elenir (`junk_index`).
- **Metrics**: Rank@1, Rank@5, Rank@10 ve mean Average Precision (mAP) metrikleri, NumPy/PyTorch karmaşık mantığıyla manuel olarak `compute_mAP` fonksiyonunda hesaplanmaktadır.

## 4. Sistem & Checkpoint Yapısı
- **Dependencies**: `requirements.txt` içerisinde tanımlı olan bağımlılıklar görece ortalama eskiliktedir (Örn: `torch==1.12.0`, `torchvision==0.13.0`, `timm==0.6.7`).
- **Donanım/PyTorch Uyumluluğu**: Eski PyTorch/Torchvision versiyonları (özellikle < 1.10) için interpolasyon ve LabelSmoothing uyumluluk kontrolleri vardır. Kodda `torch.cuda.amp` kullanılarak Mixed Precision (FP16) desteği ve torch_xla kullanılarak TPU paralelleme desteği sağlanmıştır.
- **Checkpoint System**: Her N epoch (`save_freq`, def:5) sonunda ağırlıklar `model/{name}/net_{epoch}.pth` dosyası olarak kaydedilir. Eğitimi kimin yaptığı belli olsun diye kaynak kodları (`train.py`, `model.py`) ve config (`opts.yaml`) checkpoint klasörüne yedeklenmektedir.

## 5. Eski/Deprecated Kodlar ve Modernizasyon Önerileri
1. **Config Sistemi Eksikliği**: Parametreler çoğunlukla command-line argümanları (`argparse`) olarak geçilmektedir, fakat resim boyutu (224, 224) ve bazı scheduler özellikleri `train.py` içinde **hard-coded** bırakılmıştır. Bunlar yapılandırılabilir (örneğin `.yaml`) bir dosya üzerine çekilmelidir.
2. **Dağınık Test Pipeline'ı**: Çıkarım (Inference) ve Değerlendirme (Evaluation) adımları birbirinden ayrı dosyalardadır (`extract_features.py` -> `pytorch_result.mat` -> `evaluate.py`). Eğitim sırasında validasyon yapılarak epoch sonlarında mAP raporlanmamakta, sadece doğruluk (accuracy) verilmektedir. Bu pipeline entegre edilebilir.
3. **Hard-coded İsimlendirmeler**: `evaluate.py` içinde `.mat` dosyasının adı sabit (hardcoded) ve input parametresi olarak alınmıyor.
4. **Loglama Yetersizliği**: Terminal logları dışında basit bir matplot kütüphanesi çizimi (`train.jpg`) ve DataFrame takibi var. İleride TensorBoard loglayıcısı çok daha verimli olabilir.
5. **Dışa Bağımlılık Riske Açık**: `ResNet-IBN` doğrudan `torch.hub.load('XingangPan/IBN-Net')` şeklinde github'dan indirilmeye ayarlanmış. Benzer şekilde `NASNet` için sertifika doğrulama (SSL) by-pass edilmiş. Çevrimdışı (local) veya sertifika sorunu olan yerel çalışmalarda indirme (download) hatası verebilir.
