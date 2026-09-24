# Bitirme Projesi Teknik Görev Dokümanı

## Proje Adı

**Vehicle Re-Identification Under Challenging Environmental Conditions**

---

## 1. Projenin Ana Amacı

Bu projenin amacı, **Vehicle Re-Identification (Vehicle Re-ID)** sistemlerinin zorlu çevresel ve görüntüleme koşullarındaki performansını incelemek ve mevcut bir Vehicle Re-ID pipeline'ını bu koşullarda daha dayanıklı hale getirmektir.

### Ana araştırma sorusu

> **Zorlu çevresel koşullar Vehicle Re-ID performansını ne kadar düşürüyor ve uygun veri artırma, görüntü iyileştirme veya modelleme yöntemleri bu performans kaybını ne kadar azaltabiliyor?**

### Kapsam sınırı

Projenin ana odağı **challenging-condition Vehicle Re-ID** olmalıdır. Bunun altında en fazla 2–3 kontrollü koşul seçilmelidir. Başlangıç için önerilen koşullar: **düşük ışık/gece**, **yağmur** ve **görüntü gürültüsü/kalite bozulması**.

Aşağıdaki konular ayrı araştırma problemleri olarak ele alınmamalıdır:

- benzer araçların ayırt edilmesi,
- vehicle detection,
- multi-object tracking,
- kamera takibi,
- gerçek zamanlı multi-camera tracking,
- mobil uygulama,
- production backend.

Benzer araçları ayırt edebilmek zaten Vehicle Re-ID görevinin temel gereksinimidir; bu konu ayrıca ikinci bir ana araştırma ekseni yapılmamalıdır. Aynı şekilde her türlü hava ve çevre koşulunu kapsamak yerine seçilmiş birkaç koşul kontrollü biçimde incelenmelidir.

---

# 2. Projenin Genel Akışı

```text
VeRi-776
   ↓
Mevcut Vehicle Re-ID kod tabanı
   ↓
Kodun analiz edilmesi
   ↓
Eski bağımlılıkların / API'lerin modernize edilmesi
   ↓
Baseline'ın yeniden üretilmesi
   ↓
Normal-light evaluation
   ↓
Challenging-condition benchmark oluşturulması
   ↓
Baseline'ın seçilen zorlu koşullardaki performansının ölçülmesi
   ↓
Challenging-condition enhancement yöntemlerinin uygulanması
   ↓
Kontrollü deneyler
   ↓
Karşılaştırma
   ↓
mAP / Rank-1 / Rank-5
   ↓
Sonuçların analizi
   ↓
Basit inference / retrieval demo
```

---

# 3. Başlangıç Kaynağı: Eski GitHub Repository

Elimizde başka bir araştırmacıya ait yaklaşık **5 yıllık bir GitHub repository'si** bulunmaktadır.

Bu repository projenin başlangıç kod tabanı olarak kullanılacaktır.

## Kritik kural

Repository'yi ilk aşamada yeniden yazma.

Önce mevcut kodun:

- ne yaptığını,
- hangi modeli kullandığını,
- hangi loss fonksiyonunu kullandığını,
- dataset pipeline'ının nasıl çalıştığını,
- evaluation sürecinin nasıl yapıldığını

anla.

Daha sonra yalnızca gerekli yerlerde modernizasyon / refactoring yapılmalıdır.

**Algoritmanın mantığı ile kod modernizasyonu birbirine karıştırılmamalıdır.**

---

# 4. AI Agent İçin İlk Aşama: Repository Analizi

Kod üzerinde değişiklik yapmadan önce aşağıdaki işlemler yapılmalıdır.

## 4.1 Proje yapısını çıkar

Aşağıdakileri tespit et:

- ana giriş dosyaları,
- training script,
- evaluation script,
- model dosyaları,
- dataset loader,
- loss fonksiyonları,
- config dosyaları,
- utility/helper dosyaları,
- checkpoint kodları,
- augmentation/preprocessing kodları.

## 4.2 Modeli tespit et

Şunları kesin olarak belirle:

- backbone,
- embedding dimension,
- classifier varsa classifier yapısı,
- pooling yöntemi,
- normalization kullanımı,
- metric learning yöntemi,
- triplet/contrastive/cross-entropy vb. loss kullanımı.

Özellikle repository'nin gerçekten **ResNet50-IBN + Triplet Loss** kullanıp kullanmadığını doğrula. Varsayım yapma.

## 4.3 Dataset pipeline'ını tespit et

VeRi-776 için:

- train split,
- query split,
- gallery split,
- ID bilgisi,
- camera bilgisi,
- image preprocessing,
- image resolution

nasıl kullanılıyor belirle.

## 4.4 Evaluation pipeline'ını tespit et

Özellikle:

- mAP nasıl hesaplanıyor?
- Rank-1 nasıl hesaplanıyor?
- Rank-5 hesaplanıyor mu?
- same-camera görüntüler nasıl ele alınıyor?
- distance metric nedir? (cosine / Euclidean vb.)
- query/gallery filtering nasıl yapılıyor?

belirle.

---

# 5. Modernizasyon / Optimizasyon

Repository yaklaşık 5 yıllık olduğu için modernize edilmelidir.

Ancak burada amaç yeni bir algoritma geliştirmek değil, **mevcut kodu güncel ortamda güvenilir, tekrar üretilebilir ve daha temiz hale getirmektir.**

## 5.1 Dependency modernizasyonu

Kontrol et:

- Python sürümü,
- PyTorch,
- torchvision,
- CUDA,
- NumPy,
- Pillow,
- OpenCV,
- diğer bütün bağımlılıklar.

Deprecated API'leri güncelle.

Bir bağımlılığın güncellenmesi algoritmik davranışı değiştirecekse önce bunun etkisini değerlendir.

## 5.2 Path/config temizliği

Hard-coded path kullanımlarını azalt.

Örneğin:

```text
/data/user/project/dataset
```

gibi path'ler doğrudan source code içine yazılmamalıdır.

Configuration üzerinden yönetilebilir hale getir.

## 5.3 Config sistemi

En azından aşağıdaki parametreler merkezi bir config üzerinden yönetilebilmelidir:

- dataset path,
- image size,
- batch size,
- learning rate,
- epochs,
- optimizer,
- scheduler,
- loss parameters,
- embedding dimension,
- checkpoint path,
- random seed,
- device.

Örneğin:

```text
configs/
    baseline.yaml
    low_light.yaml
```

veya eşdeğer sade bir yapı kullanılabilir.

## 5.4 Logging

Training sırasında aşağıdakiler kaydedilmelidir:

- epoch,
- training loss,
- learning rate,
- validation/evaluation sonuçları,
- checkpoint bilgisi.

## 5.5 Reproducibility

Her deneyin yeniden çalıştırılabilmesi gerekir.

Her deney için mümkün olduğunca şu bilgiler kaydedilmelidir:

- random seed,
- config,
- checkpoint,
- dataset split,
- metric sonuçları.

---

# 6. Baseline Sistem

İlk hedef mevcut sistemi **çalışır hale getirmek ve baseline sonuçlarını yeniden üretmektir.**

Beklenen temel pipeline:

```text
Image
  ↓
Backbone
  ↓
Feature extraction
  ↓
Embedding
  ↓
Distance / similarity
  ↓
Ranked gallery
  ↓
Evaluation
```

Eğer repository doğrulandıktan sonra aşağıdaki yaklaşımı kullanıyorsa baseline olarak korunmalıdır:

```text
VeRi-776
   ↓
ResNet50-IBN
   ↓
Triplet Loss
   ↓
Vehicle Embedding
   ↓
Retrieval
```

Ancak repository'nin gerçek yapısı neyse dokümanda o belirtilmelidir. **Varsayılan olarak ResNet50-IBN veya Triplet Loss ekleme. Önce mevcut kodu doğrula.**

---

# 7. Baseline Sonucunun Yeniden Üretilmesi

Daha önce elde edilmiş bir sonuç bulunuyorsa bunun aynen kabul edilmesi yerine modernize edilmiş kodla mümkün olduğunca yeniden üretilmesi gerekir.

Önceki çalışmada örnek olarak:

```text
Rank-1 ≈ 90.70%
mAP ≈ 63.56%
```

gibi sonuçlar elde edilmiş olabilir.

Bu değerler yalnızca geçmiş deney bilgisi olarak kabul edilmelidir.

**Yeni çalışmanın resmi baseline sonucu, çalıştırılan deneyden elde edilmelidir.**

Baseline doğrulanmadan zorlu koşul deneylerine geçme.

---

# 8. Zorlu Koşullar Araştırma Aşaması

Projenin asıl akademik araştırma kısmı buradadır.


## 8.0 Seçilecek Zorlu Koşullar

İlk etapta **en fazla 2–3 koşul** seçilmelidir. Önerilen öncelik:

1. **Düşük ışık / gece**
2. **Yağmur**
3. **Görüntü gürültüsü veya kalite bozulması**

Sis, kar, güçlü glare, motion blur, occlusion vb. koşullar ancak temel deneyler tamamlandıktan sonra ve zaman/kaynak uygunsa ele alınmalıdır. Projenin amacı tüm zorlu koşulları çözmek değildir.

Her koşul ayrı bir deney senaryosu olarak ele alınmalı ve aynı baseline ile karşılaştırılmalıdır. Böylece hangi koşulun performansı ne kadar etkilediği ve hangi iyileştirmenin hangi koşulda işe yaradığı görülebilir.

## 8.1 Önce performans kaybını ölç

Modeli zorlu koşul görüntüleri üzerinde çalıştırmadan önce normal koşullardaki performansı kaydet.

Ardından seçilen her zorlu koşul için kontrollü bir test senaryosu oluştur ve aynı modelin performansını ölç.

Amaç:

```text
Normal-light
     ↓
Baseline performance

Challenging-condition
     ↓
Baseline performance
```

arasındaki farkı ölçmektir.



---

# 9. Zorlu Koşul Veri Senaryoları

Öncelikle VeRi-776 dataset'inin gerçek görüntülerinde seçilen zorlu koşullara ilişkin yeterli örnek bulunup bulunmadığını araştır. Özellikle düşük ışık/gece ve yağmur koşulları için gerçek örneklerin kapsamını doğrula.

Dataset'in metadata / kamera / zaman bilgilerinin uygunluğu kontrol edilmelidir.

## Gerçek zorlu-koşul verisi yeterliyse

Mümkün olduğunca gerçek görüntüler kullanılmalıdır.

## Yeterli gerçek zorlu-koşul verisi yoksa

Kontrollü sentetik degradation uygulanabilir.

Örneğin:

- brightness azaltma,
- contrast azaltma,
- gamma değişimi,
- noise ekleme.

Ancak dönüşümler kontrollü ve tekrar üretilebilir olmalıdır.

Örneğin üç sabit seviye kullanılabilir:

```text
Challenging-condition Level 1 = mild
Challenging-condition Level 2 = medium
Challenging-condition Level 3 = severe
```

Parametreler deney konfigürasyonunda açıkça tutulmalıdır.

---

# 10. Zorlu Koşullara Karşı İyileştirme Deneyleri

İlk aşamada basit, açıklanabilir ve düşük maliyetli yöntemlerle başlanmalıdır.

Önerilen deneyler:

```text
Baseline
Baseline + Gamma Correction
Baseline + CLAHE
Baseline + Selected Enhancement / Robustness Method
```

İlk iki yöntem tamamlanmadan karmaşık deep-learning enhancement modellerine geçme.

## Gamma Correction

Challenging-condition görüntünün parlaklık dağılımını değiştirmek için kullanılabilir.

Amaç:

```text
Challenging-condition image
    ↓
Gamma correction
    ↓
Re-ID model
```

## CLAHE

Kontrastı iyileştirmek için uygulanabilir.

```text
Challenging-condition image
    ↓
CLAHE
    ↓
Re-ID model
```

## Gelişmiş enhancement

Zaman ve kaynak uygunsa bir deep-learning tabanlı challenging-condition enhancement yöntemi eklenebilir.

Ancak bu proje için zorunlu değildir.

---

# 11. Enhancement'ın Nerede Yapılacağı

Başlangıçta iki senaryo birbirinden ayrılmalıdır.

## Senaryo A — Preprocessing

```text
Image
 ↓
Challenging-condition enhancement
 ↓
Re-ID model
 ↓
Embedding
```

## Senaryo B — Model içi yaklaşım

Örneğin attention veya feature-level yöntemler daha sonra denenebilir.

Ancak bunu ilk sürümde zorunlu hale getirme.

Önce preprocessing tabanlı deneylerle temel etkiyi ölç.

---

# 12. Deney Tasarımı

Tüm deneyler aynı evaluation protocol ile yapılmalıdır.

Örnek sonuç tablosu:

| Method | Condition | mAP | Rank-1 | Rank-5 |
|---|---|---:|---:|---:|
| Baseline | Normal | ... | ... | ... |
| Baseline | Challenging-condition | ... | ... | ... |
| + Gamma | Challenging-condition | ... | ... | ... |
| + CLAHE | Challenging-condition | ... | ... | ... |
| + Enhancement | Challenging-condition | ... | ... | ... |

---

# 13. Deneylerin Adil Olması

Karşılaştırılan yöntemler mümkün olduğunca aynı koşullarda değerlendirilmelidir.

Aynı tutulması gerekenler:

- dataset split,
- query/gallery protocol,
- evaluation code,
- image resolution,
- backbone,
- training schedule (değiştirilmiyorsa),
- random seed (mümkün olduğunda),
- metric hesaplama yöntemi.

Bir deneyde birden fazla şey aynı anda değiştirilmemelidir.

Örneğin:

```text
Baseline:
ResNet50-IBN + Triplet

Experiment:
ResNet50-IBN + Triplet + CLAHE
```

Burada yalnızca CLAHE değişkeni olmalıdır.

---

# 14. Ablation Study

Zaman yeterliyse sınırlı bir ablation study yapılabilir.

Örneğin:

```text
A: Baseline
B: Baseline + Gamma
C: Baseline + CLAHE
D: Baseline + Gamma + CLAHE
```

Ancak gereksiz miktarda kombinasyon üretme.

Amaç deney sayısını büyütmek değil, sonucu açıklayabilmektir.

---

# 15. Görsel Sonuçlar

Sadece metric vermek yerine bazı retrieval sonuçları görselleştirilebilir.

Örneğin:

```text
Query
  ↓
Top-5 retrieved images
```

Normal ve seçilen zorlu koşullarında örnekler gösterilebilir.

Özellikle yanlış retrieval örnekleri değerlidir.

Örneğin:

```text
Query vehicle

Top-1 → wrong vehicle
Top-2 → same vehicle
Top-3 → same vehicle
...
```

Bu örnekler modelin seçilen zorlu koşullarda neden zorlandığını tartışmak için kullanılabilir.

---

# 16. Başarı Kriterleri

Minimum başarılı proje aşağıdaki pipeline'ın eksiksiz çalışmasıdır:

```text
VeRi-776
   ↓
Existing Re-ID codebase
   ↓
Modernized & working code
   ↓
Baseline reproduction
   ↓
Normal-light metrics
   ↓
Challenging-condition benchmark
   ↓
Challenging-condition baseline metrics
   ↓
At least 2 enhancement methods
   ↓
Quantitative comparison
   ↓
Analysis
```

İdeal durumda ayrıca:

```text
Baseline
vs
Challenging-condition
vs
Gamma
vs
CLAHE
vs
Advanced Enhancement
```

karşılaştırması bulunur.

---

# 17. Basit Demo

Araştırmayı göstermek için basit bir inference demo hazırlanabilir.

Kullanıcı bir araç resmi verdiğinde:

```text
Input image
    ↓
Re-ID model
    ↓
Embedding
    ↓
Gallery comparison
    ↓
Top-K similar vehicles
```

sonucu gösterilmelidir.

Demo'nun amacı production uygulaması geliştirmek değildir.

---

# 18. Tez İçin Önerilen Bölüm Yapısı

## 1. Introduction

- Vehicle Re-ID problemi
- Problem önemi
- Challenging-condition problemi
- Projenin amacı
- Katkılar

## 2. Related Work

- Vehicle Re-ID
- Metric learning
- Triplet Loss / Contrastive Loss
- Attention / feature learning
- Challenging-condition enhancement

## 3. Dataset and Baseline

- VeRi-776
- Dataset protocol
- Baseline model
- Training setup
- Evaluation metrics

## 4. Proposed Experimental Approach

- Challenging-condition senaryosu
- Enhancement yöntemleri
- Deney tasarımı

## 5. Experimental Results

- Baseline results
- Challenging-condition degradation
- Enhancement comparison
- Ablation
- Qualitative retrieval examples

## 6. Discussion

- Hangi yöntem daha iyi?
- Neden?
- Hangi durumlarda başarısız?
- Hesaplama maliyeti / performans dengesi

## 7. Conclusion

- Araştırma sorusunun cevabı
- Ana bulgular
- Gelecek çalışma önerileri

---

# 19. AI Agent Çalışma Protokolü

AI agent aşağıdaki sırayı takip etmelidir.

## Phase 1 — Understand

Repository'yi tamamen veya yeterli kapsamda incele.

Önce dosya yapısını, model pipeline'ını, dataset loader'ı ve evaluation kodunu çıkar.

Bu aşamada büyük değişiklik yapma.

## Phase 2 — Make It Run

Mevcut sistemi güncel ortamda çalıştır.

Hata çıkarsa:

1. Hatanın nedenini belirle.
2. Deprecated API / dependency kaynaklıysa modernize et.
3. Algoritmanın davranışını mümkün olduğunca koru.
4. Tek değişkenli küçük değişiklikler yap.
5. Değişiklik sonrası tekrar çalıştır.

## Phase 3 — Validate Baseline

Modelin şu zinciri gerçekten tamamladığını doğrula:

```text
input image
→ model
→ embedding
→ gallery distance
→ ranking
→ metric
```

Baseline sonucu tekrar üret.

## Phase 4 — Refactor

Baseline doğrulandıktan sonra:

- config ayır,
- logging ekle,
- path'leri düzelt,
- dependency'leri temizle,
- reproducibility sağla.

## Phase 5 — Challenging-Condition Benchmark

Challenging-condition test senaryosunu ekle.

Baseline'ın performans düşüşünü ölç.

## Phase 6 — Experiments

Gamma, CLAHE ve gerekiyorsa daha ileri enhancement yöntemlerini tek tek ekle.

## Phase 7 — Analysis

Sonuçları tablo/grafik olarak kaydet.

Sadece en iyi sonucu değil, tüm önemli sonuçları sakla.

## Phase 8 — Demo

En sonunda basit inference/retrieval demo oluştur.

---

# 20. AI Agent'ın Kesinlikle Yapmaması Gerekenler

- Çalışan repository'yi sebepsiz yere tamamen yeniden yazmak.
- Modernizasyon gerekçesi olmadan framework değiştirmek.
- Training pipeline'ı açıklamadan değiştirmek.
- Sonuçları uydurmak.
- Çalıştırılmamış mAP / Rank-1 değerleri yazmak.
- Dataset split'ini deneyler arasında değiştirmek.
- Evaluation protokolünü yöntemler arasında değiştirmek.
- Aynı anda model + preprocessing + loss + training schedule değiştirerek sonuç kıyaslamak.
- Challenging-condition enhancement yönteminin mutlaka faydalı olacağını varsaymak.
- Scope'u detection + tracking + re-identification + mobile uygulama gibi birçok probleme genişletmek.
- Baseline doğrulanmadan yeni özellik eklemek.

---

# 21. Önceliklendirme

Öncelik sırası kesin olarak şöyledir:

```text
1. Repository'yi anla
2. Repository'yi çalıştır
3. Eski dependency/API sorunlarını düzelt
4. Baseline'ı doğrula
5. Evaluation'ı doğrula
6. Challenging-condition benchmark oluştur
7. Challenging-condition baseline ölç
8. Gamma / CLAHE deneyleri yap
9. Sonuçları karşılaştır
10. Tez için deney kayıtlarını düzenle
11. Basit demo yap
12. Vakit kalırsa gelişmiş yöntemler ekle
```

---

# 22. Son Hedef

Projenin sonunda şu soruya sayısal ve deneysel bir cevap verilmelidir:

> **Challenging-condition koşullarında Vehicle Re-ID ne kadar performans kaybediyor ve hangi yöntem bu kaybı en iyi şekilde azaltıyor?**

Proje yalnızca:

> "ResNet50-IBN kullandım."

şeklinde bir uygulama olmamalıdır.

Bunun yerine:

> **mevcut bir Vehicle Re-ID sisteminin baseline olarak kurulması → seçilen zorlu koşullarda test edilmesi → uygun iyileştirme/robustness yöntemlerinin kontrollü olarak uygulanması → metriklerle karşılaştırılması → sonuçların yorumlanması**

şeklinde deneysel bir çalışma olmalıdır.

---

# 23. Repository Geldiğinde Yapılacak İlk İş

GitHub repository sağlandığında **önce kodu analiz et ve mevcut mimariyi belgeleyen kısa bir teknik rapor çıkar.**

Bu rapor en azından şunları içermeli:

```text
- Repository yapısı
- Entry point
- Backbone
- Loss
- Embedding dimension
- Dataset loader
- Augmentation
- Optimizer
- Scheduler
- Batch construction
- Evaluation protocol
- Metrics
- Checkpoint system
- Dependency versions
- CUDA/PyTorch compatibility
- Eski/deprecated kodlar
- Modernizasyon önerileri
```

Bu analiz tamamlanmadan büyük refactor veya yeni model ekleme.

# 5.5 Yerel Eğitim ve Donanım Kısıtı

Model eğitimleri **yerel bilgisayarda** gerçekleştirilecektir. Proje tasarlanırken cloud GPU veya sürekli erişilebilir uzak sunucu varmış gibi varsayım yapılmamalıdır.

## AI Agent'ın dikkat etmesi gerekenler

- Eğitim pipeline'ı tek bir yerel GPU ile çalışabilecek şekilde tasarlanmalıdır.
- CPU ile inference mümkün olmalıdır; ancak eğitim için mevcut GPU tercih edilmelidir.
- GPU belleği bilinmiyorsa gereksiz büyük batch size, model veya input resolution seçilmemelidir.
- Batch size, image resolution ve num_workers gibi değerler configuration üzerinden değiştirilebilir olmalıdır.
- CUDA kullanılamıyorsa sistem anlaşılır bir hata vermeli veya CPU moduna kontrollü şekilde geçebilmelidir.
- Mixed precision (AMP) destekleniyorsa eğitim belleğini ve süresini azaltmak için tercih edilmelidir; ancak önce mevcut kodun doğruluğu korunmalıdır.
- Gradient accumulation ancak gerçekten gerekli olduğunda kullanılmalıdır.
- Gereksiz yere çok büyük backbone veya embedding dimension kullanılmamalıdır.
- Deneylerin tekrarlanabilir olması için her deneyin eğitim süresi ve kullanılan donanım bilgisi kaydedilmelidir.

## Öncelik sırası

Yerel eğitim nedeniyle optimizasyon yapılırken öncelik şu sırada olmalıdır:

```text
Doğruluk
  ↓
Tekrarlanabilirlik
  ↓
GPU memory kullanımı
  ↓
Eğitim süresi
  ↓
Kod sadeliği
```

Performans uğruna algoritmanın davranışını değiştiren agresif optimizasyonlar yapılmamalıdır.

## Deney planlama

İlk aşamada küçük bir smoke test çalıştırılmalıdır:

```text
1-2 epoch
↓
küçük subset / kontrollü batch
↓
forward
↓
backward
↓
checkpoint
↓
evaluation
```

Bu test başarıyla tamamlanmadan uzun eğitim başlatılmamalıdır.

Daha sonra gerçek baseline eğitimi çalıştırılmalıdır.

### Eğitim kaynaklarının kaydedilmesi

Her önemli deney için mümkün olduğunca şu bilgiler kaydedilmelidir:

- GPU modeli,
- VRAM miktarı,
- PyTorch version,
- CUDA version,
- batch size,
- image resolution,
- epoch sayısı,
- training duration,
- peak GPU memory kullanımı,
- random seed.

Örneğin sonuç kaydı:

```text
experiment: baseline
model: ResNet50-IBN
loss: Triplet Loss
batch_size: 32
image_size: 256x128
epochs: 50
device: cuda:0
mAP: ...
Rank-1: ...
Rank-5: ...
training_time: ...
```

Değerler gerçekten çalıştırılmadan doldurulmamalıdır.

---

# 5.6 Yerel Ortama Uygun Kod Optimizasyonu

Kod modernize edilirken özellikle aşağıdaki alanlar kontrol edilmelidir:

- DataLoader'ın gereksiz yere yavaş olup olmadığı,
- `num_workers` değerinin uygunluğu,
- `pin_memory` kullanımı,
- batch transferlerinin GPU'ya verimli yapılıp yapılmadığı,
- AMP desteği,
- gereksiz CPU-GPU tensor kopyaları,
- evaluation sırasında `torch.no_grad()` / inference mode kullanımı,
- checkpoint boyutları,
- gereksiz şekilde her epoch checkpoint alınması,
- dataset'in her epoch gereksiz yere yeniden hazırlanması.

Ancak optimizasyon yapılırken önce **baseline'ın orijinal mantığının korunduğu doğrulanmalıdır.**

---

# 5.7 GPU Belleği Yetersiz Kalırsa İzlenecek Yol

GPU memory yetmezse doğrudan modeli değiştirme.

Sırasıyla:

```text
Batch size azalt
      ↓
AMP / mixed precision kullan
      ↓
Image resolution'ı kontrollü azalt
      ↓
Gradient accumulation değerlendir
      ↓
DataLoader / memory kullanımını optimize et
      ↓
Son çare olarak model boyutunu değerlendir
```

Bu değişikliklerin her biri deney sonuçlarına etkisiyle birlikte kaydedilmelidir.

---

# 18. Yerel Eğitim İçin Nihai Teknik Hedef

Projenin ana pipeline'ı, kullanıcının kendi bilgisayarında aşağıdaki şekilde çalışabilmelidir:

```text
Local machine
     ↓
CUDA GPU
     ↓
VeRi-776
     ↓
Training
     ↓
Checkpoint
     ↓
Evaluation
     ↓
mAP / Rank-1 / Rank-5
```

Cloud veya harici eğitim servisi temel proje gereksinimi değildir.

AI Agent, yerel donanım kaynaklarını aşacak eğitim konfigürasyonları önermemeli; önce mevcut GPU/VRAM durumunu tespit edip konfigürasyonu buna göre ayarlamalıdır.
