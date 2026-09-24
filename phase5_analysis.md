# Phase 5: Zorlu Koşul Benchmark Analizi

Testler başarıyla tamamlandı. 119.000 sentetik bozulmuş görüntü üretildi ve her iki model (ResNet-IBN ve Swin-T) toplam 18 senaryoda test edildi.

## 📊 Benchmark Sonuçları

| Model | Koşul | Seviye | Rank@1 | mAP | Rank@1 Düşüş | mAP Düşüş |
|-------|-------|--------|-------:|----:|---------:|---------:|
| ResNet-IBN | Normal | Base | 94.64% | 74.71% | - | - |
| Swin-T | Normal | Base | 94.87% | 75.30% | - | - |
| ResNet-IBN | darkness | L1 | 92.19% | 68.68% | ↓ 2.45% | ↓ 6.03% |
| Swin-T | darkness | L1 | 92.01% | 70.58% | ↓ 2.86% | ↓ 4.72% |
| ResNet-IBN | darkness | L2 | 82.30% | 50.71% | ↓ 12.34% | ↓ 24.00% |
| Swin-T | darkness | L2 | 84.03% | 55.31% | ↓ 10.84% | ↓ 19.99% |
| ResNet-IBN | darkness | L3 | 71.63% | 36.12% | ↓ 23.01% | ↓ 38.59% |
| Swin-T | darkness | L3 | 69.96% | 39.32% | ↓ 24.91% | ↓ 35.98% |
| ResNet-IBN | noise | L1 | 91.60% | 65.96% | ↓ 3.04% | ↓ 8.75% |
| Swin-T | noise | L1 | 93.86% | 71.30% | ↓ 1.01% | ↓ 4.00% |
| ResNet-IBN | noise | L2 | 83.79% | 47.43% | ↓ 10.85% | ↓ 27.28% |
| Swin-T | noise | L2 | 89.39% | 63.28% | ↓ 5.48% | ↓ 12.02% |
| ResNet-IBN | noise | L3 | 71.33% | 27.67% | ↓ 23.31% | ↓ 47.04% |
| Swin-T | noise | L3 | 83.49% | 52.18% | ↓ 11.38% | ↓ 23.12% |
| ResNet-IBN | rain | L1 | 86.35% | 54.29% | ↓ 8.29% | ↓ 20.42% |
| Swin-T | rain | L1 | 91.24% | 66.32% | ↓ 3.63% | ↓ 8.98% |
| ResNet-IBN | rain | L2 | 57.39% | 18.04% | ↓ 37.25% | ↓ 56.67% |
| Swin-T | rain | L2 | 73.78% | 38.33% | ↓ 21.09% | ↓ 36.97% |
| ResNet-IBN | rain | L3 | 22.59% | 5.03% | ↓ 72.05% | ↓ 69.68% |
| Swin-T | rain | L3 | 23.12% | 5.99% | ↓ 71.75% | ↓ 69.31% |

---

## 🔍 Temel Çıkarımlar ve Analiz

> [!TIP]
> **CNN vs. Transformer Karşılaştırması**
> Sonuçlar Swin Transformer'ın, özellikle **Gürültü (Noise)** koşulunda ResNet'e göre çok daha dirençli (robust) olduğunu kanıtladı. Noise Level 3'te ResNet'in Rank@1 değeri %23 düşerken, Swin Transformer sadece %11 düşüş yaşadı. Bu da self-attention mekanizmasının lokal piksel bozulmalarına (gürültüye) karşı Convolutional katmanlardan daha dayanıklı olduğuna dair çok güçlü bir tez bulgusudur.

> [!WARNING]
> **Kırılma Eşiği (Breakdown Threshold)**
> Her iki model de **Yağmur (Rain) Level 3** koşulunda işlevsiz hale geldi (Rank@1 ~%23, mAP ~%5). Bu durum sistemimizin "kırılma eşiğini" temsil etmektedir. Şiddetli yağmur, aracın global formunu ve lokal dokusunu tamamen bozduğu için iki mimari de çöküş yaşadı.

## 🚀 Sıradaki Adım: Phase 6 (Fine-Tuning)
Elde edilen bu bozulma verilerini eğitim aşamasına dahil ederek modelleri fine-tune edeceğiz. Beklentimiz:
1. Modellerin zorlu koşullara karşı dayanıklılığının artması.
2. Bu iyileştirme yapılırken normal koşullardaki başarının düşüp düşmediğinin (Catastrophic Forgetting) test edilmesi.
