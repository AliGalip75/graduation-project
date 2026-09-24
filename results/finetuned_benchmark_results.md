# Phase 6: Finetuned Modellerin Değerlendirme Sonuçları

Bu tabloda Base ve Finetuned modellerin temiz (clean) ve zorlu (degraded) koşullardaki metrikleri karşılaştırılmaktadır.

| Model | Koşul | Seviye | Rank@1 | Rank@5 | Rank@10 | mAP |
|-------|-------|--------|-------:|-------:|--------:|----:|
| ResNet-Base | Normal (Clean) | — | 94.64% | 97.56% | 98.63% | 74.72% |
| ResNet-Base | darkness | Level 1 | 92.19% | 96.36% | 98.03% | 68.68% |
| ResNet-Base | darkness | Level 2 | 82.30% | 90.76% | 93.80% | 50.71% |
| ResNet-Base | darkness | Level 3 | 71.63% | 82.54% | 87.19% | 36.12% |
| ResNet-Base | noise | Level 1 | 91.60% | 96.19% | 98.15% | 65.96% |
| ResNet-Base | noise | Level 2 | 83.79% | 92.61% | 95.23% | 47.43% |
| ResNet-Base | noise | Level 3 | 71.33% | 84.98% | 89.33% | 27.66% |
| ResNet-Base | rain | Level 1 | 86.35% | 93.62% | 96.19% | 54.29% |
| ResNet-Base | rain | Level 2 | 57.39% | 73.30% | 79.86% | 18.04% |
| ResNet-Base | rain | Level 3 | 22.53% | 39.93% | 48.81% | 5.03% |
| ResNet-Fine | Normal (Clean) | — | 95.05% | 97.56% | 98.51% | 74.71% |
| ResNet-Fine | darkness | Level 1 | 94.04% | 97.80% | 98.51% | 73.44% |
| ResNet-Fine | darkness | Level 2 | 90.41% | 95.41% | 97.50% | 68.11% |
| ResNet-Fine | darkness | Level 3 | 85.22% | 92.73% | 95.35% | 58.94% |
| ResNet-Fine | noise | Level 1 | 94.52% | 97.50% | 98.57% | 72.79% |
| ResNet-Fine | noise | Level 2 | 92.25% | 97.02% | 98.57% | 68.84% |
| ResNet-Fine | noise | Level 3 | 89.27% | 95.59% | 97.38% | 62.40% |
| ResNet-Fine | rain | Level 1 | 93.15% | 96.66% | 98.51% | 70.76% |
| ResNet-Fine | rain | Level 2 | 89.21% | 95.95% | 97.32% | 60.60% |
| ResNet-Fine | rain | Level 3 | 76.70% | 88.86% | 93.21% | 41.57% |
| Swin-Base | Normal (Clean) | — | 94.87% | 97.62% | 98.39% | 75.30% |
| Swin-Base | darkness | Level 1 | 92.01% | 96.54% | 98.03% | 70.58% |
| Swin-Base | darkness | Level 2 | 84.03% | 91.54% | 93.98% | 55.31% |
| Swin-Base | darkness | Level 3 | 69.96% | 82.78% | 87.54% | 39.32% |
| Swin-Base | noise | Level 1 | 93.86% | 97.14% | 98.33% | 71.30% |
| Swin-Base | noise | Level 2 | 89.39% | 95.59% | 97.32% | 63.28% |
| Swin-Base | noise | Level 3 | 83.49% | 92.07% | 94.87% | 52.18% |
| Swin-Base | rain | Level 1 | 91.24% | 96.36% | 98.15% | 66.32% |
| Swin-Base | rain | Level 2 | 73.78% | 87.07% | 92.67% | 38.33% |
| Swin-Base | rain | Level 3 | 23.12% | 39.87% | 47.74% | 5.99% |
| Swin-Fine | Normal (Clean) | — | 94.99% | 97.85% | 98.57% | 76.31% |
| Swin-Fine | darkness | Level 1 | 94.52% | 97.26% | 98.39% | 75.37% |
| Swin-Fine | darkness | Level 2 | 92.43% | 96.36% | 97.62% | 70.55% |
| Swin-Fine | darkness | Level 3 | 86.83% | 93.62% | 96.07% | 62.50% |
| Swin-Fine | noise | Level 1 | 94.40% | 97.02% | 98.57% | 74.02% |
| Swin-Fine | noise | Level 2 | 92.19% | 96.60% | 98.21% | 69.74% |
| Swin-Fine | noise | Level 3 | 89.21% | 95.05% | 97.62% | 63.65% |
| Swin-Fine | rain | Level 1 | 93.80% | 97.32% | 98.21% | 72.69% |
| Swin-Fine | rain | Level 2 | 90.46% | 95.71% | 97.85% | 65.90% |
| Swin-Fine | rain | Level 3 | 78.07% | 91.95% | 94.64% | 49.44% |
