import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ayarlar
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'savefig.dpi': 300})
os.makedirs("results/plots", exist_ok=True)

# Veriyi oku
with open("results/finetuned_benchmark_results.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Modellerin isimleri
model_order = ["ResNet-Base", "ResNet-Fine", "Swin-Base", "Swin-Fine"]

# Renk, Şekil ve Çizgi Stili Paleti (Hepsini belirginleştirdik)
# Base modeller açık renk, Fine-tune modeller koyu renk
palette = {"ResNet-Base": "#f87171", "ResNet-Fine": "#dc2626", "Swin-Base": "#60a5fa", "Swin-Fine": "#2563eb"}

# Hepsi için farklı işaretçiler (marker)
markers = {"ResNet-Base": "o", "ResNet-Fine": "s", "Swin-Base": "^", "Swin-Fine": "D"}

# Hepsi için farklı çizgi stilleri
line_styles = {"ResNet-Base": "--", "ResNet-Fine": "-", "Swin-Base": "-.", "Swin-Fine": "-"}

# İngilizce terimleri Türkçeye çevirmek için sözlük
condition_names = {"rain": "Yağmur", "noise": "Gürültü", "darkness": "Karanlık/Gece"}

# 1. Temiz Veri Seti Karşılaştırması (Bar Chart)
clean_df = df[df["condition"] == "clean"].copy()
if not clean_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.barplot(data=clean_df, x="model", y="mAP", order=model_order, hue="model", palette=palette, ax=axes[0], legend=False)
    axes[0].set_title("Temiz Veri Setinde mAP Başarımı")
    axes[0].set_ylabel("mAP (%)")
    axes[0].set_xlabel("Modeller")
    axes[0].set_ylim(0, 100)
    for i in axes[0].containers: axes[0].bar_label(i, fmt='%.1f')

    sns.barplot(data=clean_df, x="model", y="rank1", order=model_order, hue="model", palette=palette, ax=axes[1], legend=False)
    axes[1].set_title("Temiz Veri Setinde Rank-1 Başarımı")
    axes[1].set_ylabel("Rank-1 (%)")
    axes[1].set_xlabel("Modeller")
    axes[1].set_ylim(0, 100)
    for i in axes[1].containers: axes[1].bar_label(i, fmt='%.1f')

    plt.tight_layout()
    plt.savefig("results/plots/1_temiz_veri_basarimi.png")
    plt.close()


# 2. Zorluk Seviyelerine Göre Düşüş Grafikleri (Line Charts)
conditions = ["rain", "noise", "darkness"]
for cond in conditions:
    cond_df = df[(df["condition"] == cond) | (df["condition"] == "clean")].copy()
    
    if cond_df.empty: continue
        
    plt.figure(figsize=(8, 6))
    for model in model_order:
        model_data = cond_df[cond_df["model"] == model].sort_values("level")
        plt.plot(model_data["level"], model_data["mAP"], 
                 marker=markers[model], 
                 linestyle=line_styles[model], 
                 color=palette[model], 
                 linewidth=2.5, 
                 markersize=9,
                 label=model)

    tr_cond = condition_names[cond]
    plt.title(f"{tr_cond} Şartlarında mAP Düşüşü")
    plt.xlabel(f"{tr_cond} Şiddet Seviyesi (0 = Temiz Görüntü)")
    plt.ylabel("mAP (%)")
    plt.xticks([0, 1, 2, 3])
    plt.ylim(0, 100)
    plt.legend(title="Yapay Zeka Modelleri")
    plt.tight_layout()
    plt.savefig(f"results/plots/2_{cond}_dususu_map.png")
    plt.close()


# 3. Genel Dayanıklılık (Zorlu Şartların Ortalaması)
degraded_df = df[df["level"] > 0]
if not degraded_df.empty:
    avg_df = degraded_df.groupby("model")[["mAP", "rank1"]].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.barplot(data=avg_df, x="model", y="mAP", order=model_order, hue="model", palette=palette, ax=axes[0], legend=False)
    axes[0].set_title("Tüm Zorlu Şartlarda Ortalama mAP")
    axes[0].set_ylabel("Ortalama mAP (%)")
    axes[0].set_xlabel("Modeller")
    axes[0].set_ylim(0, 100)
    for i in axes[0].containers: axes[0].bar_label(i, fmt='%.1f')

    sns.barplot(data=avg_df, x="model", y="rank1", order=model_order, hue="model", palette=palette, ax=axes[1], legend=False)
    axes[1].set_title("Tüm Zorlu Şartlarda Ortalama Rank-1")
    axes[1].set_ylabel("Ortalama Rank-1 (%)")
    axes[1].set_xlabel("Modeller")
    axes[1].set_ylim(0, 100)
    for i in axes[1].containers: axes[1].bar_label(i, fmt='%.1f')

    plt.tight_layout()
    plt.savefig("results/plots/3_genel_dayaniklilik.png")
    plt.close()

print("Bütün Türkçe tez grafikleri 'results/plots' klasörüne kaydedildi!")
