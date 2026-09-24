"""
Phase 5 — Sentetik Zorlu Koşul Veri Seti Üretici
=================================================
Bu script, VeRi-776 test setindeki (query + gallery) görüntülere
kontrollü sentetik bozulmalar uygulayarak zorlu koşul benchmark
veri setleri oluşturur.

Koşullar:
  - darkness  : Gamma düşürme ile düşük ışık / gece simülasyonu
  - noise     : Gaussian gürültü ile kalite bozulması
  - rain      : Sentetik yağmur çizgileri + parlaklık düşürme

Seviyeler:
  - level1 (mild)   : Hafif bozulma
  - level2 (medium) : Orta bozulma
  - level3 (severe) : Ciddi bozulma

Kullanım:
  python generate_degraded_dataset.py [--conditions darkness noise rain] [--levels 1 2 3]
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time

# ─────────────────────────────────────────────────────────────────────
# Bozulma Parametreleri (tekrar üretilebilir, tezde raporlanacak)
# ─────────────────────────────────────────────────────────────────────

DEGRADATION_PARAMS = {
    "darkness": {
        # Gamma correction — gamma > 1 karartır
        # I_out = I_in ^ gamma  (normalize 0-1 aralığında)
        1: {"gamma": 2.0},      # mild:   hafif karanlık
        2: {"gamma": 3.5},      # medium: belirgin karanlık
        3: {"gamma": 5.0},      # severe: çok karanlık / gece
    },
    "noise": {
        # Gaussian noise — sigma standart sapma (0-255 ölçeğinde)
        1: {"sigma": 15},       # mild:   hafif gürültü
        2: {"sigma": 35},       # medium: belirgin gürültü
        3: {"sigma": 60},       # severe: ağır gürültü
    },
    "rain": {
        # Sentetik yağmur: çizgi sayısı + parlaklık düşürme
        1: {"num_streaks": 300,  "brightness_factor": 0.85, "streak_length": (10, 20), "streak_alpha": 0.3},
        2: {"num_streaks": 700,  "brightness_factor": 0.65, "streak_length": (15, 35), "streak_alpha": 0.5},
        3: {"num_streaks": 1200, "brightness_factor": 0.45, "streak_length": (20, 50), "streak_alpha": 0.7},
    },
}

# ─────────────────────────────────────────────────────────────────────
# Bozulma Fonksiyonları
# ─────────────────────────────────────────────────────────────────────

def apply_darkness(img_array, gamma):
    """Gamma correction ile karanlık simülasyonu."""
    normalized = img_array.astype(np.float32) / 255.0
    darkened = np.power(normalized, gamma)
    return np.clip(darkened * 255, 0, 255).astype(np.uint8)


def apply_noise(img_array, sigma, rng):
    """Gaussian noise ekleme."""
    noise = rng.normal(0, sigma, img_array.shape).astype(np.float32)
    noisy = img_array.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_rain(img_array, num_streaks, brightness_factor, streak_length, streak_alpha, rng):
    """Sentetik yağmur efekti: beyaz/gri çizgiler + parlaklık düşürme."""
    h, w, c = img_array.shape

    # 1) Parlaklık düşür
    result = (img_array.astype(np.float32) * brightness_factor).astype(np.float32)

    # 2) Yağmur çizgileri oluştur (rain streak overlay)
    rain_layer = np.zeros((h, w), dtype=np.float32)
    min_len, max_len = streak_length

    for _ in range(num_streaks):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        length = rng.integers(min_len, max_len + 1)
        # Yağmur hafif açılı düşer (dikey ağırlıklı)
        dx = rng.integers(-2, 3)  # -2..2 arası hafif yatay sapma
        thickness = rng.integers(1, 3)

        for t in range(length):
            py = y + t
            px = x + int(dx * t / max(length, 1))
            if 0 <= py < h and 0 <= px < w:
                # Kalınlık uygula
                for offset in range(thickness):
                    if 0 <= px + offset < w:
                        rain_layer[py, px + offset] = 200 + rng.integers(0, 56)

    # 3) Overlay: rain_layer'ı alpha ile karıştır
    rain_3ch = np.stack([rain_layer] * 3, axis=-1)
    result = result * (1 - streak_alpha * (rain_layer > 0).astype(np.float32)[..., None]) \
             + rain_3ch * streak_alpha

    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# Tek Görüntü İşleme (multiprocessing uyumlu)
# ─────────────────────────────────────────────────────────────────────

def process_single_image(args_tuple):
    """Tek bir görüntüyü boz ve kaydet. Multiprocessing ile kullanılır."""
    src_path, dst_path, condition, level, seed_offset = args_tuple

    params = DEGRADATION_PARAMS[condition][level]
    rng = np.random.default_rng(42 + seed_offset)  # tekrar üretilebilirlik

    try:
        img = Image.open(src_path).convert("RGB")
        img_array = np.array(img)

        if condition == "darkness":
            result = apply_darkness(img_array, params["gamma"])
        elif condition == "noise":
            result = apply_noise(img_array, params["sigma"], rng)
        elif condition == "rain":
            result = apply_rain(
                img_array,
                params["num_streaks"],
                params["brightness_factor"],
                params["streak_length"],
                params["streak_alpha"],
                rng
            )
        else:
            raise ValueError(f"Bilinmeyen koşul: {condition}")

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        Image.fromarray(result).save(dst_path, quality=95)
        return True
    except Exception as e:
        print(f"  [HATA] {src_path}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# Ana İşlem
# ─────────────────────────────────────────────────────────────────────

def generate_degraded_split(data_dir, csv_path, condition, level, output_base, split_name):
    """
    Bir split'in (query veya gallery) tüm görüntülerini bozar.
    Orijinal CSV yapısını koruyarak yeni CSV ve görüntü klasörü oluşturur.

    Örnek çıktı yapısı:
      datasets/VeRi_darkness_level2/image_query/0002_c002_00030600_0.jpg
      datasets/annot/query_darkness_level2.csv
    """
    df = pd.read_csv(csv_path)

    # Çıktı dizini: VeRi_{condition}_level{level}
    degraded_dataset_name = f"VeRi_{condition}_level{level}"
    degraded_dir = os.path.join(output_base, degraded_dataset_name)

    # İşlenecek dosyaları hazırla
    tasks = []
    new_paths = []

    for idx, row in df.iterrows():
        # Orijinal path: VeRi\image_query\xxx.jpg
        original_rel = row["path"]
        # Parçala: ["VeRi", "image_query", "xxx.jpg"]
        parts = original_rel.replace("/", os.sep).replace("\\", os.sep).split(os.sep)

        # Yeni relative path: VeRi_darkness_level2/image_query/xxx.jpg
        new_rel = os.path.join(degraded_dataset_name, *parts[1:])
        new_paths.append(new_rel)

        src_path = os.path.join(data_dir, original_rel)
        dst_path = os.path.join(output_base, new_rel)

        tasks.append((src_path, dst_path, condition, level, idx))

    # Paralel işle
    success_count = 0
    total = len(tasks)

    print(f"  [{split_name}] {total} görüntü işleniyor...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_image, t) for t in tasks]
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                success_count += 1
            if (i + 1) % 2000 == 0 or (i + 1) == total:
                print(f"    İlerleme: {i+1}/{total}")

    # Yeni CSV oluştur (path sütunu güncellenir, diğerleri aynı)
    new_df = df.copy()
    new_df["path"] = new_paths

    annot_dir = os.path.join(output_base, "annot")
    os.makedirs(annot_dir, exist_ok=True)
    out_csv = os.path.join(annot_dir, f"{split_name}_{condition}_level{level}.csv")
    new_df.to_csv(out_csv, index=False)

    print(f"  ✅ {split_name}: {success_count}/{total} başarılı → {out_csv}")
    return out_csv


def main():
    parser = argparse.ArgumentParser(
        description="Sentetik zorlu koşul veri seti üretici (Phase 5)")
    parser.add_argument("--data_dir", type=str, default="datasets",
                        help="Veri seti kök dizini (varsayılan: datasets)")
    parser.add_argument("--conditions", nargs="+",
                        default=["darkness", "noise", "rain"],
                        choices=["darkness", "noise", "rain"],
                        help="Uygulanacak koşullar")
    parser.add_argument("--levels", nargs="+", type=int,
                        default=[1, 2, 3], choices=[1, 2, 3],
                        help="Uygulanacak seviyeler")
    parser.add_argument("--query_csv", type=str,
                        default="datasets/annot/query_full.csv",
                        help="Orijinal query CSV yolu")
    parser.add_argument("--gallery_csv", type=str,
                        default="datasets/annot/gallery_full.csv",
                        help="Orijinal gallery CSV yolu")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5: Sentetik Zorlu Koşul Veri Seti Üretimi")
    print("=" * 60)
    print(f"Koşullar : {args.conditions}")
    print(f"Seviyeler: {args.levels}")
    print()

    # Parametreleri raporla (tez için)
    print("Bozulma Parametreleri:")
    print("-" * 40)
    for cond in args.conditions:
        for lvl in args.levels:
            params = DEGRADATION_PARAMS[cond][lvl]
            print(f"  {cond} Level {lvl}: {params}")
    print()

    start_time = time.time()
    generated_csvs = []

    for condition in args.conditions:
        for level in args.levels:
            print(f"\n{'─' * 50}")
            print(f"▶ {condition.upper()} Level {level}")
            print(f"{'─' * 50}")

            q_csv = generate_degraded_split(
                args.data_dir, args.query_csv,
                condition, level, args.data_dir, "query"
            )
            g_csv = generate_degraded_split(
                args.data_dir, args.gallery_csv,
                condition, level, args.data_dir, "gallery"
            )
            generated_csvs.append({
                "condition": condition,
                "level": level,
                "query_csv": q_csv,
                "gallery_csv": g_csv,
            })

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Tamamlandı! Toplam süre: {elapsed/60:.1f} dakika")
    print(f"Üretilen veri seti sayısı: {len(generated_csvs)}")
    print(f"{'=' * 60}")

    # Özet tablo
    print("\nÜretilen Veri Setleri:")
    print(f"{'Koşul':<12} {'Seviye':<8} {'Query CSV':<45} {'Gallery CSV'}")
    print("-" * 120)
    for entry in generated_csvs:
        print(f"{entry['condition']:<12} Level {entry['level']:<3} "
              f"{entry['query_csv']:<45} {entry['gallery_csv']}")


if __name__ == "__main__":
    main()
