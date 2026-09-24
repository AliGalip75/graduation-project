"""
Phase 5 — Zorlu Koşul Benchmark Test Otomasyonu
================================================
Bu script, generate_degraded_dataset.py ile üretilen tüm sentetik
veri setlerinde her iki modeli (ResNet + Swin) çalıştırır ve
sonuçları tek bir tabloda toplar.

Kullanım:
  python test_challenging.py [--conditions darkness noise rain] [--levels 1 2 3]

Çıktı:
  - Konsolda özet tablo
  - results/challenging_benchmark_results.csv (detaylı sonuçlar)
  - results/challenging_benchmark_results.md  (tez için markdown tablo)
"""

import os
import sys
import subprocess
import argparse
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────
# Model Konfigürasyonları
# ─────────────────────────────────────────────────────────────────────

MODELS = {
    "ResNet-IBN": {
        "config": os.path.join(SCRIPT_DIR, "configs", "triplet_baseline.yaml"),
        "checkpoint": os.path.join(SCRIPT_DIR, "model", "triplet_baseline", "net_69.pth"),
    },
    "Swin-T": {
        "config": os.path.join(SCRIPT_DIR, "configs", "swin_triplet_baseline.yaml"),
        "checkpoint": os.path.join(SCRIPT_DIR, "model", "swin_triplet_baseline", "net_69.pth"),
    },
}

# Baseline (temiz veri) sonuçları — daha önce ölçüldü
BASELINE_RESULTS = {
    "ResNet-IBN": {"rank1": 94.64, "rank5": 97.55, "rank10": 98.63, "mAP": 74.71},
    "Swin-T":     {"rank1": 94.87, "rank5": 97.62, "rank10": 98.39, "mAP": 75.30},
}


def parse_eval_output(output_text):
    """
    evaluate.py çıktısından metrikleri parse eder.
    Örnek satır: Rank@1:0.948749 Rank@5:0.976162 Rank@10:0.983909 mAP:0.753023
    """
    for line in output_text.split("\n"):
        if "Rank@1" in line and "mAP" in line:
            parts = line.strip().split()
            metrics = {}
            for p in parts:
                if ":" in p:
                    key, val = p.split(":")
                    metrics[key] = float(val) * 100  # yüzdeye çevir
            return {
                "rank1": round(metrics.get("Rank@1", 0), 2),
                "rank5": round(metrics.get("Rank@5", 0), 2),
                "rank10": round(metrics.get("Rank@10", 0), 2),
                "mAP": round(metrics.get("mAP", 0), 2),
            }
    return None


def run_test(config_path, checkpoint_path, query_csv, gallery_csv, data_dir="datasets"):
    """
    test.py'yi verilen parametrelerle çalıştırır.
    Config dosyasını geçici olarak düzenlemek yerine, test.py'nin
    beklediği CSV yollarını config'e dinamik olarak enjekte ederiz.
    """
    import yaml
    import tempfile

    # Config'i oku ve CSV yollarını değiştir
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config["query_csv_path"] = query_csv
    config["gallery_csv_path"] = gallery_csv
    config["data_dir"] = data_dir

    # Geçici config dosyası oluştur
    tmp_config = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False, dir=SCRIPT_DIR
    )
    yaml.dump(config, tmp_config)
    tmp_config.close()

    try:
        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "test.py"),
            "--config", tmp_config.name,
            "--checkpoint", checkpoint_path,
            "--eval_gpu"
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=SCRIPT_DIR
        )

        output = result.stdout + result.stderr
        metrics = parse_eval_output(output)

        if metrics is None:
            print(f"  [UYARI] Metrik parse edilemedi. Çıktı:")
            print(output[-500:] if len(output) > 500 else output)

        return metrics
    finally:
        os.unlink(tmp_config.name)


def main():
    parser = argparse.ArgumentParser(
        description="Zorlu koşul benchmark test otomasyonu (Phase 5)")
    parser.add_argument("--data_dir", type=str, default="datasets",
                        help="Veri seti kök dizini")
    parser.add_argument("--conditions", nargs="+",
                        default=["darkness", "noise", "rain"],
                        choices=["darkness", "noise", "rain"])
    parser.add_argument("--levels", nargs="+", type=int,
                        default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--models", nargs="+",
                        default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Test edilecek modeller")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 5: Zorlu Koşul Benchmark Testi")
    print("=" * 70)
    print(f"Modeller  : {args.models}")
    print(f"Koşullar  : {args.conditions}")
    print(f"Seviyeler : {args.levels}")
    print()

    results = []

    # Baseline sonuçlarını ekle
    for model_name in args.models:
        baseline = BASELINE_RESULTS[model_name]
        results.append({
            "model": model_name,
            "condition": "clean",
            "level": 0,
            "rank1": baseline["rank1"],
            "rank5": baseline["rank5"],
            "rank10": baseline["rank10"],
            "mAP": baseline["mAP"],
            "rank1_drop": 0.0,
            "mAP_drop": 0.0,
        })

    total_tests = len(args.models) * len(args.conditions) * len(args.levels)
    test_num = 0

    for model_name in args.models:
        model_info = MODELS[model_name]
        baseline = BASELINE_RESULTS[model_name]

        for condition in args.conditions:
            for level in args.levels:
                test_num += 1
                print(f"\n[{test_num}/{total_tests}] {model_name} | {condition} Level {level}")
                print("-" * 50)

                query_csv = os.path.join(
                    args.data_dir, "annot", f"query_{condition}_level{level}.csv")
                gallery_csv = os.path.join(
                    args.data_dir, "annot", f"gallery_{condition}_level{level}.csv")

                # CSV'lerin varlığını kontrol et
                if not os.path.exists(query_csv) or not os.path.exists(gallery_csv):
                    print(f"  [ATLANDI] CSV bulunamadı: {query_csv}")
                    continue

                start = time.time()
                metrics = run_test(
                    model_info["config"],
                    model_info["checkpoint"],
                    query_csv, gallery_csv,
                    args.data_dir
                )
                elapsed = time.time() - start

                if metrics:
                    rank1_drop = baseline["rank1"] - metrics["rank1"]
                    mAP_drop = baseline["mAP"] - metrics["mAP"]

                    results.append({
                        "model": model_name,
                        "condition": condition,
                        "level": level,
                        "rank1": metrics["rank1"],
                        "rank5": metrics["rank5"],
                        "rank10": metrics["rank10"],
                        "mAP": metrics["mAP"],
                        "rank1_drop": round(rank1_drop, 2),
                        "mAP_drop": round(mAP_drop, 2),
                    })
                    print(f"  Rank@1: {metrics['rank1']:.2f}% (↓{rank1_drop:.2f})")
                    print(f"  mAP:    {metrics['mAP']:.2f}% (↓{mAP_drop:.2f})")
                    print(f"  Süre:   {elapsed:.0f}s")
                else:
                    print(f"  [HATA] Test başarısız!")

    # ─────────────────────────────────────────────────────────
    # Sonuçları kaydet
    # ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)

    # 1) CSV
    import csv
    csv_path = os.path.join("results", "challenging_benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "condition", "level", "rank1", "rank5", "rank10", "mAP",
            "rank1_drop", "mAP_drop"
        ])
        writer.writeheader()
        writer.writerows(results)

    # 2) JSON
    json_path = os.path.join("results", "challenging_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # 3) Markdown tablo (tez için)
    md_path = os.path.join("results", "challenging_benchmark_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 5: Zorlu Koşul Benchmark Sonuçları\n\n")
        f.write("| Model | Koşul | Seviye | Rank@1 | Rank@5 | Rank@10 | mAP | Rank@1 Düşüş | mAP Düşüş |\n")
        f.write("|-------|-------|--------|-------:|-------:|--------:|----:|---------:|---------:|\n")
        for r in results:
            lvl = f"Level {r['level']}" if r['level'] > 0 else "—"
            cond = r['condition'] if r['condition'] != "clean" else "Normal (Baseline)"
            f.write(
                f"| {r['model']} | {cond} | {lvl} | "
                f"{r['rank1']:.2f}% | {r['rank5']:.2f}% | {r['rank10']:.2f}% | "
                f"{r['mAP']:.2f}% | {r['rank1_drop']:+.2f}% | {r['mAP_drop']:+.2f}% |\n"
            )

    print(f"\n{'=' * 70}")
    print("SONUÇ TABLOSU")
    print(f"{'=' * 70}")
    print(f"{'Model':<12} {'Koşul':<18} {'Seviye':<8} {'Rank@1':>8} {'mAP':>8} {'R1↓':>8} {'mAP↓':>8}")
    print("-" * 82)
    for r in results:
        lvl = f"L{r['level']}" if r['level'] > 0 else "Base"
        print(f"{r['model']:<12} {r['condition']:<18} {lvl:<8} "
              f"{r['rank1']:>7.2f}% {r['mAP']:>7.2f}% "
              f"{r['rank1_drop']:>+7.2f}% {r['mAP_drop']:>+7.2f}%")

    print(f"\nSonuçlar kaydedildi:")
    print(f"  CSV:      {csv_path}")
    print(f"  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
