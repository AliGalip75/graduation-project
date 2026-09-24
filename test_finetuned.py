"""
Phase 6 — Finetuned Model Evaluation (Robustness & Catastrophic Forgetting)
==========================================================================
Bu script, hem Base hem de Finetuned modelleri (ResNet + Swin):
1. Clean (temiz) veri setinde test ederek 'Catastrophic Forgetting' oranını ölçer.
2. Zorlu koşul (degraded) veri setlerinde test ederek iyileşme (Improvement) miktarını ölçer.

Çıktı:
  - Konsolda özet tablo
  - results/finetuned_benchmark_results.csv (detaylı sonuçlar)
  - results/finetuned_benchmark_results.md  (tez için markdown tablo)
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
    "ResNet-Base": {
        "config": os.path.join(SCRIPT_DIR, "configs", "triplet_baseline.yaml"),
        "checkpoint": os.path.join(SCRIPT_DIR, "model", "triplet_baseline", "net_69.pth"),
    },
    "ResNet-Fine": {
        "config": os.path.join(SCRIPT_DIR, "configs", "triplet_finetune.yaml"),
        "checkpoint": os.path.join(SCRIPT_DIR, "model", "triplet_finetune", "net_19.pth"),
    },
    "Swin-Base": {
        "config": os.path.join(SCRIPT_DIR, "configs", "swin_triplet_baseline.yaml"),
        "checkpoint": os.path.join(SCRIPT_DIR, "model", "swin_triplet_baseline", "net_69.pth"),
    },
    "Swin-Fine": {
        "config": os.path.join(SCRIPT_DIR, "configs", "swin_triplet_finetune.yaml"),
        "checkpoint": os.path.join(SCRIPT_DIR, "model", "swin_triplet_finetune", "net_19.pth"),
    },
}

def parse_eval_output(output_text):
    for line in output_text.split("\n"):
        if "Rank@1" in line and "mAP" in line:
            parts = line.strip().split()
            metrics = {}
            for p in parts:
                if ":" in p:
                    key, val = p.split(":")
                    metrics[key] = float(val) * 100
            return {
                "rank1": round(metrics.get("Rank@1", 0), 2),
                "rank5": round(metrics.get("Rank@5", 0), 2),
                "rank10": round(metrics.get("Rank@10", 0), 2),
                "mAP": round(metrics.get("mAP", 0), 2),
            }
    return None

def run_test(config_path, checkpoint_path, query_csv, gallery_csv, data_dir="datasets"):
    import yaml
    import tempfile

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config["query_csv_path"] = query_csv
    config["gallery_csv_path"] = gallery_csv
    config["data_dir"] = data_dir

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
    parser = argparse.ArgumentParser(description="Phase 6: Finetuned Model Evaluation")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--conditions", nargs="+", default=["clean", "darkness", "noise", "rain"])
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 6: Finetuned Model Benchmark")
    print("=" * 70)

    results = []
    
    # Store clean results of base models to calculate drops/improvements later if needed,
    # though we can just output raw numbers. Let's record everything.
    
    for model_name, model_info in MODELS.items():
        for condition in args.conditions:
            levels_to_run = [0] if condition == "clean" else args.levels
            for level in levels_to_run:
                print(f"\nEvaluating {model_name} | {condition} Level {level}")
                print("-" * 50)
                
                if condition == "clean":
                    query_csv = os.path.join(args.data_dir, "annot", "query_full.csv")
                    gallery_csv = os.path.join(args.data_dir, "annot", "gallery_full.csv")
                else:
                    query_csv = os.path.join(args.data_dir, "annot", f"query_{condition}_level{level}.csv")
                    gallery_csv = os.path.join(args.data_dir, "annot", f"gallery_{condition}_level{level}.csv")

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
                    results.append({
                        "model": model_name,
                        "condition": condition,
                        "level": level,
                        "rank1": metrics["rank1"],
                        "rank5": metrics["rank5"],
                        "rank10": metrics["rank10"],
                        "mAP": metrics["mAP"]
                    })
                    print(f"  Rank@1: {metrics['rank1']:.2f}%")
                    print(f"  mAP:    {metrics['mAP']:.2f}%")
                    print(f"  Süre:   {elapsed:.0f}s")
                else:
                    print(f"  [HATA] Test başarısız!")

    os.makedirs("results", exist_ok=True)

    # 1) CSV
    import csv
    csv_path = os.path.join("results", "finetuned_benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "condition", "level", "rank1", "rank5", "rank10", "mAP"
        ])
        writer.writeheader()
        writer.writerows(results)

    # 2) JSON
    json_path = os.path.join("results", "finetuned_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # 3) Markdown Tablosu
    md_path = os.path.join("results", "finetuned_benchmark_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 6: Finetuned Modellerin Değerlendirme Sonuçları\n\n")
        f.write("Bu tabloda Base ve Finetuned modellerin temiz (clean) ve zorlu (degraded) koşullardaki metrikleri karşılaştırılmaktadır.\n\n")
        f.write("| Model | Koşul | Seviye | Rank@1 | Rank@5 | Rank@10 | mAP |\n")
        f.write("|-------|-------|--------|-------:|-------:|--------:|----:|\n")
        for r in results:
            lvl = f"Level {r['level']}" if r['level'] > 0 else "—"
            cond = r['condition'] if r['condition'] != "clean" else "Normal (Clean)"
            f.write(
                f"| {r['model']} | {cond} | {lvl} | "
                f"{r['rank1']:.2f}% | {r['rank5']:.2f}% | {r['rank10']:.2f}% | "
                f"{r['mAP']:.2f}% |\n"
            )

    print(f"\n{'=' * 70}")
    print("SONUÇ TABLOSU")
    print(f"{'=' * 70}")
    print(f"{'Model':<15} {'Koşul':<12} {'Seviye':<8} {'Rank@1':>8} {'mAP':>8}")
    print("-" * 55)
    for r in results:
        lvl = f"L{r['level']}" if r['level'] > 0 else "Base"
        print(f"{r['model']:<15} {r['condition']:<12} {lvl:<8} "
              f"{r['rank1']:>7.2f}% {r['mAP']:>7.2f}%")

    print(f"\nSonuçlar kaydedildi:")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")

if __name__ == "__main__":
    main()
