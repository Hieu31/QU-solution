"""
Evaluate Base V3.1 Fine-tuned Model on All 3 Legacy Benchmarks:
  1. reparos-user-centric-v2 (88 queries)
  2. reparos-diagnostic-10k (10,000 queries)
  3. reparos-compositional-4k (4,000 queries)
Compares directly against Base V1, Base V2, and Base V3.
"""

import os
import sys
import json
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Any
import ctranslate2
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path("/kaggle/working/QU-solution")
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarking.io import read_jsonl, write_jsonl, normalize_text
from benchmarking.metrics import score_rows

BENCHMARKS = {
    "user_centric_v2": REPO_ROOT / "benchmark/reparos-user-centric-v2/gold.jsonl",
    "diagnostic_10k": REPO_ROOT / "benchmark/reparos-diagnostic-10k/gold.jsonl",
    "compositional_4k": REPO_ROOT / "benchmark/reparos-compositional-4k/gold.jsonl",
}

FT_MODEL_PATH = REPO_ROOT / "artifacts/checkpoints/base_v3_1_finetune/ctranslate2_export"

# Baseline reference benchmarks from 3-generation evaluation
BASELINES = {
    "user_centric_v2": {
        "Base V1": {"r1": 32.95, "ci_r1": 42.05, "r5": 38.64, "r10": 43.18, "ci_r10": 54.55, "f1": 38.16, "reg": 15.91, "cer": 29.71},
        "Base V2": {"r1": 47.73, "ci_r1": 55.68, "r5": 60.23, "r10": 60.23, "ci_r10": 71.59, "f1": 54.19, "reg": 15.91, "cer": 48.91},
        "Base V3": {"r1": 50.00, "ci_r1": 57.95, "r5": 56.82, "r10": 59.09, "ci_r10": 71.59, "f1": 56.41, "reg": 12.50, "cer": 49.55},
    },
    "compositional_4k": {
        "Base V1": {"r1": 43.18, "r5": 61.55, "r10": 67.05, "f1": 44.47, "reg": 11.88, "cer": 39.67},
        "Base V2": {"r1": 64.05, "r5": 82.53, "r10": 86.30, "f1": 64.92, "reg": 4.32, "cer": 78.20},
        "Base V3": {"r1": 62.95, "r5": 82.45, "r10": 86.45, "f1": 63.46, "reg": 4.58, "cer": 78.56},
    },
    "diagnostic_10k": {
        "Base V1": {"r1": 77.66, "r5": 87.95, "r10": 90.16, "f1": 77.51, "clean": 90.37, "reg": 6.98, "cer": 85.02, "addr_abbr": 91.00, "addr_sym": 36.12, "vni": 88.50},
        "Base V2": {"r1": 70.40, "r5": 81.10, "r10": 83.06, "f1": 70.72, "clean": 91.15, "reg": 7.67, "cer": 70.09, "addr_abbr": 18.88, "addr_sym": 27.12, "vni": 89.12},
        "Base V3": {"r1": 74.41, "r5": 84.57, "r10": 87.62, "f1": 74.35, "clean": 90.23, "reg": 7.15, "cer": 83.17, "addr_abbr": 73.38, "addr_sym": 18.12, "vni": 93.38},
    }
}


def run_batch_inference(model_path: Path, gold_rows: List[Dict[str, Any]], batch_size: int = 128):
    sp = spm.SentencePieceProcessor(model_file=str(model_path / "tokenizer.model"))
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    try:
        translator = ctranslate2.Translator(str(model_path), device=device, compute_type="default", intra_threads=4)
    except Exception as e:
        print(f"  [Warning] Failed to initialize on {device} ({e}), falling back to cpu...")
        translator = ctranslate2.Translator(str(model_path), device="cpu", compute_type="default", intra_threads=4)

    predictions = []
    t_start = time.perf_counter()

    for i in range(0, len(gold_rows), batch_size):
        batch = gold_rows[i : i + batch_size]
        norm_inputs = [unicodedata.normalize("NFC", str(row["input"]).strip().lower()) for row in batch]
        tokenized = [sp.encode(text, out_type=str) for text in norm_inputs]

        results = translator.translate_batch(
            tokenized, beam_size=10, num_hypotheses=10, return_scores=True, max_decoding_length=100
        )

        for row, norm_in, res in zip(batch, norm_inputs, results):
            hyps = [normalize_text(sp.decode(toks)) for toks in res.hypotheses]
            if not hyps:
                hyps = [norm_in]
            predictions.append({
                "query_id": row["query_id"],
                "error_type": row.get("error_type", "unknown"),
                "input": row["input"],
                "expected": row["expected"],
                "output": hyps[0],
                "hypotheses": hyps,
                "scores": [float(s) for s in res.scores],
            })

    elapsed = time.perf_counter() - t_start
    return predictions, elapsed


def compute_case_insensitive_exact(joined: List[tuple[Dict, Dict]]) -> Dict[str, float]:
    total = len(joined)
    r1 = r3 = r5 = r10 = 0
    for gold, pred in joined:
        target = normalize_text(str(gold["expected"])).lower()
        hyps = [normalize_text(str(h)).lower() for h in pred.get("hypotheses", [pred["output"]])]
        if hyps and hyps[0] == target:
            r1 += 1
        if target in hyps[:3]:
            r3 += 1
        if target in hyps[:5]:
            r5 += 1
        if target in hyps[:10]:
            r10 += 1
    return {
        "ci_recall@1": r1 / max(total, 1),
        "ci_recall@3": r3 / max(total, 1),
        "ci_recall@5": r5 / max(total, 1),
        "ci_recall@10": r10 / max(total, 1),
    }


def main():
    print("=" * 80)
    print("REPAROS BASE V3.1 FINE-TUNED BENCHMARK EVALUATION")
    print(f"Model under test: {FT_MODEL_PATH}")
    print("=" * 80)

    if not FT_MODEL_PATH.exists():
        print(f"ERROR: Model path {FT_MODEL_PATH} not found!")
        sys.exit(1)

    ft_results = {}

    for bench_key, bench_path in BENCHMARKS.items():
        print(f"\nEvaluating on {bench_key.upper()} ({bench_path.name})...")
        gold_rows = read_jsonl(bench_path)

        preds, elapsed = run_batch_inference(FT_MODEL_PATH, gold_rows)
        joined = [(g, p) for g, p in zip(gold_rows, preds)]
        metrics = score_rows(joined, include_breakdown=True)
        ci_metrics = compute_case_insensitive_exact(joined)

        ft_results[bench_key] = {
            "metrics": metrics,
            "ci_metrics": ci_metrics,
            "latency_ms": (elapsed / len(gold_rows)) * 1000.0,
        }
        print(f"  Completed in {elapsed:.2f}s ({len(gold_rows)/elapsed:.1f} qps, {elapsed/len(gold_rows)*1000:.2f} ms/query)")

    # -------------------------------------------------------------
    # Print Comparative Tables
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARATIVE BENCHMARK REPORT: BASE V3.1 FT vs PREVIOUS GENERATIONS")
    print("=" * 80)

    # 1. USER_CENTRIC_V2
    print("\n### 1. Benchmark: USER-CENTRIC-V2 (88 queries thực tế)")
    u_m = ft_results["user_centric_v2"]["metrics"]
    u_ci = ft_results["user_centric_v2"]["ci_metrics"]
    b_u = BASELINES["user_centric_v2"]

    print("| Chỉ số (Metric) | Base V1 | Base V2 | Base V3 | Base V3.1 FT | V3.1 vs V3 | V3.1 vs V1 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for title, key, is_ci in [
        ("Exact Match (Recall@1)", "r1", False),
        ("Case-Insensitive R@1", "ci_r1", True),
        ("Recall@5", "r5", False),
        ("Recall@10", "r10", False),
        ("Case-Insensitive R@10", "ci_r10", True),
        ("Autocorrect F1", "f1", False),
        ("Regression Rate", "reg", False),
        ("CER Net Reduction", "cer", False),
    ]:
        v1 = b_u["Base V1"][key]
        v2 = b_u["Base V2"][key]
        v3 = b_u["Base V3"][key]
        if is_ci:
            ft_val = u_ci[f"ci_recall@{key.split('_r')[1]}"] * 100
        elif key == "r1":
            ft_val = u_m["ranking"]["recall@1"] * 100
        elif key == "r5":
            ft_val = u_m["ranking"]["recall@5"] * 100
        elif key == "r10":
            ft_val = u_m["ranking"]["recall@10"] * 100
        elif key == "f1":
            ft_val = u_m["correction"]["autocorrect_f1"] * 100
        elif key == "reg":
            ft_val = u_m["safety"]["regression_rate"] * 100
        elif key == "cer":
            ft_val = u_m["distance"]["cer_net_reduction"] * 100

        d_v3 = f"{ft_val - v3:+.2f}%"
        d_v1 = f"{ft_val - v1:+.2f}%"
        print(f"| {title} | {v1:.2f}% | {v2:.2f}% | {v3:.2f}% | **{ft_val:.2f}%** | {d_v3} | {d_v1} |")

    # 2. COMPOSITIONAL_4K
    print("\n### 2. Benchmark: COMPOSITIONAL-4K (4,000 queries lỗi tổ hợp đa tầng)")
    c_m = ft_results["compositional_4k"]["metrics"]
    b_c = BASELINES["compositional_4k"]

    print("| Chỉ số (Metric) | Base V1 | Base V2 | Base V3 | Base V3.1 FT | V3.1 vs V3 | V3.1 vs V1 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for title, key in [
        ("Exact Match (Recall@1)", "r1"),
        ("Recall@5", "r5"),
        ("Recall@10", "r10"),
        ("Autocorrect F1", "f1"),
        ("Regression Rate", "reg"),
        ("CER Net Reduction", "cer"),
    ]:
        v1 = b_c["Base V1"][key]
        v2 = b_c["Base V2"][key]
        v3 = b_c["Base V3"][key]
        if key == "r1":
            ft_val = c_m["ranking"]["recall@1"] * 100
        elif key == "r5":
            ft_val = c_m["ranking"]["recall@5"] * 100
        elif key == "r10":
            ft_val = c_m["ranking"]["recall@10"] * 100
        elif key == "f1":
            ft_val = c_m["correction"]["autocorrect_f1"] * 100
        elif key == "reg":
            ft_val = c_m["safety"]["regression_rate"] * 100
        elif key == "cer":
            ft_val = c_m["distance"]["cer_net_reduction"] * 100

        d_v3 = f"{ft_val - v3:+.2f}%"
        d_v1 = f"{ft_val - v1:+.2f}%"
        print(f"| {title} | {v1:.2f}% | {v2:.2f}% | {v3:.2f}% | **{ft_val:.2f}%** | {d_v3} | {d_v1} |")

    # 3. DIAGNOSTIC_10K
    print("\n### 3. Benchmark: DIAGNOSTIC-10K (10,000 queries chuyên sâu)")
    d_m = ft_results["diagnostic_10k"]["metrics"]
    b_d = BASELINES["diagnostic_10k"]

    print("| Chỉ số (Metric) | Base V1 | Base V2 | Base V3 | Base V3.1 FT | V3.1 vs V3 | V3.1 vs V1 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for title, key in [
        ("Exact Match (Recall@1)", "r1"),
        ("Recall@5", "r5"),
        ("Recall@10", "r10"),
        ("Autocorrect F1", "f1"),
        ("Clean Preservation", "clean"),
        ("Regression Rate", "reg"),
        ("CER Net Reduction", "cer"),
    ]:
        v1 = b_d["Base V1"][key]
        v2 = b_d["Base V2"][key]
        v3 = b_d["Base V3"][key]
        if key == "r1":
            ft_val = d_m["ranking"]["recall@1"] * 100
        elif key == "r5":
            ft_val = d_m["ranking"]["recall@5"] * 100
        elif key == "r10":
            ft_val = d_m["ranking"]["recall@10"] * 100
        elif key == "f1":
            ft_val = d_m["correction"]["autocorrect_f1"] * 100
        elif key == "clean":
            ft_val = d_m["safety"]["clean_preservation_rate"] * 100
        elif key == "reg":
            ft_val = d_m["safety"]["regression_rate"] * 100
        elif key == "cer":
            ft_val = d_m["distance"]["cer_net_reduction"] * 100

        d_v3 = f"{ft_val - v3:+.2f}%"
        d_v1 = f"{ft_val - v1:+.2f}%"
        print(f"| {title} | {v1:.2f}% | {v2:.2f}% | {v3:.2f}% | **{ft_val:.2f}%** | {d_v3} | {d_v1} |")

    # 4. ERROR BREAKDOWN ON DIAGNOSTIC_10K
    print("\n### 4. Bóc tách Các Nhóm Lỗi Trọng Tâm trên DIAGNOSTIC-10K (Top-1 Accuracy)")
    d_types = d_m.get("by_error_type", {})
    print("| Họ lỗi (Error Type) | Số mẫu | Base V1 | Base V2 | Base V3 | Base V3.1 FT | V3.1 vs V3 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    
    key_types = [
        ("address_symbol (Số nhà xuyệt /)", "address_symbol", 36.12, 27.12, 18.12),
        ("address_abbreviation (Viết tắt)", "address_abbreviation", 91.00, 18.88, 73.38),
        ("vni_leak (Gõ nhầm VNI)", "vni_leak", 88.50, 89.12, 93.38),
        ("telex_leak (Gõ nhầm Telex)", "telex_leak", 93.25, 89.75, 92.38),
        ("missing_diacritics_full (Không dấu)", "missing_diacritics_full", 82.50, 75.50, 80.00),
        ("word_boundary (Dính/tách từ)", "word_boundary", 79.88, 80.50, 80.38),
        ("clean (Bảo vệ câu đúng)", "clean", 90.64, 91.36, 90.36),
    ]
    for label, etype, v1_v, v2_v, v3_v in key_types:
        if etype in d_types:
            c = d_types[etype]["queries"]
            ft_acc = d_types[etype]["ranking"]["recall@1"] * 100
            d_v3 = f"{ft_acc - v3_v:+.2f}%"
            print(f"| **{label}** | {c} | {v1_v:.2f}% | {v2_v:.2f}% | {v3_v:.2f}% | **{ft_acc:.2f}%** | {d_v3} |")

    # Save to report json
    out_file = REPO_ROOT / "artifacts/evaluation_three_generations/base_v3_1_finetune_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ft_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nDetailed report saved to {out_file}!")


if __name__ == "__main__":
    main()
