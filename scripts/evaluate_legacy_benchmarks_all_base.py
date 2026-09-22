"""
Comprehensive Benchmark Evaluation: Base V1 vs Base V2 vs Base V3
Evaluates all three generations on:
  1. reparos-user-centric-v2 (88 queries)
  2. reparos-diagnostic-10k (10,000 queries)
  3. reparos-compositional-4k (4,000 queries)
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

# Ensure utf-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarking.io import read_jsonl, write_jsonl, normalize_text
from benchmarking.metrics import score_rows

BENCHMARKS = {
    "user_centric_v2": REPO_ROOT / "benchmark/reparos-user-centric-v2/gold.jsonl",
    "diagnostic_10k": REPO_ROOT / "benchmark/reparos-diagnostic-10k/gold.jsonl",
    "compositional_4k": REPO_ROOT / "benchmark/reparos-compositional-4k/gold.jsonl",
}

MODELS = {
    "Base V1": {
        "path": REPO_ROOT / "artifacts/opennmt-production-kaggle-t4-v1-ctranslate2-float32",
        "description": "Base V1 (Production Kaggle T4, 1-layer Transformer, 8k vocab)",
    },
    "Base V2": {
        "path": REPO_ROOT / "artifacts/reparos-base-v2-32-final-1/ctranslate2",
        "description": "Base V2 (Curriculum V2 32k, 1-layer Transformer, 8k vocab)",
    },
    "Base V3": {
        "path": REPO_ROOT / "artifacts/checkpoints/base_v3_production/ctranslate2_export",
        "description": "Base V3 Production (2-layer Enc / 1-layer Dec, 2048 FF, 12k vocab)",
    },
}

# Add Base V3.1 Fine-tuned if available
ft_path = REPO_ROOT / "artifacts/checkpoints/base_v3_1_finetune/ctranslate2_export"
if ft_path.exists():
    MODELS["Base V3.1 FT"] = {
        "path": ft_path,
        "description": "Base V3.1 Production Fine-tuned (1M pairs)",
    }

OUT_DIR = REPO_ROOT / "artifacts/evaluation_three_generations"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_batch_inference(
    model_path: Path,
    gold_rows: List[Dict[str, Any]],
    batch_size: int = 128,
    beam_size: int = 10,
    num_hypotheses: int = 10,
) -> tuple[List[Dict[str, Any]], float]:
    """Runs batch inference using CTranslate2 and SentencePiece."""
    sp = spm.SentencePieceProcessor(model_file=str(model_path / "tokenizer.model"))
    translator = ctranslate2.Translator(
        str(model_path), device="cpu", compute_type="default", intra_threads=4
    )

    predictions = []
    total_start = time.perf_counter()

    for i in range(0, len(gold_rows), batch_size):
        batch = gold_rows[i : i + batch_size]
        # Lowercase and normalize query as required by the pipeline
        norm_inputs = [unicodedata.normalize("NFC", str(row["input"]).strip().lower()) for row in batch]
        tokenized = [sp.encode(text, out_type=str) for text in norm_inputs]

        t0 = time.perf_counter()
        results = translator.translate_batch(
            tokenized,
            beam_size=beam_size,
            num_hypotheses=num_hypotheses,
            return_scores=True,
            max_decoding_length=100,
        )
        batch_latency = (time.perf_counter() - t0) * 1000.0
        avg_latency = batch_latency / max(len(batch), 1)

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
                "latency_ms": avg_latency,
            })

    total_time = time.perf_counter() - total_start
    return predictions, total_time


def compute_case_insensitive_exact(joined: List[tuple[Dict, Dict]]) -> Dict[str, float]:
    """Computes case-insensitive and case-folded exact accuracy & recalls."""
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
    print("ReparoS Three Generations Benchmark: Base V1 vs Base V2 vs Base V3")
    print("=" * 80)

    all_results = {}

    for bench_key, bench_path in BENCHMARKS.items():
        print(f"\n" + "#" * 80)
        print(f"EVALUATING BENCHMARK: {bench_key.upper()} ({bench_path.name})")
        print("#" * 80)

        gold_rows = read_jsonl(bench_path)
        print(f"Loaded {len(gold_rows)} test cases.")

        bench_results = {}

        for model_name, model_info in MODELS.items():
            print(f"\n--- Running {model_name} on {bench_key} ---")
            m_path = model_info["path"]
            if not m_path.exists():
                print(f"ERROR: Model path {m_path} does not exist!")
                continue

            preds, elapsed = run_batch_inference(m_path, gold_rows)
            pred_file = OUT_DIR / f"{bench_key}_{model_name.lower().replace(' ', '_')}_preds.jsonl"
            write_jsonl(pred_file, preds)

            joined = [(g, p) for g, p in zip(gold_rows, preds)]
            metrics = score_rows(joined, include_breakdown=True)
            ci_metrics = compute_case_insensitive_exact(joined)

            # Performance stats
            throughput = len(gold_rows) / max(elapsed, 0.001)
            latency_per_query = (elapsed / max(len(gold_rows), 1)) * 1000.0

            bench_results[model_name] = {
                "metrics": metrics,
                "ci_metrics": ci_metrics,
                "elapsed_sec": elapsed,
                "throughput_qps": throughput,
                "latency_ms": latency_per_query,
            }

            print(f"Completed in {elapsed:.2f}s ({throughput:.1f} qps, {latency_per_query:.2f} ms/query)")
            print(f"  Exact Match (R@1):       {metrics['ranking']['recall@1']*100:.2f}% (Case-Insensitive: {ci_metrics['ci_recall@1']*100:.2f}%)")
            print(f"  Recall@5:                {metrics['ranking']['recall@5']*100:.2f}% (Case-Insensitive: {ci_metrics['ci_recall@5']*100:.2f}%)")
            print(f"  Recall@10:               {metrics['ranking']['recall@10']*100:.2f}% (Case-Insensitive: {ci_metrics['ci_recall@10']*100:.2f}%)")
            print(f"  Autocorrect F1:          {metrics['correction']['autocorrect_f1']*100:.2f}%")
            print(f"  Clean Preservation Rate: {metrics['safety']['clean_preservation_rate']*100:.2f}%")
            print(f"  Regression Rate:         {metrics['safety']['regression_rate']*100:.2f}%")
            print(f"  CER Net Reduction:       {metrics['distance']['cer_net_reduction']*100:.2f}%")

        all_results[bench_key] = bench_results

    # Save complete results
    out_json = OUT_DIR / "three_generations_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nAll benchmark results saved to {out_json}")

    # Generate Summary Markdown
    print("\n" + "=" * 80)
    print("OVERALL COMPARATIVE BENCHMARK SUMMARY")
    print("=" * 80)

    for bench_key in BENCHMARKS:
        print(f"\n### Benchmark: {bench_key.upper()}")
        b_res = all_results[bench_key]
        has_ft = "Base V3.1 FT" in b_res
        if has_ft:
            header = "| Metric | Base V1 | Base V2 | Base V3 | Base V3.1 FT | V3.1 vs V3 | V3.1 vs V1 |"
            sep = "|---|---:|---:|---:|---:|---:|---:|"
        else:
            header = "| Metric | Base V1 | Base V2 | Base V3 | V3 vs V1 | V3 vs V2 |"
            sep = "|---|---:|---:|---:|---:|---:|"
        print(header)
        print(sep)

        v1_m = b_res.get("Base V1", {}).get("metrics", {})
        v2_m = b_res.get("Base V2", {}).get("metrics", {})
        v3_m = b_res.get("Base V3", {}).get("metrics", {})
        ft_m = b_res.get("Base V3.1 FT", {}).get("metrics", {}) if has_ft else {}

        v1_ci = b_res.get("Base V1", {}).get("ci_metrics", {})
        v2_ci = b_res.get("Base V2", {}).get("ci_metrics", {})
        v3_ci = b_res.get("Base V3", {}).get("ci_metrics", {})
        ft_ci = b_res.get("Base V3.1 FT", {}).get("ci_metrics", {}) if has_ft else {}

        def fmt_row(title, val1, val2, val3, val_ft=None, is_pct=True):
            if has_ft and val_ft is not None:
                d_v3 = f"{val_ft - val3:+.2f}%" if (val3 is not None and val_ft is not None) else "N/A"
                d_v1 = f"{val_ft - val1:+.2f}%" if (val1 is not None and val_ft is not None) else "N/A"
                if not is_pct:
                    d_v3 = f"{val_ft - val3:+.4f}" if (val3 is not None and val_ft is not None) else "N/A"
                    d_v1 = f"{val_ft - val1:+.4f}" if (val1 is not None and val_ft is not None) else "N/A"
                s1 = f"{val1:.2f}%" if is_pct else f"{val1:.4f}"
                s2 = f"{val2:.2f}%" if is_pct else f"{val2:.4f}"
                s3 = f"{val3:.2f}%" if is_pct else f"{val3:.4f}"
                s_ft = f"{val_ft:.2f}%" if is_pct else f"{val_ft:.4f}"
                return f"| {title} | {s1} | {s2} | {s3} | **{s_ft}** | {d_v3} | {d_v1} |"
            else:
                d1 = f"{val3 - val1:+.2f}%" if (val1 is not None and val3 is not None) else "N/A"
                d2 = f"{val3 - val2:+.2f}%" if (val2 is not None and val3 is not None) else "N/A"
                s1 = f"{val1:.2f}%" if val1 is not None else "N/A"
                s2 = f"{val2:.2f}%" if val2 is not None else "N/A"
                s3 = f"{val3:.2f}%" if val3 is not None else "N/A"
                return f"| {title} | {s1} | {s2} | {s3} | {d1} | {d2} |"

        print(fmt_row("Exact Match (Recall@1)", v1_m['ranking']['recall@1']*100, v2_m['ranking']['recall@1']*100, v3_m['ranking']['recall@1']*100, ft_m.get('ranking', {}).get('recall@1', 0)*100 if has_ft else None))
        print(fmt_row("Case-Insensitive R@1", v1_ci['ci_recall@1']*100, v2_ci['ci_recall@1']*100, v3_ci['ci_recall@1']*100, ft_ci.get('ci_recall@1', 0)*100 if has_ft else None))
        print(fmt_row("Recall@5", v1_m['ranking']['recall@5']*100, v2_m['ranking']['recall@5']*100, v3_m['ranking']['recall@5']*100, ft_m.get('ranking', {}).get('recall@5', 0)*100 if has_ft else None))
        print(fmt_row("Case-Insensitive R@5", v1_ci['ci_recall@5']*100, v2_ci['ci_recall@5']*100, v3_ci['ci_recall@5']*100, ft_ci.get('ci_recall@5', 0)*100 if has_ft else None))
        print(fmt_row("Recall@10", v1_m['ranking']['recall@10']*100, v2_m['ranking']['recall@10']*100, v3_m['ranking']['recall@10']*100, ft_m.get('ranking', {}).get('recall@10', 0)*100 if has_ft else None))
        print(fmt_row("Case-Insensitive R@10", v1_ci['ci_recall@10']*100, v2_ci['ci_recall@10']*100, v3_ci['ci_recall@10']*100, ft_ci.get('ci_recall@10', 0)*100 if has_ft else None))
        print(fmt_row("Autocorrect F1", v1_m['correction']['autocorrect_f1']*100, v2_m['correction']['autocorrect_f1']*100, v3_m['correction']['autocorrect_f1']*100, ft_m.get('correction', {}).get('autocorrect_f1', 0)*100 if has_ft else None))
        print(fmt_row("Exact Noisy Accuracy", v1_m['correction']['exact_noisy_accuracy']*100, v2_m['correction']['exact_noisy_accuracy']*100, v3_m['correction']['exact_noisy_accuracy']*100, ft_m.get('correction', {}).get('exact_noisy_accuracy', 0)*100 if has_ft else None))
        print(fmt_row("Clean Preservation", v1_m['safety']['clean_preservation_rate']*100, v2_m['safety']['clean_preservation_rate']*100, v3_m['safety']['clean_preservation_rate']*100, ft_m.get('safety', {}).get('clean_preservation_rate', 0)*100 if has_ft else None))
        print(fmt_row("Regression Rate", v1_m['safety']['regression_rate']*100, v2_m['safety']['regression_rate']*100, v3_m['safety']['regression_rate']*100, ft_m.get('safety', {}).get('regression_rate', 0)*100 if has_ft else None))
        print(fmt_row("CER Net Reduction", v1_m['distance']['cer_net_reduction']*100, v2_m['distance']['cer_net_reduction']*100, v3_m['distance']['cer_net_reduction']*100, ft_m.get('distance', {}).get('cer_net_reduction', 0)*100 if has_ft else None))
        print(fmt_row("Latency (ms/query)", b_res['Base V1']['latency_ms'], b_res['Base V2']['latency_ms'], b_res['Base V3']['latency_ms'], b_res.get('Base V3.1 FT', {}).get('latency_ms') if has_ft else None, is_pct=False))


if __name__ == "__main__":
    main()
