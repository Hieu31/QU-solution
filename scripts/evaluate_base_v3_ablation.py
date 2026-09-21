#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# Ensure src/ is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if (REPO_ROOT / "src").is_dir() and str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import sentencepiece as spm

from reparos.data_registry import HELDOUT_BRANDS
from reparos.serving.ctranslate2 import CTranslate2Predictor, export_opennmt_checkpoint


def normalize(text: str) -> str:
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text).strip().lower())
    return " ".join(norm.split())


def compute_cer(reference: str, hypothesis: str) -> float:
    """Computes character error rate using Levenshtein distance."""
    r = list(reference)
    h = list(hypothesis)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(r)][len(h)] / max(1, len(r))


DIAGNOSTIC_QUERIES = [
    "bv chợ ray",
    "benh vien cho ray",
    "shoppe food",
    "quan an q1",
    "bitis gan day",
    "pizza 4p's",
    "j&t express",
    "co.opmart nguyen kiem",
    "mcdonald's",
    "uong tra sua gong cha",
    "212/22 duong nguyen oanh",
    "d. pasteur q3",
]


class BaseV3Evaluator:
    def __init__(self, model_dir: Path, tokenizer_path: Path, device: str = "cpu"):
        import ctranslate2
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(tokenizer_path))
        self.translator = ctranslate2.Translator(str(model_dir), device=device)

    def predict(self, text: str) -> str:
        res = self.predict_batch([text])
        return res[0] if res else normalize(text)

    def predict_batch(self, texts: List[str], batch_size: int = 128) -> List[str]:
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            tokenized_chunk = [self.sp.encode_as_pieces(t) for t in chunk]
            step_results = self.translator.translate_batch(
                tokenized_chunk,
                beam_size=10,
                num_hypotheses=1,
                max_decoding_length=100,
            )
            for sr in step_results:
                tokens = sr.hypotheses[0]
                text = self.sp.decode_pieces(tokens)
                results.append(normalize(text))
        return results

    def evaluate_dataset(self, src_file: Path, tgt_file: Path) -> Dict[str, float]:
        with open(src_file, "r", encoding="utf-8") as f_s, open(tgt_file, "r", encoding="utf-8") as f_t:
            srcs = [normalize(l) for l in f_s if l.strip()]
            tgts = [normalize(l) for l in f_t if l.strip()]

        assert len(srcs) == len(tgts), f"Mismatch between {src_file} and {tgt_file}"
        total = len(srcs)
        if total == 0:
            return {"exact_match": 0.0, "cer": 0.0, "unk_rate": 0.0, "total": 0}

        preds = self.predict_batch(srcs)

        exact_match = 0
        total_cer = 0.0
        unk_count = 0
        brand_retention = 0
        brand_total = 0

        precomputed_heldout = [b.lower() for b in HELDOUT_BRANDS]

        for s, t, p in zip(srcs, tgts, preds):
            if p == t:
                exact_match += 1
            total_cer += compute_cer(t, p)
            if "<unk>" in p or " ⁇ " in p:
                unk_count += 1

            # Check brand retention if target contains held-out brand
            for hb in precomputed_heldout:
                if hb in t:
                    brand_total += 1
                    if hb in p:
                        brand_retention += 1
                    break

        em_rate = (exact_match / total) * 100
        avg_cer = (total_cer / total) * 100
        unk_rate = (unk_count / total) * 100
        brand_rate = (brand_retention / brand_total * 100) if brand_total > 0 else 100.0

        return {
            "total": total,
            "exact_match": em_rate,
            "cer": avg_cer,
            "unk_rate": unk_rate,
            "brand_retention": brand_rate,
        }

    def run_diagnostics(self) -> List[Dict[str, str]]:
        diag_preds = self.predict_batch(DIAGNOSTIC_QUERIES)
        return [{"query": q, "prediction": p} for q, p in zip(DIAGNOSTIC_QUERIES, diag_preds)]


def evaluate_run(
    checkpoint_or_ct2: Path,
    tokenizer_path: Path,
    eval_dir: Path,
    device: str = "cpu",
    trust_checkpoint: bool = True,
) -> Dict[str, any]:
    model_dir = checkpoint_or_ct2
    if checkpoint_or_ct2.is_file() and checkpoint_or_ct2.suffix == ".pt":
        ct2_dir = checkpoint_or_ct2.parent / "ctranslate2_export"
        print(f"Exporting OpenNMT checkpoint {checkpoint_or_ct2} -> {ct2_dir}...")
        export_opennmt_checkpoint(
            checkpoint_or_ct2,
            tokenizer_path,
            ct2_dir,
            quantization="float16" if device == "cuda" else "float32",
            trust_checkpoint=trust_checkpoint,
        )
        model_dir = ct2_dir

    evaluator = BaseV3Evaluator(model_dir, tokenizer_path, device=device)

    eval_sets = [
        ("plasticity", eval_dir / "plasticity.src", eval_dir / "plasticity.tgt"),
        ("retention", eval_dir / "retention.src", eval_dir / "retention.tgt"),
        ("protection_seen", eval_dir / "protection_seen.src", eval_dir / "protection_seen.tgt"),
        ("protection_heldout", eval_dir / "protection_heldout.src", eval_dir / "protection_heldout.tgt"),
        ("user_centric", eval_dir / "user_centric.src", eval_dir / "user_centric.tgt"),
    ]

    results = {}
    for name, src, tgt in eval_sets:
        if src.exists() and tgt.exists():
            print(f"  Evaluating {name} ({src.name})...")
            results[name] = evaluator.evaluate_dataset(src, tgt)
        else:
            print(f"  Skipping {name} (file not found)")

    print("  Running qualitative diagnostics...")
    diagnostics = evaluator.run_diagnostics()

    return {
        "metrics": results,
        "diagnostics": diagnostics,
    }


def print_comparison_report(all_results: Dict[str, dict]):
    print("\n" + "=" * 95)
    print("                 REPAROS BASE V3 ABLATION COMPARISON REPORT                  ")
    print("=" * 95)
    header = f"{'Metric / Evaluation Suite':<30} | " + " | ".join(f"{run:^18}" for run in all_results.keys())
    print(header)
    print("-" * len(header))

    suites = ["plasticity", "retention", "protection_seen", "protection_heldout", "user_centric"]
    metric_keys = [("Exact Match (%)", "exact_match"), ("CER (%)", "cer"), ("UNK Rate (%)", "unk_rate")]

    for suite in suites:
        print(f"\n[{suite.upper()}]")
        for label, m_key in metric_keys:
            row = f"  - {label:<26} | "
            vals = []
            for run_name, r in all_results.items():
                m = r["metrics"].get(suite, {})
                v = m.get(m_key, 0.0)
                vals.append(f"{v:>16.2f}%")
            row += " | ".join(vals)
            print(row)
        # Brand retention for heldout
        if suite == "protection_heldout":
            row = f"  - {'Brand Protection (%)':<26} | "
            vals = []
            for run_name, r in all_results.items():
                m = r["metrics"].get(suite, {})
                v = m.get("brand_retention", 0.0)
                vals.append(f"{v:>16.2f}%")
            row += " | ".join(vals)
            print(row)

    print("\n" + "=" * 95)
    print("                          QUALITATIVE DIAGNOSTICS                            ")
    print("=" * 95)
    first_run = list(all_results.keys())[0]
    diag_queries = [d["query"] for d in all_results[first_run]["diagnostics"]]

    for q in diag_queries:
        print(f"\nInput: \"{q}\"")
        for run_name, r in all_results.items():
            for d in r["diagnostics"]:
                if d["query"] == q:
                    print(f"  {run_name:<16} -> \"{d['prediction']}\"")
                    break
    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base V3 Ablation models on Frozen Eval Suite.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/base_v3_eval"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer_v3/tokenizer.model"))
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Path(s) to checkpoint or CT2 model dir")
    parser.add_argument("--names", nargs="+", help="Names for the runs (e.g. Run_A Run_B Run_C)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-json", type=Path, default=Path("data/ablation_evaluation_results.json"))
    args = parser.parse_args()

    names = args.names or [Path(cp).name for cp in args.checkpoints]
    assert len(names) == len(args.checkpoints), "Number of names must match number of checkpoints"

    all_results = {}
    for name, cp in zip(names, args.checkpoints):
        print(f"\nEvaluating {name} ({cp})...")
        res = evaluate_run(Path(cp), args.tokenizer, args.eval_dir, device=args.device)
        all_results[name] = res

    print_comparison_report(all_results)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Evaluation report saved to {args.output_json}")


if __name__ == "__main__":
    main()
