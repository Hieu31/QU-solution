from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from benchmarking.io import normalize_text, read_jsonl
from benchmarking.performance import percentile
from benchmarking.runners import load_decoding
from reparos.serving.ctranslate2 import CTranslate2Predictor
from webspell.pipeline.model import WebSpellModel


def build_opennmt_engine(checkpoint: str, tokenizer: str, decoding_config: str):
    from onmt.bin.translate import _get_parser
    from onmt.inference_engine import InferenceEnginePY
    from onmt.utils.misc import set_random_seed, use_gpu
    from onmt.utils.parse import ArgumentParser

    settings = load_decoding(decoding_config)
    arguments = [
        "-model", checkpoint, "-src", "unused", "-output", "unused",
        "-beam_size", str(settings.beam_size), "-n_best", str(settings.num_hypotheses),
        "-min_length", str(settings.min_decoding_length),
        "-max_length", str(settings.max_decoding_length), "-max_length_ratio", "0",
        "-length_penalty", "avg", "-alpha", str(settings.length_penalty),
        "-coverage_penalty", "none", "-beta", str(settings.coverage_penalty),
        "-block_ngram_repeat", str(settings.no_repeat_ngram_size),
        "-transforms", "sentencepiece", "-src_subword_model", tokenizer,
        "-tgt_subword_model", tokenizer, "-gpu", "-1",
    ]
    if settings.disable_unk:
        arguments.append("-ban_unk_token")
    opt = _get_parser().parse_args(arguments)
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)
    set_random_seed(opt.seed, use_gpu(opt))
    return InferenceEnginePY(opt)


def opennmt_top1(engine, text: str) -> str:
    _, predictions = engine.infer_list([text])
    value = predictions[0][0] if isinstance(predictions[0], list) else predictions[0]
    return normalize_text(str(value))


def measure(call, queries: list[str], warmup: int, repeats: int) -> tuple[list[str], list[list[float]]]:
    for _ in range(warmup):
        for query in queries:
            call(query)
    outputs = [""] * len(queries)
    samples = [[] for _ in queries]
    for _ in range(repeats):
        for index, query in enumerate(queries):
            started = time.perf_counter_ns()
            output = call(query)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            outputs[index] = normalize_text(str(output))
            samples[index].append(elapsed)
    return outputs, samples


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True); parser.add_argument("--webspell-model", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--ctranslate2-model", required=True); parser.add_argument("--decoding-config", required=True)
    parser.add_argument("--warmup", type=int, default=3); parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gold = read_jsonl(args.gold); queries = [row["input"] for row in gold]
    webspell = WebSpellModel.load(args.webspell_model)
    onmt = build_opennmt_engine(args.checkpoint, args.tokenizer, args.decoding_config)
    ct2 = CTranslate2Predictor(args.ctranslate2_model, device="cpu", compute_type="float32")
    try:
        web_outputs, web_times = measure(lambda q: webspell.corrected_text(q, webspell.predict(q)), queries, args.warmup, args.repeats)
        onmt_outputs, onmt_times = measure(lambda q: opennmt_top1(onmt, q), queries, args.warmup, args.repeats)
        decode = load_decoding(args.decoding_config)
        ct2_outputs, ct2_times = measure(lambda q: ct2.predict(q, decoding=decode)["top1_query"], queries, args.warmup, args.repeats)
    finally:
        webspell.close()
        onmt.terminate()
    records = []
    for index, row in enumerate(gold):
        record = {"query_id": row["query_id"], "input": row["input"], "expected": row["expected"]}
        for name, outputs, samples in (("webspell", web_outputs, web_times), ("opennmt", onmt_outputs, onmt_times), ("ctranslate2", ct2_outputs, ct2_times)):
            record[name] = {"output": outputs[index], "correct": outputs[index] == row["expected"],
                            "p50_ms": percentile(samples[index], .5), "p95_ms": percentile(samples[index], .95)}
        records.append(record)
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"warmup": args.warmup, "repeats": args.repeats, "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("#\tInput\tExpected\tWebSpell output\tWeb ms p50/p95\tOpenNMT output\tONMT ms p50/p95\tCTranslate2 output\tCT2 ms p50/p95")
    for index, row in enumerate(records, 1):
        cells = [str(index), row["input"], row["expected"]]
        for name in ("webspell", "opennmt", "ctranslate2"):
            value = row[name]; cells.extend([value["output"] + (" ✓" if value["correct"] else " ✗"), f"{value['p50_ms']:.3f} / {value['p95_ms']:.3f}"])
        print("\t".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
