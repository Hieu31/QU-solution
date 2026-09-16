from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarking import __version__
from benchmarking.io import load_parallel_dataset, read_jsonl, strict_join, write_jsonl
from benchmarking.metrics import score_rows
from benchmarking.performance import artifact_size, latency_summary
from benchmarking.runners import load_decoding, parity_rows, run_ctranslate2, run_opennmt, run_webspell


def write_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_command(args: argparse.Namespace) -> None:
    rows = load_parallel_dataset(args.data, args.split, args.limit)
    write_jsonl(args.output, rows)
    write_json(Path(args.output).with_name("manifest.json"), {
        "schema_version": 1, "benchmark_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(), "split": args.split,
        "rows": len(rows), "seed": args.seed, "gold": str(Path(args.output).resolve()),
        "gold_sha256": sha256(args.output), "python": sys.version, "platform": platform.platform(),
        "normalization": "Unicode NFC + whitespace collapse; case preserved",
    })


def score_command(args: argparse.Namespace) -> None:
    gold = read_jsonl(args.gold)
    result: dict[str, object] = {"schema_version": 1, "systems": {}}
    regressions = []
    for value in args.prediction:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError("prediction must be NAME=PATH")
        metrics = score_rows(strict_join(gold, read_jsonl(path)))
        regressions.extend({"system": name, **row} for row in metrics.pop("regressions"))
        result["systems"][name] = metrics
    write_json(args.output, result)
    write_jsonl(Path(args.output).with_name("regressions.jsonl"), regressions)
    write_report(Path(args.output).with_name("report.md"), result)


def resources_command(args: argparse.Namespace) -> None:
    result = {"schema_version": 1, "systems": {}}
    for value in args.prediction:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError("prediction must be NAME=PATH")
        rows = read_jsonl(path)
        amortized = any("amortized" in str(row.get("latency_note", "")) for row in rows)
        result["systems"][name] = latency_summary(
            [float(row["latency_ms"]) for row in rows], official=False,
            note=("OpenNMT subprocess batch timing is amortized and is not batch=1 latency." if amortized else
                  "Quality-run timing only; official run requires warmup=100 and 10,000 randomized requests."),
        )
    for value in args.artifact:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError("artifact must be NAME=PATH")
        result["systems"].setdefault(name, {})["artifact_size_bytes"] = artifact_size(path)
    write_json(args.output, result)


def write_report(path: Path, quality: dict) -> None:
    lines = [
        "# Three-backend benchmark", "",
        "| System | Clean preservation | False correction | Regression | Exact noisy | Auto F1 | R@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in quality["systems"].items():
        safety, correction, ranking = value["safety"], value["correction"], value["ranking"]
        lines.append(
            f"| {name} | {safety['clean_preservation_rate']:.4f} | "
            f"{safety['false_correction_rate']:.4f} | {safety['regression_rate']:.4f} | "
            f"{correction['exact_noisy_accuracy']:.4f} | {correction['autocorrect_f1']:.4f} | "
            f"{ranking['recall@5']:.4f} |"
        )
    lines.extend(["", "Safety gates: clean preservation >= 99%, false correction <= 1%, regression <= 0.5%.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark-three")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data", required=True); prepare.add_argument("--output", required=True)
    prepare.add_argument("--split", default="test"); prepare.add_argument("--limit", type=int, default=0)
    prepare.add_argument("--seed", type=int, default=2026); prepare.set_defaults(func=prepare_command)
    webspell = commands.add_parser("run-webspell")
    webspell.add_argument("--gold", required=True); webspell.add_argument("--model", required=True); webspell.add_argument("--output", required=True)
    webspell.set_defaults(func=lambda a: run_webspell(read_jsonl(a.gold), a.model, a.output))
    opennmt = commands.add_parser("run-opennmt")
    opennmt.add_argument("--gold", required=True); opennmt.add_argument("--checkpoint", required=True)
    opennmt.add_argument("--tokenizer", required=True); opennmt.add_argument("--decoding-config"); opennmt.add_argument("--output", required=True)
    opennmt.set_defaults(func=lambda a: run_opennmt(read_jsonl(a.gold), a.checkpoint, a.tokenizer, a.output, decoding=load_decoding(a.decoding_config)))
    ct2 = commands.add_parser("run-ctranslate2")
    ct2.add_argument("--gold", required=True); ct2.add_argument("--model", required=True); ct2.add_argument("--decoding-config")
    ct2.add_argument("--device", default="cpu"); ct2.add_argument("--compute-type", default="default"); ct2.add_argument("--output", required=True)
    ct2.set_defaults(func=lambda a: run_ctranslate2(read_jsonl(a.gold), a.model, a.output, decoding=load_decoding(a.decoding_config), device=a.device, compute_type=a.compute_type))
    score = commands.add_parser("score")
    score.add_argument("--gold", required=True); score.add_argument("--prediction", action="append", required=True); score.add_argument("--output", required=True)
    score.set_defaults(func=score_command)
    parity = commands.add_parser("parity")
    parity.add_argument("--opennmt", required=True); parity.add_argument("--ctranslate2", required=True); parity.add_argument("--output", required=True)
    parity.set_defaults(func=lambda a: write_json(a.output, parity_rows(read_jsonl(a.opennmt), read_jsonl(a.ctranslate2))))
    resources = commands.add_parser("resources")
    resources.add_argument("--prediction", action="append", default=[])
    resources.add_argument("--artifact", action="append", default=[])
    resources.add_argument("--output", required=True)
    resources.set_defaults(func=resources_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
