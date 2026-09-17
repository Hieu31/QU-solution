from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from reparos.curriculum_v2 import prepare_curriculum_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare capability curriculum v2; never starts training")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("pilot", "full"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = prepare_curriculum_v2(args.source, args.output, args.profile, args.seed)

    # Evaluation is part of the deployable curriculum bundle. Recreate it on
    # every materialization so deleting curriculum-v2 cannot silently remove
    # the pilot gates.
    repo_root = Path(__file__).resolve().parents[1]
    evaluation_root = Path(args.output).resolve().parent / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    evaluation_sources = {
        "user-centric-v2.jsonl": repo_root / "benchmark/reparos-user-centric-v2/gold.jsonl",
        "composition-4k.jsonl": repo_root / "benchmark/reparos-compositional-4k/gold.jsonl",
        "diagnostic-10k.jsonl": repo_root / "benchmark/reparos-diagnostic-10k/gold.jsonl",
    }
    for output_name, source_path in evaluation_sources.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"missing frozen evaluation source: {source_path}")
        shutil.copy2(source_path, evaluation_root / output_name)
    result["evaluation"] = {
        name: str((evaluation_root / name).resolve()) for name in evaluation_sources
    }
    # Keep the CLI safe on legacy Windows consoles (for example CP1252).
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
