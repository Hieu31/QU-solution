from __future__ import annotations

import argparse
import json

from reparos.curriculum_v2 import prepare_curriculum_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare capability curriculum v2; never starts training")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("pilot", "full"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = prepare_curriculum_v2(args.source, args.output, args.profile, args.seed)
    # Keep the CLI safe on legacy Windows consoles (for example CP1252).
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
