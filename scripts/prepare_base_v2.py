from __future__ import annotations

import argparse
import json
import sys

from reparos.base_v2 import BaseV2Config
from reparos.base_v2_release import prepare_base_v2


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Generate the quality-gated ReparoS Base V2 dataset from leak-free OSM splits."
    )
    parser.add_argument("--source", required=True, help="Root containing train/validation/test noisy_pairs.csv")
    parser.add_argument("--output", required=True, help="Output root")
    parser.add_argument(
        "--profiles",
        default="32",
        help="Comma-separated nested profile sizes, for example 8,16,32 (default: 32)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-per-family", type=int, default=20)
    parser.add_argument("--max-groups-per-split", type=int)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Write the complete dataset. Without this flag, only sampled audit files are written.",
    )
    args = parser.parse_args()

    config = BaseV2Config(
        profiles=tuple(int(value.strip()) for value in args.profiles.split(",") if value.strip()),
        seed=args.seed,
        sample_per_family=args.sample_per_family,
        materialize=args.materialize,
        max_groups_per_split=args.max_groups_per_split,
    )
    result = prepare_base_v2(args.source, args.output, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
