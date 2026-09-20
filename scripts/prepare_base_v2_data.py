from __future__ import annotations

import argparse
import json

from reparos.base_v2 import BaseV2Config
from reparos.base_v2_strict import prepare_base_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit or materialize strict deterministic ReparoS Base V2 data")
    parser.add_argument("--source", required=True, help="prepared-v4-leakfree root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profiles", default="8,16,32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-groups-per-split", type=int)
    parser.add_argument("--sample-per-family", type=int, default=20)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    profiles = tuple(int(value) for value in args.profiles.split(",") if value.strip())
    result = prepare_base_v2(args.source, args.output, BaseV2Config(
        seed=args.seed, profiles=profiles,
        max_groups_per_split=args.max_groups_per_split,
        materialize=args.materialize,
        sample_per_family=args.sample_per_family,
    ))
    print(json.dumps({
        "schema_version": result["schema_version"],
        "materialized": args.materialize,
        "output": args.output,
        "leakage": result["leakage"],
        "splits": {
            split: {"clean_groups": value["clean_groups"], "profiles": value["profiles"]}
            for split, value in result["splits"].items()
        },
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
