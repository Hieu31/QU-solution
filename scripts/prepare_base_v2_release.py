from __future__ import annotations

import argparse
import json

from reparos.base_v2 import BaseV2Config
from reparos.base_v2_release import prepare_base_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit or materialize release-candidate ReparoS Base V2")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profiles", default="32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-groups-per-split", type=int)
    parser.add_argument("--sample-per-family", type=int, default=20)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    result = prepare_base_v2(args.source, args.output, BaseV2Config(
        seed=args.seed,
        profiles=tuple(int(x) for x in args.profiles.split(",") if x.strip()),
        max_groups_per_split=args.max_groups_per_split,
        materialize=args.materialize,
        sample_per_family=args.sample_per_family,
    ))
    print(json.dumps({
        "materialized": args.materialize,
        "output": args.output,
        "leakage": result["leakage"],
        "quality_gate": result["quality_gate"],
        "counts": {
            split: {
                "clean": value["clean_groups"],
                "noisy": {profile: counts["noisy"] for profile, counts in value["profiles"].items()},
            }
            for split, value in result["splits"].items()
        },
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
