from __future__ import annotations

import argparse
import json

from reparos.typing_curriculum import CurriculumConfig, prepare_typing_curriculum


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ReparoS IME/abbreviation/mixed/domain curriculum data; never starts training")
    parser.add_argument("--source", required=True, help="prepared-v4-leakfree root containing split/noisy_pairs.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ime-variants", type=int, default=2)
    parser.add_argument("--abbreviation-variants", type=int, default=2)
    parser.add_argument("--mixed-variants", type=int, default=2)
    args = parser.parse_args()
    manifest = prepare_typing_curriculum(
        args.source,
        args.output,
        CurriculumConfig(
            seed=args.seed,
            ime_variants_per_query=args.ime_variants,
            abbreviation_variants_per_query=args.abbreviation_variants,
            mixed_variants_per_query=args.mixed_variants,
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

