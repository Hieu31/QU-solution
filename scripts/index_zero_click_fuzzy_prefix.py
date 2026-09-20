from pathlib import Path


path = Path("scripts/build_zero_click_production_benchmark.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    'pools[(value[0], len(value) // 5)].append(target)',
    'pools[(value[:2], len(value) // 5)].append(target)',
)
text = text.replace(
    "pools.get((value[0], bucket), ())",
    "pools.get((value[:2], bucket), ())",
)
path.write_text(text, encoding="utf-8")
