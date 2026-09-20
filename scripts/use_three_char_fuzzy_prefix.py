from pathlib import Path


path = Path("scripts/build_zero_click_production_benchmark.py")
text = path.read_text(encoding="utf-8").replace("value[:2]", "value[:3]")
path.write_text(text, encoding="utf-8")
