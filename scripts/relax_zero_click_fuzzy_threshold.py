from pathlib import Path


path = Path("scripts/build_zero_click_production_benchmark.py")
text = path.read_text(encoding="utf-8")
text = text.replace("score_cutoff=88", "score_cutoff=80")
text = text.replace("top_score < 92 or top_score - second_score < 5", "top_score < 85 or top_score - second_score < 3")
path.write_text(text, encoding="utf-8")
