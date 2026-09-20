from pathlib import Path


path = Path("scripts/build_zero_click_production_benchmark.py")
text = path.read_text(encoding="utf-8")
old = '''    fuzzy_source = sorted(counts, key=lambda key: (-counts[key], key))[:200_000]
'''
new = '''    # The frequent 30K unmatched queries are sufficient for a 500-row
    # high-confidence stratum and keep runtime bounded.
    fuzzy_source = sorted(counts, key=lambda key: (-counts[key], key))[:30_000]
'''
assert old in text
text = text.replace(old, new)
old_choices = '''        if not choices:
            continue
        matches = process.extract(value, choices, scorer=fuzz.WRatio, limit=2, score_cutoff=88)
'''
new_choices = '''        if not choices:
            continue
        choices = list(dict.fromkeys(choices))
        matches = process.extract(value, choices, scorer=fuzz.WRatio, limit=2, score_cutoff=88)
'''
assert old_choices in text
path.write_text(text.replace(old_choices, new_choices), encoding="utf-8")
