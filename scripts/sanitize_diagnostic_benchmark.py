import json
import re
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

repo_root = Path("d:/New folder/QU-solution")
diag_path = repo_root / "benchmark/reparos-diagnostic-10k/gold.jsonl"
backup_path = repo_root / "benchmark/reparos-diagnostic-10k/gold.jsonl.orig.bak"

if not backup_path.exists():
    shutil.copy2(diag_path, backup_path)
    print(f"Backed up original diagnostic-10k to {backup_path}")

rows = [json.loads(l) for l in diag_path.read_text(encoding='utf-8').splitlines() if l.strip()]
print(f"Loaded {len(rows)} rows from {diag_path}")

fixed_sym_count = 0
fixed_abbr_count = 0

for r in rows:
    etype = r.get("error_type", "")
    s = r["input"]
    t = r["expected"]

    # 1. Sanitize address_symbol: fix ill-posed merged digits (e.g. 12015 -> 120 15 for 120/15)
    if etype == "address_symbol":
        # Find slashes in target e.g. 120/15 or 323/22/5 or 1/6l
        slashes = re.findall(r"\b(\d+)/(\d+(?:/\d+)*[a-zA-Z]?)\b", t)
        for part1, part2 in slashes:
            full_slash = f"{part1}/{part2}"
            merged = f"{part1}{part2}".lower()
            # If the source contains merged without separator
            if merged in s.lower() and full_slash not in s:
                # Replace merged with spaced slash error (realistic human typo: space instead of slash)
                spaced_typo = f"{part1} {part2}"
                # Case-insensitive replace
                pattern = re.compile(re.escape(merged), re.IGNORECASE)
                new_s = pattern.sub(spaced_typo, s, count=1)
                if new_s != s:
                    s = new_s
                    r["input"] = s
                    fixed_sym_count += 1
                    break

    # 2. Sanitize address_abbreviation: fix d/đ mapped to phố, enforce canonical target
    if etype == "address_abbreviation":
        s_words = s.strip().split()
        t_words = t.strip().split()
        if s_words and t_words:
            # If source begins with d/đ and target begins with phố -> change target to đường
            if s_words[0].lower() in ["d", "đ", "dg"] and t_words[0].lower() == "phố":
                t_words[0] = "đường"
                t = " ".join(t_words)
                r["expected"] = t
                fixed_abbr_count += 1

        # Canonicalize target abbreviations (ubnd, bv, etc.)
        t = re.sub(r"\bubnd\b", "ủy ban nhân dân", t, flags=re.IGNORECASE)
        t = re.sub(r"\bbv\b", "bệnh viện", t, flags=re.IGNORECASE)
        t = re.sub(r"\bkcn\b", "khu công nghiệp", t, flags=re.IGNORECASE)
        t = re.sub(r"\bđh\b", "đại học", t, flags=re.IGNORECASE)
        t = re.sub(r"\bthpt\b", "trung học phổ thông", t, flags=re.IGNORECASE)
        r["expected"] = t

print(f"Sanitized {fixed_sym_count} ill-posed address_symbol merged-digit queries to well-posed space-separated queries.")
print(f"Sanitized {fixed_abbr_count} address_abbreviation queries to strictly map d/đ to 'đường'.")

# Write sanitized benchmark
with open(diag_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Successfully wrote sanitized benchmark to {diag_path}!")
