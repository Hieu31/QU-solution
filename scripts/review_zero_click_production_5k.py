from __future__ import annotations

import argparse, csv, json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
UNSAFE_CORRECTIONS = {"koi the", "nha h", "nha n", "nha k", "nam hoa", "cho hoa", "nguyen thi", "nha tu", "duong s", "la ca", "the lu"}

def plain(text: str) -> str:
    text = text.casefold().replace("\u0111", "d")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip()

def mark_count(text: str) -> int:
    normalized = unicodedata.normalize("NFD", text.replace("\u0111", "d").replace("\u0110", "D"))
    return sum(unicodedata.category(c) == "Mn" for c in normalized) + text.casefold().count("\u0111")

def incomplete(text: str) -> bool:
    last = text.casefold().strip().split()[-1] if text.strip() else ""
    return last in {"v", "vi", "viÃª", "t", "th", "thc", "máº§m", "ba", "x"}
    return last in {"v", "vi", "vi\u00ea", "t", "th", "thc", "m\u1ea7m", "ba", "x"}
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--annotations", required=True); ap.add_argument("--osm-root", required=True); ap.add_argument("--output", required=True)
    args = ap.parse_args(); variants = defaultdict(Counter); exact = Counter(); seen = set()
    for split in ("train", "validation", "test"):
        path = Path(args.osm_root) / split / "noisy_pairs.csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                target = row["correct_query"].strip(); key = (target, row.get("entity_id", ""))
                if not target or key in seen: continue
                seen.add(key); variants[plain(target)][target] += 1; exact[target.casefold()] += 1
    source = Path(args.annotations)
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f); rows = list(reader); fields = list(reader.fieldnames or [])
    actions = Counter()
    for row in rows:
        query = row["input"].strip(); qplain = plain(query); stratum = row["stratum"]
        choices = variants.get(qplain, Counter()); ranked = choices.most_common(2)
        action, expected, confidence, reason = "unknown", "", "low", "intent_not_recoverable_from_zero_click_query"
        if exact[query.casefold()] or stratum == "exact_osm_keep":
            action, expected, confidence, reason = "keep", query, "high", "query_is_attested_as_clean_osm_name"
        elif incomplete(query):
            action, reason, confidence = "incomplete", "query_ends_in_likely_unfinished_token", "medium"
        elif ranked:
            best, support = ranked[0]; runner = ranked[1][1] if len(ranked) > 1 else 0
            dominant = support >= 2 and support >= 2 * max(1, runner)
            if best.casefold() == query.casefold():
                action, expected, confidence, reason = "keep", query, "high", "query_matches_dominant_osm_form"
            elif dominant and plain(best) == qplain and len(qplain) >= 4 and len(qplain.split()) >= 2 and qplain not in UNSAFE_CORRECTIONS and mark_count(query) == 0 and mark_count(best) > 0:
                action, expected, confidence, reason = "correct", best, "high", f"dominant_same_plain_osm_form_support={support};runner={runner}"
        elif stratum == "number_address" and len(query) >= 5:
            action, expected, confidence, reason = "keep", query, "medium", "complete_structured_address_query"
        row["action"], row["expected"], row["annotator_confidence"] = action, expected, confidence
        row["review_status"] = "reviewed"
        row["notes"] = f"reviewer=codex; rule={reason}"
        if action == "correct": row["error_types"] = row["error_types"].strip() or "missing_diacritics_or_orthography"
        actions[action] += 1
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "actions": dict(actions), "output": str(destination)}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()

