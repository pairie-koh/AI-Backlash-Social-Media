"""Sample YES-classified videos and dump title + transcript snippet for manual audit."""
import csv
import json
import random
import sys
from pathlib import Path

csv.field_size_limit(2**31 - 1)

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "youtube_ai_backlash_filtered.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "audit_sample.json"

random.seed(7)

yes_rows = []
no_rows = []
with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = (row.get("_backlash") or "").strip().upper()
        if label == "YES":
            yes_rows.append(row)
        elif label == "NO":
            no_rows.append(row)

print(f"YES rows: {len(yes_rows)}  NO rows: {len(no_rows)}")

def to_int(x):
    try:
        return int(x or 0)
    except Exception:
        return 0

yes_sorted = sorted(yes_rows, key=lambda r: to_int(r.get("views")), reverse=True)
top_views = yes_sorted[:8]
bottom_views = [r for r in yes_sorted if to_int(r.get("views")) > 0][-8:]
random_pick = random.sample(yes_rows, 14)

picks = []
seen = set()
for bucket, rows in [("top_views", top_views), ("bottom_views", bottom_views), ("random", random_pick)]:
    for r in rows:
        vid = r.get("video_id")
        if vid in seen:
            continue
        seen.add(vid)
        transcript = (r.get("_transcript_text") or "").strip()
        snippet = transcript[:1500]
        picks.append({
            "bucket": bucket,
            "video_id": vid,
            "title": r.get("title"),
            "keyword": r.get("_search_keyword"),
            "views": to_int(r.get("views")),
            "likes": to_int(r.get("likes")),
            "channel": r.get("youtuber"),
            "description_head": (r.get("description") or "")[:300],
            "transcript_len": len(transcript),
            "transcript_snippet": snippet,
            "label": r.get("_backlash"),
        })

OUT_PATH.write_text(json.dumps(picks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(picks)} samples to {OUT_PATH}")
