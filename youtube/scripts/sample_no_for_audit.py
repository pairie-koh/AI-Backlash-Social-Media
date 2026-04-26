"""Sample NO-classified videos to audit for false negatives."""
import csv
import json
import random
from pathlib import Path

csv.field_size_limit(2**31 - 1)

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "youtube_ai_backlash_filtered.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "audit_sample_NO.json"

random.seed(13)

no_rows = []
with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if (row.get("_backlash") or "").strip().upper() == "NO":
            no_rows.append(row)

print(f"NO rows: {len(no_rows)}")

def to_int(x):
    try:
        return int(x or 0)
    except Exception:
        return 0

# Suspicious-looking NO rows: keywords that should usually be backlash
suspicious_keywords = {
    "AI replacing workers", "AI taking jobs", "will AI take my job", "AI layoffs",
    "AI ruining everything", "AI slop", "I hate AI", "AI stealing art",
    "AI cheating school", "ChatGPT cheating", "deepfake danger", "AI surveillance",
    "AI carbon footprint", "AI misinformation", "AI voice cloning danger",
    "AI overhyped",
}

suspicious = [r for r in no_rows if (r.get("_search_keyword") or "") in suspicious_keywords]
suspicious_sorted = sorted(suspicious, key=lambda r: to_int(r.get("views")), reverse=True)

high_view_no = sorted(no_rows, key=lambda r: to_int(r.get("views")), reverse=True)[:8]
suspicious_sample = suspicious_sorted[:8] if len(suspicious_sorted) >= 8 else suspicious_sorted
random_sample = random.sample(no_rows, 14)

picks = []
seen = set()
for bucket, rows in [
    ("high_view_NO", high_view_no),
    ("suspicious_kw_NO", suspicious_sample),
    ("random_NO", random_sample),
]:
    for r in rows:
        vid = r.get("video_id")
        if vid in seen:
            continue
        seen.add(vid)
        transcript = (r.get("_transcript_text") or "").strip()
        picks.append({
            "bucket": bucket,
            "video_id": vid,
            "title": r.get("title"),
            "keyword": r.get("_search_keyword"),
            "views": to_int(r.get("views")),
            "channel": r.get("youtuber"),
            "description_head": (r.get("description") or "")[:300],
            "transcript_len": len(transcript),
            "transcript_snippet": transcript[:1500],
            "label": r.get("_backlash"),
        })

OUT_PATH.write_text(json.dumps(picks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(picks)} samples to {OUT_PATH}")
