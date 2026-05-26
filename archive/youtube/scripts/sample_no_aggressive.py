"""Targeted false-negative audit: NO rows whose title contains backlash trigger words."""
import csv
import json
import random
import re
from pathlib import Path

csv.field_size_limit(2**31 - 1)

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "youtube_ai_backlash_filtered.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "audit_sample_NO_aggressive.json"

random.seed(31)

TRIGGERS = re.compile(
    r"\b("
    r"ruin(?:ed|ing|s)?|destroy(?:ed|ing|s)?|kill(?:ed|ing|s)?|dead|dying|die|"
    r"danger(?:ous)?|threat(?:en|ening)?|scary|terrifying|nightmare|disaster|crisis|"
    r"hate|stop|ban|fight|against|enemy|enemies|"
    r"fear|afraid|worry|worried|warn(?:ing)?|alarm|"
    r"layoff|layoffs|fired|replaced?|replacing|stealing|theft|steal|"
    r"slop|garbage|trash|fake|scam|lie|lying|liar|"
    r"bubble|hype(?:d)?|overrated|overhyped|fraud|"
    r"deepfake|surveillance|spying|stalking|"
    r"dystop|orwell|terminator|skynet|apocalyp|"
    r"cheating|plagiari|"
    r"refuse|refusing|boycott|protest|"
    r"end of|going to die|going to kill|will replace"
    r")\b",
    re.IGNORECASE,
)

# AI must actually be referenced
AI_RE = re.compile(r"\b(ai|a\.i\.|artificial intelligence|chatgpt|gpt|llm|deepfake|gen(?:erative)?\s*ai|claude|gemini|copilot|midjourney|stable diffusion|sora|veo)\b", re.IGNORECASE)

candidates = []
with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if (row.get("_backlash") or "").strip().upper() != "NO":
            continue
        title = row.get("title") or ""
        desc = row.get("description") or ""
        # Must reference AI in title OR description, AND have a trigger word in title
        if not AI_RE.search(title) and not AI_RE.search(desc[:500]):
            continue
        if not TRIGGERS.search(title):
            continue
        candidates.append(row)

print(f"NO rows with AI mention + backlash trigger in title: {len(candidates)}")

def to_int(x):
    try: return int(x or 0)
    except: return 0

# Stratify: half top-views, half random
candidates_sorted = sorted(candidates, key=lambda r: to_int(r.get("views")), reverse=True)
top = candidates_sorted[:15]
rand = random.sample(candidates, 15) if len(candidates) >= 15 else candidates

picks, seen = [], set()
for bucket, rows in [("top_view_suspect_NO", top), ("random_suspect_NO", rand)]:
    for r in rows:
        vid = r.get("video_id")
        if vid in seen: continue
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
            "transcript_snippet": transcript[:1800],
            "label": r.get("_backlash"),
        })

OUT_PATH.write_text(json.dumps(picks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(picks)} samples to {OUT_PATH}")
