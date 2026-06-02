"""Shared loader for sub-category drill-down plots (TikTok only, for now).

Mirrors the canonical filtering used by plot_2026_x_share_engagement.py so the
drill-downs are consistent with the published main bar plots:

  - Backlash:  rows with _backlash_final == YES; theme = _topic_primary.
  - Positive:  drop corpus-pollution rows, promote model_stan / general_hype
               secondaries to primary, keep _theme non-empty and != unknown.
  - 2026 cut:  create_time (or other date field) year == "2026".

YouTube halves are intentionally omitted until the classified YouTube CSVs are
available on this machine (they are >100MB and gitignored).
"""

import csv
import json
import re
from pathlib import Path

csv.field_size_limit(2**31 - 1)

REPO = Path(__file__).resolve().parent.parent

# ---- canonical pollution filter (copied from plot_2026_x_share_engagement) ----
AI_TEAM_PAT = re.compile(
    r"(#ai[\s_-]?team\b|ai team x|aiteam2026|aiteam membership|"
    r"partnership ai team|datang.*ai team|aiteam[\s\U0001F1EE]?‍?)",
    re.I,
)
HASHTAG_PIGGYBACK_PATS = [re.compile(r"only good vibes.*2026.*#ai", re.I)]
SECONDARY_PROMOTE_THEMES = {"model_stan", "general_hype"}


def _row_blob(r):
    return " ".join([r.get("description", "") or "", r.get("hashtags", "") or "",
                     r.get("title", "") or ""]).lower()


def _is_corpus_pollution(r):
    blob = _row_blob(r)
    if AI_TEAM_PAT.search(blob):
        return True
    return any(p.search(blob) for p in HASHTAG_PIGGYBACK_PATS)


def _promote_secondary(r):
    if (r.get("_theme") or "") not in SECONDARY_PROMOTE_THEMES:
        return
    raw = (r.get("_secondary_themes") or "").strip()
    if not raw or raw == "[]":
        return
    try:
        arr = json.loads(raw)
    except Exception:
        return
    if arr and arr[0]:
        r["_theme"] = arr[0]


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def year_of(r):
    for f in ("date_posted", "create_time", "date", "published_date", "upload_date"):
        v = r.get(f) or ""
        if v and len(v) >= 4 and v[:4].isdigit():
            return v[:4]
    return None


def is_2026(r):
    return year_of(r) == "2026"


def safe_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def engagement_of(r):
    return safe_int(r.get("views") or r.get("play_count"))


def load_tiktok(only_2026=True):
    """Returns (backlash_rows, positive_rows) for TikTok, canonical-filtered."""
    b = [r for r in load_csv(REPO / "tiktok/data/final/tiktok_backlash_final.csv")
         if (r.get("_backlash_final") or "").upper() == "YES"
         and (r.get("_topic_primary") or "")]

    p = []
    for r in load_csv(REPO / "tiktok/data/final/tiktok_positive_final.csv"):
        if _is_corpus_pollution(r):
            continue
        _promote_secondary(r)
        if (r.get("_theme") or "") and r.get("_theme") != "unknown":
            p.append(r)

    if only_2026:
        b = [r for r in b if is_2026(r)]
        p = [r for r in p if is_2026(r)]
    return b, p
