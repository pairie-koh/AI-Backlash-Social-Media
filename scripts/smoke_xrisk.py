"""Targeted smoke test for the new xrisk topic.

Picks candidate rows from the prior `regulation` / `general_neg` buckets (where
x-risk content is most likely hiding) plus a few controls from other buckets,
runs the new classifier prompt, and reports old vs new labels.

Run:
    OPENAI_API_KEY=... python scripts/smoke_xrisk.py
"""
import csv
import io
import json
import os
import sys
from pathlib import Path

csv.field_size_limit(2**31 - 1)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

import importlib.util


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tt_topics = _load("tt_classify_topics", ROOT / "tiktok" / "scripts" / "classify_topics.py")
yt_topics = _load("yt_classify_topics", ROOT / "youtube" / "scripts" / "classify_topics.py")
TT_PROMPT = tt_topics.TOPIC_PROMPT
parse_response = tt_topics.parse_response

from openai import OpenAI

MODEL = os.environ.get("LLM_MODEL", "gpt-5-mini")
client = OpenAI()


def call(prompt: str) -> dict | None:
    resp = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_response(resp.choices[0].message.content)


def pick(progress_path: Path, n_per_bucket: dict[str, int]) -> dict[str, str]:
    """Return {post_id: old_primary} sampled per bucket."""
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    buckets: dict[str, list[str]] = {}
    for pid, entry in data.items():
        primary = entry.get("primary", "") or ""
        buckets.setdefault(primary, []).append(pid)
    picked: dict[str, str] = {}
    for bucket, n in n_per_bucket.items():
        for pid in buckets.get(bucket, [])[:n]:
            picked[pid] = bucket
    return picked


def smoke_tiktok():
    print("=" * 70)
    print("TIKTOK SMOKE TEST")
    print("=" * 70)
    picks = pick(
        ROOT / "tiktok" / "data" / "tiktok_topic_classify_progress.v8topics.json",
        {"regulation": 12, "general_neg": 10, "jobs": 4, "art": 4},
    )
    csv_path = ROOT / "tiktok" / "data" / "tiktok_ai_backlash_final.csv"
    rows_by_id = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = row.get("post_id", "")
            if pid in picks:
                rows_by_id[pid] = row

    flagged_xrisk = []
    for i, (pid, old) in enumerate(picks.items(), 1):
        row = rows_by_id.get(pid)
        if not row:
            continue
        desc = (row.get("description") or "")[:1500]
        trans = (row.get("_transcript") or "")[:8000]
        prompt = TT_PROMPT.format(description=desc, transcript=trans)
        try:
            r = call(prompt)
        except Exception as e:
            print(f"  [{i:2}] {pid}  old={old:11}  ERROR: {str(e)[:80]}")
            continue
        if not r:
            print(f"  [{i:2}] {pid}  old={old:11}  PARSE FAIL")
            continue
        new = r.get("primary", "")
        ev = (r.get("evidence", "") or "")[:90]
        marker = "  *** XRISK ***" if new == "xrisk" else ""
        print(f"  [{i:2}] {pid}  old={old:11}  new={new:11}  ev={ev}{marker}")
        if new == "xrisk":
            flagged_xrisk.append((pid, old, ev))

    print(f"\n  TikTok xrisk hits: {len(flagged_xrisk)} / {len(picks)}")
    return flagged_xrisk


def smoke_youtube():
    print()
    print("=" * 70)
    print("YOUTUBE SMOKE TEST")
    print("=" * 70)
    picks = pick(
        ROOT / "youtube" / "data" / "youtube_topic_classify_progress.v8topics.json",
        {"regulation": 12, "general_neg": 10, "jobs": 4, "art": 4},
    )
    csv_path = ROOT / "youtube" / "data" / "youtube_ai_backlash_final.csv"
    rows_by_id = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = row.get("video_id", "") or row.get("post_id", "")
            if pid in picks:
                rows_by_id[pid] = row

    flagged_xrisk = []
    for i, (pid, old) in enumerate(picks.items(), 1):
        row = rows_by_id.get(pid)
        if not row:
            continue
        title = (row.get("title") or "")[:300]
        desc = (row.get("description") or "")[:1000]
        trans_raw = (row.get("_transcript") or row.get("transcript") or "")
        trans = yt_topics.sample_transcript(trans_raw)
        prompt = yt_topics.TOPIC_PROMPT.format(title=title, description=desc, transcript=trans)
        try:
            r = call(prompt)
        except Exception as e:
            print(f"  [{i:2}] {pid}  old={old:11}  ERROR: {str(e)[:80]}")
            continue
        if not r:
            print(f"  [{i:2}] {pid}  old={old:11}  PARSE FAIL")
            continue
        new = r.get("primary", "")
        ev = (r.get("evidence", "") or "")[:90]
        marker = "  *** XRISK ***" if new == "xrisk" else ""
        print(f"  [{i:2}] {pid}  old={old:11}  new={new:11}  ev={ev}{marker}")
        if new == "xrisk":
            flagged_xrisk.append((pid, old, ev))

    print(f"\n  YouTube xrisk hits: {len(flagged_xrisk)} / {len(picks)}")
    return flagged_xrisk


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    print(f"Model: {MODEL}\n")
    tt = smoke_tiktok()
    yt = smoke_youtube()
    print("\n" + "=" * 70)
    print(f"TOTAL xrisk flags: {len(tt) + len(yt)} / {30+30} candidates")
    print("=" * 70)


if __name__ == "__main__":
    main()
