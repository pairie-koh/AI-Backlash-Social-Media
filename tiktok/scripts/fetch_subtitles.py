"""Fetch TikTok auto-generated subtitles (WebVTT) directly from Bright Data scrape.

42.7% of scraped TikTok records have subtitle_url populated — these are TikTok's
own Whisper-generated captions, hosted on TikTok's CDN. Pulling them avoids the
yt-dlp + Groq-Whisper pipeline entirely (no per-IP throttling, no transcription cost).

Output: writes plain-text transcripts to data/whisper_transcripts/{post_id}.txt
(same dir whisper_transcribe.py uses, so the filter script picks them up unchanged).
"""

import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw" / "tiktok_raw.json"
OUT_DIR = DATA / "whisper_transcripts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->")
CUE_NUM_RE = re.compile(r"^\d+$")


def webvtt_to_text(vtt: str) -> str:
    lines = []
    for line in vtt.splitlines():
        s = line.strip()
        if not s or s == "WEBVTT":
            continue
        if s.startswith("NOTE") or s.startswith("STYLE"):
            continue
        if TIMESTAMP_RE.match(s):
            continue
        if CUE_NUM_RE.match(s):
            continue
        lines.append(s)
    return " ".join(lines).strip()


def fetch_one(post_id: str, url: str, session: requests.Session):
    out_path = OUT_DIR / f"{post_id}.txt"
    if out_path.exists():
        return ("skip_exists", post_id)
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return ("http_error", post_id, r.status_code)
        text = webvtt_to_text(r.text)
        if not text:
            return ("empty", post_id)
        out_path.write_text(text, encoding="utf-8")
        return ("ok", post_id, len(text))
    except Exception as e:
        return ("error", post_id, str(e)[:80])


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"Loading raw videos...")
    videos = json.load(open(RAW, encoding="utf-8"))
    targets = [
        (str(v["post_id"]), v["subtitle_url"])
        for v in videos
        if v.get("post_id") and v.get("subtitle_url")
    ]
    print(f"  total videos: {len(videos)}")
    print(f"  with subtitle_url: {len(targets)}")

    pending = [(pid, url) for pid, url in targets if not (OUT_DIR / f"{pid}.txt").exists()]
    print(f"  already in whisper_transcripts/: {len(targets) - len(pending)}")
    print(f"  to fetch: {len(pending)}")
    print(f"  workers: {workers}")
    print()

    if not pending:
        print("Nothing to do.")
        return

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; backlash-research/1.0)"})

    counts = {"ok": 0, "empty": 0, "http_error": 0, "error": 0, "skip_exists": 0}
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_one, pid, url, session) for pid, url in pending]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            kind = res[0]
            counts[kind] = counts.get(kind, 0) + 1
            if i % 250 == 0 or i == len(pending):
                elapsed = time.time() - start
                rate = i / max(elapsed, 1e-9)
                eta = (len(pending) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i}/{len(pending)}] ok={counts['ok']} "
                    f"empty={counts['empty']} http_err={counts['http_error']} "
                    f"err={counts['error']} | {rate:.1f}/s, ETA {eta:.0f}s"
                )

    elapsed = int(time.time() - start)
    print(f"\nDone in {elapsed//60}m {elapsed%60}s")
    print(f"  Successful: {counts['ok']}")
    print(f"  Empty: {counts['empty']}")
    print(f"  HTTP errors: {counts['http_error']}")
    print(f"  Other errors: {counts['error']}")
    print(f"  Total transcripts on disk: {len(list(OUT_DIR.glob('*.txt')))}")


if __name__ == "__main__":
    main()
