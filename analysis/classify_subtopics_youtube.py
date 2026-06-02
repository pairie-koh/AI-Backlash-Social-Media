"""Tier 2 sub-classification for YouTube — via OpenRouter realtime.

Mirrors classify_subtopics.py (TikTok) but routes through OpenRouter's
OpenAI-compatible endpoint so we can use the same Haiku 4.5 model without
holding an Anthropic key. No batch discount; we make up for it with high
parallelism (default 32 workers) since this is fire-and-wait realtime.

Source rows come through drilldown_common.load_youtube (canonical filter),
so sub-topics line up exactly with the published main bar plots. We classify
ALL dates (not just 2026) to maximise n; the plotting step filters to 2026.

Output sidecars (keyed by video_id):
    youtube/data/final/youtube_backlash_subtopics.csv
    youtube/data/final/youtube_positive_subtopics.csv
Columns: video_id, theme, subtopic, confidence

Resume-safe: rows already in the sidecar are skipped on re-run.

Usage:
    python analysis/classify_subtopics_youtube.py --corpus backlash --limit 20  # smoke
    python analysis/classify_subtopics_youtube.py --corpus backlash             # full
    python analysis/classify_subtopics_youtube.py --corpus positive
    WORKERS=64 python analysis/classify_subtopics_youtube.py --corpus positive  # crank
"""

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drilldown_common import load_youtube, evidence_of_yt, REPO
from classify_subtopics import (PROMPT, OUT_RE, THEME_DESC, menu_text,
                                BACKLASH_SUBTAX, POSITIVE_SUBTAX)

csv.field_size_limit(2**31 - 1)

MODEL = os.environ.get("MODEL", "anthropic/claude-haiku-4.5")
WORKERS = int(os.environ.get("WORKERS", "32"))
BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 3


def get_api_key():
    envp = REPO / ".env"
    if not envp.exists():
        raise RuntimeError(f"missing {envp}")
    for line in envp.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY not in .env")


def build_prompt(r, theme_key, subtax):
    theme = r[theme_key]
    return PROMPT.format(
        platform="YouTube",
        theme=theme,
        theme_desc=THEME_DESC.get(theme, theme),
        menu=menu_text(subtax[theme]),
        description=(r.get("description") or "")[:600],
        hashtags="",  # YouTube has no hashtags field; tags live inside description
        keyword=r.get("_search_keyword") or "",
        evidence=evidence_of_yt(r) or "(none)",
    )


def classify_row(client, r, theme_key, subtax):
    prompt = build_prompt(r, theme_key, subtax)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL, max_tokens=40,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            m = OUT_RE.search(text)
            if not m:
                return "other", "low"
            sub = m.group(1).lower()
            theme = r[theme_key]
            if sub not in subtax[theme] and sub != "other":
                sub = "other"
            return sub, m.group(2).lower()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err


def sidecar_path(corpus):
    return REPO / f"youtube/data/final/youtube_{corpus}_subtopics.csv"


def load_done(corpus):
    p = sidecar_path(corpus)
    if not p.exists():
        return set()
    with open(p, encoding="utf-8", newline="") as f:
        return {d["video_id"] for d in csv.DictReader(f)}


def run(corpus, limit):
    if corpus == "backlash":
        rows, _ = load_youtube(only_2026=False)
        theme_key, subtax = "_topic_primary", BACKLASH_SUBTAX
    else:
        _, rows = load_youtube(only_2026=False)
        theme_key, subtax = "_theme", POSITIVE_SUBTAX

    rows = [r for r in rows if r.get(theme_key) in subtax and r.get("video_id")]

    done = load_done(corpus)
    todo = [r for r in rows if r["video_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{corpus}] total={len(rows):,} done={len(done):,} "
          f"todo_now={len(todo):,}  model={MODEL}  workers={WORKERS}")
    if not todo:
        return []

    client = openai.OpenAI(api_key=get_api_key(), base_url=BASE_URL)

    out_path = sidecar_path(corpus)
    write_header = not out_path.exists()
    file_lock = threading.Lock()
    f_out = open(out_path, "a", encoding="utf-8", newline="")
    writer = csv.writer(f_out)
    if write_header:
        writer.writerow(["video_id", "theme", "subtopic", "confidence"])
        f_out.flush()

    stats = {"ok": 0, "err": 0, "errors_logged": 0}

    def work(r):
        try:
            sub, conf = classify_row(client, r, theme_key, subtax)
            with file_lock:
                writer.writerow([r["video_id"], r[theme_key], sub, conf])
                f_out.flush()
                stats["ok"] += 1
            return True
        except Exception as e:
            with file_lock:
                stats["err"] += 1
                if stats["errors_logged"] < 5:
                    print(f"  err: {str(e)[:140]}")
                    stats["errors_logged"] += 1
            return False

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, _ in enumerate(as_completed(futs), 1):
            if i % 200 == 0 or i == len(futs):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                eta = (len(futs) - i) / rate if rate else 0
                print(f"  ...{i}/{len(futs)}  ok={stats['ok']:,} err={stats['err']:,} "
                      f"{rate:.1f} req/s  eta {eta/60:.1f} min")

    f_out.close()
    print(f"[{corpus}] done: ok={stats['ok']:,} err={stats['err']:,} "
          f"-> {out_path.name}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["backlash", "positive"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(args.corpus, args.limit)


if __name__ == "__main__":
    main()
