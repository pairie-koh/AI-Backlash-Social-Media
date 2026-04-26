"""
Filter TikTok videos for AI backlash using LLM classification.

Deduplicates raw Bright Data output, then uses an LLM to classify each
video as expressing AI backlash (YES) or not (NO).

Usage:
    export OPENROUTER_API_KEY=...
    python scripts/filter_by_topic.py
"""

import csv
import io
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_JSON = DATA_DIR / "raw" / "tiktok_raw.json"
WHISPER_DIR = DATA_DIR / "whisper_transcripts"
TIKTOK_DIR = DATA_DIR / "tiktok_transcripts"
OUTPUT_CSV = DATA_DIR / "tiktok_ai_backlash_filtered.csv"
PROGRESS_FILE = DATA_DIR / "filter_llm_progress.json"
KEYS_FILE = REPO_ROOT / ".openrouter_keys"

# --- API setup ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEYS = os.environ.get("OPENROUTER_API_KEYS", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")
WORKERS_PER_KEY = int(os.environ.get("WORKERS_PER_KEY", "3"))

BACKLASH_PROMPT = """A TikTok video was found by searching for AI-related keywords. Does this video express criticism, concern, or backlash about AI?

YES — The video expresses negativity, worry, or criticism about AI. This includes:
- Fear of job displacement or automation replacing workers
- Concern about AI's environmental impact (energy, water, data centers, carbon)
- Criticism of AI-generated art, music, or writing (art theft, creative displacement)
- Privacy or surveillance concerns related to AI
- Deepfake, misinformation, or manipulation worries
- AI in education concerns (cheating, plagiarism, deskilling)
- Calls to regulate, ban, or slow down AI
- Existential risk, safety, or alignment concerns
- Complaints about AI slop, spam, or low-quality AI content flooding the internet
- Frustration with AI customer service replacing human support
- Criticism that AI products are getting worse (ChatGPT, Copilot, Google AI, etc.)
- AI ruining search results or making the internet worse
- General anti-AI sentiment or tech backlash
- Personal stories about being negatively affected by AI
- Celebrating human-made work as resistance to AI ("no AI used", "real artist")

NO — The video mentions AI but is NOT expressing backlash. This includes:
- AI tutorials, tips, or how-to content
- Positive AI product reviews or demos
- AI hype or excitement content
- Neutral news reporting about AI without critical angle
- Using AI tools in the video (but not criticizing them)
- AI memes or humor without a critical message
- Marketing or promotional AI content
- General tech content that happens to mention AI

Respond with exactly one word: YES or NO

---
Description: {description}
Transcript: {transcript}"""


def _load_openrouter_keys():
    """Load OpenRouter keys from env (comma-sep) or .openrouter_keys file."""
    if OPENROUTER_API_KEYS.strip():
        return [k.strip() for k in OPENROUTER_API_KEYS.split(",") if k.strip()]
    if KEYS_FILE.exists():
        return [
            ln.strip()
            for ln in KEYS_FILE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    if OPENROUTER_API_KEY:
        return [OPENROUTER_API_KEY]
    return []


def get_llm_clients():
    """Return list of (client, model) pairs — one per OpenRouter key, or fallback."""
    from openai import OpenAI

    keys = _load_openrouter_keys()
    if keys:
        return [
            (OpenAI(base_url="https://openrouter.ai/api/v1", api_key=k), LLM_MODEL)
            for k in keys
        ]
    if OPENAI_API_KEY:
        return [(OpenAI(api_key=OPENAI_API_KEY), "gpt-4o-mini")]
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            return [(anthropic.Anthropic(api_key=ANTHROPIC_API_KEY), "claude-haiku-4-20250414")]
        except ImportError:
            pass
    print("ERROR: Set OPENROUTER_API_KEYS / OPENROUTER_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY")
    sys.exit(1)


def llm_call(client, model: str, prompt: str, max_tokens: int = 10) -> str:
    """Unified LLM call for OpenAI-compatible and Anthropic clients."""
    try:
        import anthropic
        is_anthropic = isinstance(client, anthropic.Anthropic)
    except ImportError:
        is_anthropic = False

    if is_anthropic:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    else:
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()


def get_transcript(post_id: str) -> str:
    """Load transcript: whisper > tiktok captions."""
    whisper_file = WHISPER_DIR / f"{post_id}.txt"
    if whisper_file.exists():
        text = whisper_file.read_text(encoding="utf-8").strip()
        if text:
            return text

    tiktok_file = TIKTOK_DIR / f"{post_id}.txt"
    if tiktok_file.exists():
        text = tiktok_file.read_text(encoding="utf-8").strip()
        if text:
            return text

    return ""


def deduplicate(raw_videos: list[dict]) -> list[dict]:
    """Deduplicate by post_id."""
    seen = set()
    deduped = []
    for v in raw_videos:
        pid = str(v.get("post_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            deduped.append(v)
    return deduped


def classify_backlash(videos: list[dict]) -> list[dict]:
    """Use LLM to classify each video as backlash or not (concurrent, key-pooled)."""
    clients = get_llm_clients()
    n_keys = len(clients)
    n_workers = max(1, n_keys * WORKERS_PER_KEY)
    model = clients[0][1]

    print(f"  Using model: {model}")
    print(f"  Keys: {n_keys}, workers: {n_workers} ({WORKERS_PER_KEY} per key)")
    print(f"  Videos to classify: {len(videos)}")

    # Resume from progress
    already_done = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            already_done = json.load(f)
        print(f"  Resuming: {len(already_done)} already classified")

    # Build pending list — load transcripts and skip already-done
    pending = []
    for v in videos:
        pid = str(v.get("post_id", ""))
        v["_transcript"] = get_transcript(pid)
        if pid in already_done:
            v["_backlash"] = already_done[pid]
            continue
        pending.append(v)
    print(f"  Pending: {len(pending)} (skipping {len(videos) - len(pending)} already classified)")

    state_lock = threading.Lock()
    counters = {"api_calls": 0, "errors": 0, "completed": 0}
    start = time.time()

    def classify_one(idx_v):
        idx, v = idx_v
        pid = str(v.get("post_id", ""))
        client, model_ = clients[idx % n_keys]

        description = (v.get("description", "") or "")[:500]
        transcript_text = (v.get("_transcript", "") or "")[:1500]
        prompt = BACKLASH_PROMPT.format(description=description, transcript=transcript_text)

        try:
            result = llm_call(client, model_, prompt, max_tokens=10).upper().strip()
            if result not in ("YES", "NO"):
                if "YES" in result:
                    result = "YES"
                elif "NO" in result:
                    result = "NO"
                else:
                    result = "UNKNOWN"
            with state_lock:
                counters["api_calls"] += 1
        except Exception as e:
            with state_lock:
                counters["errors"] += 1
            print(f"    Error on {pid}: {e}")
            result = "UNKNOWN"

        v["_backlash"] = result
        with state_lock:
            already_done[pid] = result
            counters["completed"] += 1
            done = counters["completed"]
            if done % 50 == 0 or done == len(pending):
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(already_done, f)
                yes_count = sum(1 for t in already_done.values() if t == "YES")
                no_count = sum(1 for t in already_done.values() if t == "NO")
                rate = done / max(time.time() - start, 1e-9)
                eta = (len(pending) - done) / rate if rate > 0 else float("inf")
                print(
                    f"    [{done}/{len(pending)}] {rate:.1f}/s | "
                    f"YES: {yes_count} NO: {no_count} | "
                    f"errors: {counters['errors']} | "
                    f"ETA: {eta/60:.1f}m"
                )
        return v

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(classify_one, (i, v)) for i, v in enumerate(pending)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"    Worker exception: {e}")

    # Final save
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(already_done, f)

    elapsed = int(time.time() - start)
    print(f"\n  Done in {elapsed//60}m {elapsed%60}s. API calls: {counters['api_calls']}, errors: {counters['errors']}")
    topic_counts = Counter(v.get("_backlash", "UNKNOWN") for v in videos)
    for t, c in topic_counts.most_common():
        print(f"    {t}: {c} ({c/len(videos)*100:.1f}%)")

    return videos


def save_output(videos: list[dict]):
    """Save final filtered CSV."""
    fieldnames = [
        "post_id", "url", "description", "create_time",
        "digg_count", "share_count", "collect_count", "comment_count",
        "play_count", "video_duration", "hashtags",
        "profile_username", "profile_followers", "is_verified", "region",
        "_search_keyword", "_backlash", "_transcript",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in videos:
            row = {**v}
            if isinstance(row.get("hashtags"), list):
                row["hashtags"] = ", ".join(row["hashtags"])
            writer.writerow(row)

    print(f"\nSaved {len(videos)} videos to {OUTPUT_CSV}")


def main():
    # Load raw data
    print("Loading raw data...")
    with open(RAW_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  Raw videos: {len(raw)}")

    # Deduplicate
    videos = deduplicate(raw)
    print(f"  After dedup: {len(videos)} unique videos")

    # LLM backlash classification
    print("\n=== LLM classification (backlash vs neutral) ===")
    classified = classify_backlash(videos)

    save_output(classified)

    # Summary
    yes_vids = [v for v in classified if v.get("_backlash") == "YES"]
    no_vids = [v for v in classified if v.get("_backlash") == "NO"]
    yes_views = sum(int(v.get("play_count", 0) or 0) for v in yes_vids)
    no_views = sum(int(v.get("play_count", 0) or 0) for v in no_vids)
    print(f"\n=== FINAL SUMMARY ===")
    print(f"  BACKLASH (YES): {len(yes_vids)} videos, {yes_views:,} views")
    print(f"  NOT BACKLASH (NO): {len(no_vids)} videos, {no_views:,} views")


if __name__ == "__main__":
    main()
