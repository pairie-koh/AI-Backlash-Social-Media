"""
Filter YouTube videos for AI sentiment using LLM classification.

Deduplicates raw Bright Data output, then uses an LLM to classify each
video as expressing the requested polarity (backlash or positive sentiment).

Usage:
    export OPENROUTER_API_KEY=...
    python youtube/scripts/filter_by_topic.py                                       # backlash on default raw
    python youtube/scripts/filter_by_topic.py --raw data/raw/youtube_positive_raw.json --polarity positive
"""

import argparse
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
DEFAULT_RAW_JSON = DATA_DIR / "raw" / "youtube_raw.json"
KEYS_FILE = REPO_ROOT / ".openrouter_keys"

# --- API setup ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEYS = os.environ.get("OPENROUTER_API_KEYS", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")
WORKERS_PER_KEY = int(os.environ.get("WORKERS_PER_KEY", "3"))

POSITIVE_PROMPT = """A YouTube video was found by searching for AI-related keywords. Does this video express enthusiasm, gratitude, advocacy, or positive sentiment about AI?

YES — The video expresses positive feelings, personal benefit, or pro-AI advocacy. This includes:
- Personal stories about AI helping career, productivity, or life
- "AI changed my life" / "AI helped me cope" / "I'm cooked without AI"
- Vibe coding / built-something-with-Claude / Cursor changed my life
- Agent-demo content (Manus, Operator, Claude Computer Use)
- Model-stan praise (Claude is goated, GPT cooked, Sonnet ate, Sora is insane)
- Genuine sentiment toward AI companions (AI girlfriend, character.ai, "Claude is my therapist")
- Education wins (ChatGPT helped me study, AI tutor, AI got me through finals)
- Creative-tool celebration (Midjourney art, Suno music, Runway AI, made art with AI)
- "AI is the future" / "AI is the real deal" / "AI underrated" / "I love AI"
- Affiliate / promotional / tutorial content selling AI tools (count as YES — flag in `register` later)
- Pro-AI hashtags and identity content (#proAI, #AIenthusiast, team AI)
- AI side-hustle / monetization wins
- Synthetic-media creative use (funny AI video, AI lip sync, AI animation, AI face swap)

NO — The video mentions AI but is NOT expressing positive sentiment. This includes:
- Criticism, concern, or backlash about AI (jobs, art theft, environment, deepfakes, slop, regulation)
- Neutral news reporting without affect
- Mixed / ambivalent takes that lean negative or neutral
- Off-topic content where AI is incidental
- Anti-AI advocacy or fear content
- Frustration with AI products

Respond with exactly one word: YES or NO

---
Title: {title}
Description: {description}
Transcript: {transcript}"""


BACKLASH_PROMPT = """A YouTube video was found by searching for AI-related keywords. Does this video express criticism, concern, or backlash about AI?

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
Title: {title}
Description: {description}
Transcript: {transcript}"""


def _load_openrouter_keys():
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


def llm_call(client, model, prompt, max_tokens=10):
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


def get_transcript(video):
    """Extract transcript from YouTube video data."""
    ft = video.get("formatted_transcript")
    if ft and isinstance(ft, list) and len(ft) > 0:
        parts = []
        for item in ft:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or "")
            elif isinstance(item, str):
                parts.append(item)
        text = " ".join(parts).strip()
        if text:
            return text

    t = video.get("transcript", "")
    if t and isinstance(t, str) and len(t) > 10:
        return t.strip()

    return ""


def deduplicate(raw_videos):
    """Deduplicate by video_id."""
    seen = set()
    deduped = []
    for v in raw_videos:
        vid = v.get("video_id", "")
        if vid and vid not in seen:
            seen.add(vid)
            deduped.append(v)
    return deduped


def classify_videos(videos, prompt_template, column_name, progress_file):
    """Use LLM to classify each video YES/NO for the given prompt (concurrent, key-pooled)."""
    clients = get_llm_clients()
    n_keys = len(clients)
    n_workers = max(1, n_keys * WORKERS_PER_KEY)
    model = clients[0][1]

    print(f"  Using model: {model}")
    print(f"  Keys: {n_keys}, workers: {n_workers} ({WORKERS_PER_KEY} per key)")
    print(f"  Videos to classify: {len(videos)}")
    print(f"  Progress file: {progress_file}")
    print(f"  Output column: {column_name}")

    already_done = {}
    if progress_file.exists():
        with open(progress_file, encoding="utf-8") as f:
            already_done = json.load(f)
        print(f"  Resuming: {len(already_done)} already classified")

    pending = []
    for v in videos:
        vid = v.get("video_id", "")
        v["_transcript_text"] = get_transcript(v)
        if vid in already_done:
            v[column_name] = already_done[vid]
            continue
        pending.append(v)
    print(f"  Pending: {len(pending)} (skipping {len(videos) - len(pending)} already classified)")

    state_lock = threading.Lock()
    counters = {"api_calls": 0, "errors": 0, "completed": 0}
    start = time.time()

    def classify_one(idx_v):
        idx, v = idx_v
        vid = v.get("video_id", "")
        client, model_ = clients[idx % n_keys]

        title = (v.get("title", "") or "")[:300]
        description = (v.get("description", "") or "")[:500]
        transcript_text = (v.get("_transcript_text", "") or "")[:1500]
        prompt = prompt_template.format(
            title=title, description=description, transcript=transcript_text,
        )

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
            print(f"    Error on {vid}: {e}")
            result = "UNKNOWN"

        v[column_name] = result
        with state_lock:
            already_done[vid] = result
            counters["completed"] += 1
            done = counters["completed"]
            if done % 50 == 0 or done == len(pending):
                with open(progress_file, "w", encoding="utf-8") as f:
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

    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(already_done, f)

    elapsed = int(time.time() - start)
    print(f"\n  Done in {elapsed//60}m {elapsed%60}s. API calls: {counters['api_calls']}, errors: {counters['errors']}")
    topic_counts = Counter(v.get(column_name, "UNKNOWN") for v in videos)
    for t, c in topic_counts.most_common():
        print(f"    {t}: {c} ({c/len(videos)*100:.1f}%)")

    return videos


def save_output(videos, output_csv, column_name):
    fieldnames = [
        "video_id", "url", "title", "description", "date_posted",
        "likes", "views", "num_comments", "video_length",
        "youtuber", "channel_url", "subscribers", "verified",
        "_search_keyword", column_name, "_transcript_text",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in videos:
            row = {**v}
            if isinstance(row.get("hashtags"), list):
                parts = []
                for h in row["hashtags"]:
                    if isinstance(h, dict):
                        parts.append(h.get("hashtag", "") or h.get("name", ""))
                    else:
                        parts.append(str(h))
                row["hashtags"] = ", ".join(parts)
            writer.writerow(row)

    print(f"\nSaved {len(videos)} videos to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Filter YouTube videos by AI sentiment polarity.")
    parser.add_argument(
        "--raw", default=str(DEFAULT_RAW_JSON),
        help="Path to raw Bright Data JSON (default: youtube_raw.json).",
    )
    parser.add_argument(
        "--polarity", choices=["positive", "negative"], default="negative",
        help="Sentiment to classify for: 'negative' = backlash (default), 'positive' = enthusiasm.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_path = Path(args.raw)
    if not raw_path.is_absolute():
        candidate = REPO_ROOT / raw_path
        raw_path = candidate if candidate.exists() else (DATA_DIR / raw_path)
    if not raw_path.exists():
        sys.exit(f"ERROR: raw file not found at {raw_path}")

    if args.polarity == "positive":
        prompt_template = POSITIVE_PROMPT
        column_name = "_positive"
        stem = raw_path.stem.replace("_raw", "")
        output_csv = DATA_DIR / f"{stem}_positive_filtered.csv"
        progress_file = DATA_DIR / f"{stem}_positive_filter_progress.json"
    else:
        prompt_template = BACKLASH_PROMPT
        column_name = "_backlash"
        stem = raw_path.stem.replace("_raw", "")
        if stem == "youtube":
            output_csv = DATA_DIR / "youtube_ai_backlash_filtered.csv"
            progress_file = DATA_DIR / "youtube_filter_llm_progress.json"
        else:
            output_csv = DATA_DIR / f"{stem}_backlash_filtered.csv"
            progress_file = DATA_DIR / f"{stem}_backlash_filter_progress.json"

    print(f"Raw input    : {raw_path}")
    print(f"Polarity     : {args.polarity}")
    print(f"Output CSV   : {output_csv}")
    print(f"Progress file: {progress_file}")
    print()

    print("Loading raw data...")
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  Raw videos: {len(raw)}")

    videos = deduplicate(raw)
    print(f"  After dedup: {len(videos)} unique videos")

    print(f"\n=== LLM classification ({args.polarity} vs not) ===")
    classified = classify_videos(videos, prompt_template, column_name, progress_file)

    save_output(classified, output_csv, column_name)

    yes_vids = [v for v in classified if v.get(column_name) == "YES"]
    no_vids = [v for v in classified if v.get(column_name) == "NO"]
    yes_views = sum(int(v.get("views", 0) or 0) for v in yes_vids)
    no_views = sum(int(v.get("views", 0) or 0) for v in no_vids)
    label = "POSITIVE" if args.polarity == "positive" else "BACKLASH"
    print(f"\n=== FINAL SUMMARY ===")
    print(f"  {label} (YES): {len(yes_vids)} videos, {yes_views:,} views")
    print(f"  NOT {label} (NO): {len(no_vids)} videos, {no_views:,} views")


if __name__ == "__main__":
    main()
