"""
Filter YouTube videos for AI backlash using LLM classification.

Deduplicates raw Bright Data output, then uses an LLM to classify each
video as expressing AI backlash (YES) or not (NO).

Usage:
    export OPENROUTER_API_KEY=...
    python youtube/scripts/filter_by_topic.py
"""

import csv
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_JSON = DATA_DIR / "raw" / "youtube_raw.json"
OUTPUT_CSV = DATA_DIR / "youtube_ai_backlash_filtered.csv"
PROGRESS_FILE = DATA_DIR / "youtube_filter_llm_progress.json"

# --- API setup ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")

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


def get_llm_client():
    from openai import OpenAI

    if OPENROUTER_API_KEY:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY), LLM_MODEL
    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), "gpt-4o-mini"
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY), "claude-haiku-4-20250414"
        except ImportError:
            pass
    print("ERROR: Set OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
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


def classify_backlash(videos):
    """Use LLM to classify each video as backlash or not."""
    client, model = get_llm_client()
    print(f"  Using model: {model}")
    print(f"  Videos to classify: {len(videos)}")

    already_done = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            already_done = json.load(f)
        print(f"  Resuming: {len(already_done)} already classified")

    api_calls = 0
    errors = 0

    for i, v in enumerate(videos):
        vid = v.get("video_id", "")

        # Extract transcript for this video
        transcript = get_transcript(v)
        v["_transcript_text"] = transcript

        if vid in already_done:
            v["_backlash"] = already_done[vid]
            continue

        title = (v.get("title", "") or "")[:300]
        description = (v.get("description", "") or "")[:500]
        transcript_text = transcript[:1500]

        prompt = BACKLASH_PROMPT.format(
            title=title, description=description, transcript=transcript_text,
        )

        try:
            result = llm_call(client, model, prompt, max_tokens=10).upper().strip()
            if result not in ("YES", "NO"):
                if "YES" in result:
                    result = "YES"
                elif "NO" in result:
                    result = "NO"
                else:
                    result = "UNKNOWN"
            api_calls += 1
        except Exception as e:
            print(f"    Error on {vid}: {e}")
            result = "UNKNOWN"
            errors += 1

        v["_backlash"] = result
        already_done[vid] = result

        if (api_calls + errors) % 25 == 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(already_done, f)
            yes_count = sum(1 for t in already_done.values() if t == "YES")
            no_count = sum(1 for t in already_done.values() if t == "NO")
            print(f"    [{i+1}/{len(videos)}] API calls: {api_calls} | YES: {yes_count}, NO: {no_count}")

        time.sleep(0.15)

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(already_done, f)

    print(f"\n  Done. API calls: {api_calls}, errors: {errors}")
    topic_counts = Counter(v.get("_backlash", "UNKNOWN") for v in videos)
    for t, c in topic_counts.most_common():
        print(f"    {t}: {c} ({c/len(videos)*100:.1f}%)")

    return videos


def save_output(videos):
    fieldnames = [
        "video_id", "url", "title", "description", "date_posted",
        "likes", "views", "num_comments", "video_length",
        "youtuber", "channel_url", "subscribers", "verified",
        "_search_keyword", "_backlash", "_transcript_text",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
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

    print(f"\nSaved {len(videos)} videos to {OUTPUT_CSV}")


def main():
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
    yes_views = sum(int(v.get("views", 0) or 0) for v in yes_vids)
    no_views = sum(int(v.get("views", 0) or 0) for v in no_vids)
    print(f"\n=== FINAL SUMMARY ===")
    print(f"  BACKLASH (YES): {len(yes_vids)} videos, {yes_views:,} views")
    print(f"  NOT BACKLASH (NO): {len(no_vids)} videos, {no_views:,} views")


if __name__ == "__main__":
    main()
