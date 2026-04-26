"""
Two-pass filter for YouTube AI backlash videos.

Pass 1 (keyword, free): Keep only videos that mention AI-related terms
        in title, description, hashtags, or transcript.

Pass 2 (LLM, cheap): For each survivor, ask whether the video expresses
        criticism, concern, or backlash about AI.

Usage:
    export OPENROUTER_API_KEY=...
    python youtube/scripts/filter_by_topic.py

    python youtube/scripts/filter_by_topic.py --keyword-only
"""

import argparse
import csv
import io
import json
import os
import re
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

# --- AI keyword patterns (case-insensitive) ---
AI_PATTERNS = [
    r"\bAI\b",
    r"artificial\s+intelligence",
    r"machine\s+learning",
    r"deep\s+learning",
    r"ChatGPT",
    r"chat\s+GPT",
    r"OpenAI",
    r"open\s+AI",
    r"Midjourney",
    r"mid\s+journey",
    r"DALL[\-\s]?E",
    r"Copilot",
    r"Gemini\s+AI",
    r"Claude\s+AI",
    r"Sora\s+AI",
    r"Stable\s+Diffusion",
    r"AI\s+taking\s+jobs",
    r"AI\s+replacing",
    r"AI\s+layoffs",
    r"AI\s+data\s+center",
    r"AI\s+energy",
    r"AI\s+art\s+theft",
    r"AI\s+stealing",
    r"AI\s+cheating",
    r"AI\s+surveillance",
    r"AI\s+deepfake",
    r"AI\s+dangerous",
    r"ban\s+AI",
    r"stop\s+AI",
    r"anti[\-\s]?AI",
    r"#noAI",
    r"#antiAI",
]
AI_RE = re.compile("|".join(AI_PATTERNS), re.IGNORECASE)

# --- LLM prompt for pass 2 ---
BACKLASH_PROMPT = """A YouTube video mentions AI or AI-related technology. Does this video express criticism, concern, or backlash about AI?

YES — The video expresses negativity, worry, or criticism about AI. This includes:
- Fear of job displacement or automation replacing workers
- Concern about AI's environmental impact (energy, water, data centers, carbon)
- Criticism of AI-generated art, music, or writing (art theft, creative displacement)
- Privacy or surveillance concerns related to AI
- Deepfake, misinformation, or manipulation worries
- AI in education concerns (cheating, plagiarism, deskilling)
- Calls to regulate, ban, or slow down AI
- Existential risk, safety, or alignment concerns
- General anti-AI sentiment or tech backlash
- Personal stories about being negatively affected by AI
- Criticism of AI companies (OpenAI, Google, Meta, etc.)

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


def find_ai_terms(text):
    matches = AI_RE.findall(text)
    terms = set()
    for m in matches:
        m_clean = m.strip().lower()
        if m_clean in ("ai",):
            terms.add("AI")
        elif "chatgpt" in m_clean or "chat gpt" in m_clean:
            terms.add("ChatGPT")
        elif "openai" in m_clean or "open ai" in m_clean:
            terms.add("OpenAI")
        elif "midjourney" in m_clean or "mid journey" in m_clean:
            terms.add("Midjourney")
        elif "dall" in m_clean:
            terms.add("DALL-E")
        elif "stable diffusion" in m_clean:
            terms.add("Stable Diffusion")
        elif "artificial intelligence" in m_clean:
            terms.add("AI")
        else:
            terms.add(m_clean.replace(" ", "_"))
    return sorted(terms)


def pass1_keyword_filter(raw_videos):
    seen = set()
    deduped = []
    for v in raw_videos:
        vid = v.get("video_id", "")
        if vid and vid not in seen:
            seen.add(vid)
            deduped.append(v)
    print(f"  After dedup: {len(deduped)} unique videos")

    matched = []
    no_transcript = 0

    for v in deduped:
        title = v.get("title", "") or ""
        description = v.get("description", "") or ""
        hashtags_raw = v.get("hashtags", [])
        if isinstance(hashtags_raw, list):
            parts = []
            for h in hashtags_raw:
                if isinstance(h, dict):
                    parts.append(h.get("hashtag", "") or h.get("name", ""))
                else:
                    parts.append(str(h))
            hashtags = " ".join(parts)
        else:
            hashtags = str(hashtags_raw or "")
        transcript = get_transcript(v)

        if not transcript:
            no_transcript += 1

        found_in = []
        if find_ai_terms(title):
            found_in.append("title")
        if find_ai_terms(description):
            found_in.append("description")
        if find_ai_terms(hashtags):
            found_in.append("hashtags")
        if find_ai_terms(transcript):
            found_in.append("transcript")

        all_text = f"{title} {description} {hashtags} {transcript}"
        ai_terms = find_ai_terms(all_text)

        if ai_terms:
            v["_ai_terms"] = ", ".join(ai_terms)
            v["_match_source"] = ", ".join(found_in)
            v["_transcript_text"] = transcript
            matched.append(v)

    print(f"  No transcript available: {no_transcript}")
    print(f"  Pass 1 result: {len(matched)} mention AI-related terms")

    term_counts = Counter()
    for v in matched:
        for t in v["_ai_terms"].split(", "):
            term_counts[t] += 1
    for t, c in term_counts.most_common(15):
        print(f"    {t}: {c}")

    return matched


def pass2_llm_classify(videos):
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

        if vid in already_done:
            v["_backlash"] = already_done[vid]
            continue

        title = (v.get("title", "") or "")[:300]
        description = (v.get("description", "") or "")[:500]
        transcript = (v.get("_transcript_text", "") or "")[:1500]

        prompt = BACKLASH_PROMPT.format(
            title=title, description=description, transcript=transcript,
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

    print(f"\n  Pass 2 complete. API calls: {api_calls}, errors: {errors}")
    topic_counts = Counter(v.get("_backlash", "UNKNOWN") for v in videos)
    for t, c in topic_counts.most_common():
        print(f"    {t}: {c} ({c/len(videos)*100:.1f}%)")

    return videos


def save_output(videos):
    fieldnames = [
        "video_id", "url", "title", "description", "date_posted",
        "likes", "views", "num_comments", "video_length",
        "youtuber", "channel_url", "subscribers", "verified",
        "_search_keyword", "_ai_terms", "_match_source", "_backlash",
        "_transcript_text",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword-only", action="store_true",
                        help="Run only the keyword pass (no LLM)")
    args = parser.parse_args()

    print("Loading raw data...")
    with open(RAW_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  Raw videos: {len(raw)}")

    print("\n=== PASS 1: Keyword filter (AI-related terms) ===")
    survivors = pass1_keyword_filter(raw)

    if args.keyword_only:
        for v in survivors:
            v["_backlash"] = ""
        save_output(survivors)
        return

    print("\n=== PASS 2: LLM classification (backlash vs neutral) ===")
    classified = pass2_llm_classify(survivors)

    save_output(classified)

    yes_vids = [v for v in classified if v.get("_backlash") == "YES"]
    no_vids = [v for v in classified if v.get("_backlash") == "NO"]
    yes_views = sum(int(v.get("views", 0) or 0) for v in yes_vids)
    no_views = sum(int(v.get("views", 0) or 0) for v in no_vids)
    print(f"\n=== FINAL SUMMARY ===")
    print(f"  BACKLASH (YES): {len(yes_vids)} videos, {yes_views:,} views")
    print(f"  NOT BACKLASH (NO): {len(no_vids)} videos, {no_views:,} views")


if __name__ == "__main__":
    main()
