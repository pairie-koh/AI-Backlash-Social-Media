"""
Two-pass filter for TikTok AI backlash videos.

Pass 1 (keyword, free): Keep only videos that mention AI-related terms
        in description, hashtags, or transcript.

Pass 2 (LLM, cheap): For each survivor, ask whether the video expresses
        criticism, concern, or backlash about AI.

Usage:
    # Set one of these:
    export OPENROUTER_API_KEY=...
    export ANTHROPIC_API_KEY=...

    # Run both passes:
    python scripts/filter_by_topic.py

    # Run only keyword pass (no LLM needed):
    python scripts/filter_by_topic.py --keyword-only
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_JSON = DATA_DIR / "raw" / "tiktok_raw.json"
WHISPER_DIR = DATA_DIR / "whisper_transcripts"
TIKTOK_DIR = DATA_DIR / "tiktok_transcripts"
OUTPUT_CSV = DATA_DIR / "tiktok_ai_backlash_filtered.csv"
PROGRESS_FILE = DATA_DIR / "filter_llm_progress.json"

# --- API setup ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")

# --- AI backlash keyword patterns (case-insensitive) ---
# These catch the broad AI topic; the LLM pass then filters for backlash specifically.
AI_PATTERNS = [
    # Core AI terms
    r"\bAI\b",
    r"artificial\s+intelligence",
    # Major products / companies
    r"ChatGPT",
    r"chat\s+GPT",
    r"OpenAI",
    r"open\s+AI",
    r"Midjourney",
    r"mid\s+journey",
    r"DALL[\-\s]?E",
    r"Copilot",
    r"Stable\s+Diffusion",
    r"Google\s+AI",
    # Job displacement
    r"AI\s+taking\s+jobs",
    r"AI\s+replacing",
    r"AI\s+layoffs",
    r"AI\s+killed\s+my\s+career",
    r"AI\s+take\s+my\s+job",
    r"AI\s+replace\s+me",
    r"AI\s+job\s+loss",
    r"AI\s+automation",
    # Environment
    r"AI\s+data\s+center",
    r"AI\s+energy",
    r"AI\s+water\s+usage",
    r"AI\s+carbon",
    r"AI\s+climate",
    r"AI\s+power\s+grid",
    # Creative theft
    r"AI\s+art\s+theft",
    r"AI\s+stealing",
    r"stop\s+AI\s+art",
    r"AI\s+music\s+theft",
    r"AI\s+voice\s+cloning",
    r"AI\s+vs\s+artists",
    r"AI\s+copyright",
    r"protect\s+artists",
    # Privacy
    r"AI\s+watching\s+me",
    r"AI\s+spying",
    r"AI\s+facial\s+recognition",
    r"AI\s+surveillance",
    r"AI\s+tracking",
    # Education
    r"AI\s+cheating",
    r"ban\s+ChatGPT",
    r"AI\s+ruining\s+education",
    r"AI\s+plagiarism",
    # Safety
    r"AI\s+dangerous",
    r"ban\s+AI",
    r"AI\s+out\s+of\s+control",
    r"AI\s+regulation",
    r"stop\s+AI\b",
    # Deepfakes & misinformation
    r"AI\s+deepfake",
    r"deepfake\s+danger",
    r"AI\s+scam",
    r"AI\s+fake\s+news",
    r"AI\s+misinformation",
    # Slop & quality degradation
    r"AI\s+slop",
    r"AI\s+spam",
    r"AI\s+garbage",
    r"AI\s+making\s+internet\s+worse",
    r"AI\s+ruined\s+search",
    r"AI\s+ruining\s+everything",
    r"AI\s+generated\s+content",
    # Everyday frustration & company backlash
    r"AI\s+customer\s+service",
    r"can.t\s+talk\s+to\s+a?\s*human",
    r"ChatGPT\s+getting\s+worse",
    r"OpenAI\s+bad",
    r"Copilot\s+sucks",
    r"AI\s+chatbot",
    r"AI\s+overrated",
    r"AI\s+bubble",
    # Anti-AI sentiment & affirmative resistance
    r"anti[\-\s]?AI",
    r"#noAI",
    r"#antiAI",
    r"#SupportHumanArtists",
    r"human\s+made\s+not\s+AI",
    r"#AIart",
    r"no\s+AI\s+used",
    r"real\s+artist",
    r"AI\s+overhyped",
    r"quit\s+AI",
    r"hate\s+AI",
]
AI_RE = re.compile("|".join(AI_PATTERNS), re.IGNORECASE)

# --- LLM prompt for pass 2 ---
BACKLASH_PROMPT = """A TikTok video mentions AI or AI-related technology. Does this video express criticism, concern, or backlash about AI?

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


def get_llm_client():
    """Get an OpenAI-compatible client."""
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


def find_ai_terms(text: str) -> list[str]:
    """Return AI-related terms found in text."""
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
            terms.add("Stable_Diffusion")
        elif "copilot" in m_clean:
            terms.add("Copilot")
        elif "artificial intelligence" in m_clean:
            terms.add("AI")
        elif "deepfake" in m_clean:
            terms.add("deepfake")
        elif "slop" in m_clean:
            terms.add("AI_slop")
        elif "spam" in m_clean:
            terms.add("AI_spam")
        elif "garbage" in m_clean:
            terms.add("AI_garbage")
        elif "customer service" in m_clean:
            terms.add("AI_customer_service")
        elif "human" in m_clean and ("talk" in m_clean or "made" in m_clean):
            terms.add("human_vs_AI")
        elif "supporthumanartists" in m_clean:
            terms.add("human_artists")
        elif "antiai" in m_clean or "noai" in m_clean or "anti ai" in m_clean:
            terms.add("anti_AI")
        elif "surveillance" in m_clean:
            terms.add("AI_surveillance")
        elif "tracking" in m_clean:
            terms.add("AI_tracking")
        elif "plagiarism" in m_clean:
            terms.add("AI_plagiarism")
        elif "misinformation" in m_clean:
            terms.add("AI_misinformation")
        elif "copyright" in m_clean:
            terms.add("AI_copyright")
        elif "protect" in m_clean and "artist" in m_clean:
            terms.add("protect_artists")
        elif "vs" in m_clean and "artist" in m_clean:
            terms.add("AI_vs_artists")
        elif "climate" in m_clean:
            terms.add("AI_climate")
        elif "power grid" in m_clean:
            terms.add("AI_power_grid")
        elif "automation" in m_clean:
            terms.add("AI_automation")
        elif "job loss" in m_clean:
            terms.add("AI_job_loss")
        elif "chatbot" in m_clean:
            terms.add("AI_chatbot")
        elif "overrated" in m_clean:
            terms.add("AI_overrated")
        elif "overhyped" in m_clean:
            terms.add("AI_overhyped")
        elif "bubble" in m_clean:
            terms.add("AI_bubble")
        elif "ruining everything" in m_clean:
            terms.add("AI_ruining_everything")
        elif "generated content" in m_clean:
            terms.add("AI_generated_content")
        elif "#aiart" in m_clean:
            terms.add("AI_art_hashtag")
        elif "no ai used" in m_clean:
            terms.add("no_AI_used")
        elif "real artist" in m_clean:
            terms.add("real_artist")
        elif "quit" in m_clean:
            terms.add("quit_AI")
        elif "hate" in m_clean:
            terms.add("hate_AI")
        elif "stop ai" in m_clean:
            terms.add("stop_AI")
        elif "google" in m_clean:
            terms.add("Google_AI")
        else:
            terms.add(m_clean.replace(" ", "_"))
    return sorted(terms)


def pass1_keyword_filter(raw_videos: list[dict]) -> list[dict]:
    """Pass 1: Keep only videos that mention AI-related terms."""
    # Deduplicate
    seen = set()
    deduped = []
    for v in raw_videos:
        pid = str(v.get("post_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            deduped.append(v)
    print(f"  After dedup: {len(deduped)} unique videos")

    matched = []
    no_transcript = 0

    for v in deduped:
        pid = str(v.get("post_id", ""))
        description = v.get("description", "") or ""
        hashtags_raw = v.get("hashtags", [])
        hashtags = " ".join(hashtags_raw) if isinstance(hashtags_raw, list) else str(hashtags_raw)

        transcript = get_transcript(pid)
        if not transcript:
            no_transcript += 1

        # Check each source
        found_in = []
        if find_ai_terms(description):
            found_in.append("description")
        if find_ai_terms(hashtags):
            found_in.append("hashtags")
        if find_ai_terms(transcript):
            found_in.append("transcript")

        all_text = f"{description} {hashtags} {transcript}"
        ai_terms = find_ai_terms(all_text)

        if ai_terms:
            v["_ai_terms"] = ", ".join(ai_terms)
            v["_match_source"] = ", ".join(found_in)
            v["_transcript"] = transcript
            matched.append(v)

    print(f"  No transcript available: {no_transcript}")
    print(f"  Pass 1 result: {len(matched)} mention AI-related terms")

    # Breakdown
    from collections import Counter
    term_counts = Counter()
    for v in matched:
        for t in v["_ai_terms"].split(", "):
            term_counts[t] += 1
    for t, c in term_counts.most_common(15):
        print(f"    {t}: {c}")

    return matched


def pass2_llm_classify(videos: list[dict]) -> list[dict]:
    """Pass 2: Use LLM to classify each video as backlash or not."""
    client, model = get_llm_client()
    print(f"  Using model: {model}")
    print(f"  Videos to classify: {len(videos)}")

    # Resume from progress
    already_done = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            already_done = json.load(f)
        print(f"  Resuming: {len(already_done)} already classified")

    api_calls = 0
    errors = 0

    for i, v in enumerate(videos):
        pid = str(v.get("post_id", ""))

        if pid in already_done:
            v["_backlash"] = already_done[pid]
            continue

        description = (v.get("description", "") or "")[:500]
        transcript = (v.get("_transcript", "") or "")[:1500]

        prompt = BACKLASH_PROMPT.format(
            description=description,
            transcript=transcript,
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
            print(f"    Error on {pid}: {e}")
            result = "UNKNOWN"
            errors += 1

        v["_backlash"] = result
        already_done[pid] = result

        # Save progress every 25
        if (api_calls + errors) % 25 == 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(already_done, f)
            yes_count = sum(1 for t in already_done.values() if t == "YES")
            no_count = sum(1 for t in already_done.values() if t == "NO")
            print(f"    [{i+1}/{len(videos)}] API calls: {api_calls} | YES: {yes_count}, NO: {no_count}")

        time.sleep(0.15)

    # Final save
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(already_done, f)

    print(f"\n  Pass 2 complete. API calls: {api_calls}, errors: {errors}")
    from collections import Counter
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
        "_search_keyword", "_ai_terms", "_match_source", "_backlash", "_transcript",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword-only", action="store_true",
                        help="Run only the keyword pass (no LLM)")
    args = parser.parse_args()

    # Load raw data
    print("Loading raw data...")
    with open(RAW_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  Raw videos: {len(raw)}")

    # Pass 1: keyword filter
    print("\n=== PASS 1: Keyword filter (AI-related terms) ===")
    survivors = pass1_keyword_filter(raw)

    if args.keyword_only:
        for v in survivors:
            v["_backlash"] = ""
        save_output(survivors)
        return

    # Pass 2: LLM backlash classification
    print("\n=== PASS 2: LLM classification (backlash vs neutral) ===")
    classified = pass2_llm_classify(survivors)

    save_output(classified)

    # Summary
    yes_vids = [v for v in classified if v.get("_backlash") == "YES"]
    no_vids = [v for v in classified if v.get("_backlash") == "NO"]
    total_views = sum(int(v.get("play_count", 0) or 0) for v in classified)
    yes_views = sum(int(v.get("play_count", 0) or 0) for v in yes_vids)
    no_views = sum(int(v.get("play_count", 0) or 0) for v in no_vids)
    print(f"\n=== FINAL SUMMARY ===")
    print(f"  BACKLASH (YES): {len(yes_vids)} videos, {yes_views:,} views")
    print(f"  NOT BACKLASH (NO): {len(no_vids)} videos, {no_views:,} views")


if __name__ == "__main__":
    main()
