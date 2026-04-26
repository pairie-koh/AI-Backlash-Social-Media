"""
Classify TikTok AI backlash videos by theme.

For videos already classified as backlash (_backlash=YES), determine the
primary and secondary backlash themes.

Themes:
  - JOBS: Job displacement, automation, layoffs, unemployment
  - ENVIRONMENT: Data centers, energy, water, carbon, climate
  - CREATIVE: Art theft, AI art, music, writing, voice cloning
  - PRIVACY: Surveillance, tracking, facial recognition, data collection
  - EDUCATION: Cheating, plagiarism, deskilling, academic integrity
  - SAFETY: Existential risk, alignment, AI out of control, regulation
  - MISINFO: Deepfakes, misinformation, fake news, manipulation
  - GENERAL: General anti-AI sentiment, tech backlash, AI bubble/hype

Usage:
    set OPENROUTER_API_KEY=...
    python scripts/classify_backlash_theme.py

    # Preview without API calls:
    python scripts/classify_backlash_theme.py --dry-run
"""

import argparse
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
FILTERED_CSV = DATA_DIR / "tiktok_ai_backlash_filtered.csv"
WHISPER_DIR = DATA_DIR / "whisper_transcripts"
TIKTOK_DIR = DATA_DIR / "tiktok_transcripts"
PROGRESS_FILE = DATA_DIR / "backlash_theme_progress.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")

VALID_THEMES = ["JOBS", "ENVIRONMENT", "CREATIVE", "PRIVACY", "EDUCATION", "SAFETY", "MISINFO", "GENERAL"]

CLASSIFY_PROMPT = """A TikTok video expresses criticism or concern about AI. Classify the PRIMARY THEME of the backlash.

Pick the single BEST category:

JOBS — Job displacement, automation replacing workers, AI layoffs, unemployment fears, career anxiety
Examples: "AI is taking our jobs", "My company replaced 50 people with AI", "Which jobs will AI kill?"

ENVIRONMENT — AI's environmental impact: data center energy/water usage, carbon footprint, power grid strain
Examples: "AI data centers are draining our water", "The carbon cost of ChatGPT", "AI is destroying the planet"

CREATIVE — AI threatening creative work: art theft, AI-generated art/music/writing, voice cloning, copyright
Examples: "AI is stealing artists' work", "Stop AI art", "They cloned my voice with AI"

PRIVACY — AI surveillance, tracking, facial recognition, data collection, AI spying
Examples: "AI is watching everything you do", "Facial recognition is out of control", "AI knows too much about you"

EDUCATION — AI in schools: cheating, plagiarism, ChatGPT bans, academic integrity, deskilling students
Examples: "Students are using ChatGPT to cheat", "AI is ruining education", "Schools should ban AI"

SAFETY — Existential risk, AI alignment, regulation calls, AI dangerous, loss of control
Examples: "AI could destroy humanity", "We need to regulate AI now", "AI is out of control"

MISINFO — Deepfakes, AI misinformation, fake news, AI scams, AI manipulation
Examples: "AI deepfakes are ruining democracy", "You can't trust anything anymore", "AI scam calls"

GENERAL — General anti-AI sentiment, tech backlash, AI bubble, AI overhyped, doesn't fit above
Examples: "AI is ruining everything", "The AI bubble will burst", "I hate AI"

Respond with exactly one word from: JOBS, ENVIRONMENT, CREATIVE, PRIVACY, EDUCATION, SAFETY, MISINFO, GENERAL

---
Description: {description}
Transcript: {transcript}"""


def load_backlash_videos():
    videos = []
    with open(FILTERED_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("_backlash", "").upper() == "YES":
                videos.append(row)
    return videos


def get_transcript(post_id, csv_transcript=""):
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
    return csv_transcript


def get_client():
    from openai import OpenAI
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY")
        sys.exit(1)
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


def classify(client, description, transcript):
    prompt = CLASSIFY_PROMPT.format(
        description=description[:500],
        transcript=transcript[:1500],
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip().upper()
    if text in VALID_THEMES:
        return text
    # Try to find a valid theme in the response
    for theme in VALID_THEMES:
        if theme in text:
            return theme
    return "GENERAL"


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would be classified, no API calls")
    args = parser.parse_args()

    videos = load_backlash_videos()
    print(f"Loaded {len(videos)} backlash videos\n")

    # Load progress
    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"Already classified: {len(progress)}")

    remaining = [v for v in videos if v.get("post_id", "") not in progress]
    print(f"Remaining: {len(remaining)}\n")

    if args.dry_run:
        print("[DRY RUN] Would classify these videos:")
        for v in remaining[:5]:
            desc = (v.get("description", "") or "")[:80].replace("\n", " ")
            print(f"  {v.get('post_id', '')}: {desc}...")
        if len(remaining) > 5:
            print(f"  ... and {len(remaining) - 5} more")
        print(f"\nEstimated API calls: {len(remaining)}")
        return

    client = get_client()
    print(f"Model: {LLM_MODEL}")
    print(f"Classifying {len(remaining)} videos...\n")

    api_calls = 0
    errors = 0

    for i, v in enumerate(remaining):
        pid = v.get("post_id", "")
        description = v.get("description", "") or ""
        transcript = get_transcript(pid, v.get("_transcript", ""))

        try:
            result = classify(client, description, transcript)
            progress[pid] = result
            api_calls += 1
        except Exception as e:
            print(f"  Error on {pid}: {e}")
            progress[pid] = "ERROR"
            errors += 1

        if (api_calls + errors) % 25 == 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f)
            counts = Counter(progress.values())
            print(f"  [{i+1}/{len(remaining)}] calls: {api_calls} | {dict(counts.most_common())}")

        time.sleep(0.15)

    # Final save
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f)

    # Results
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}\n")

    counts = Counter(progress.values())
    n = len(progress)
    print(f"Theme distribution ({n} backlash videos):")
    print(f"  {'Theme':<15} {'Count':>6} {'%':>7}")
    print(f"  {'-'*15} {'-'*6} {'-'*7}")
    for theme in VALID_THEMES + ["ERROR"]:
        c = counts.get(theme, 0)
        if c > 0:
            print(f"  {theme:<15} {c:>6} {c/n*100:>6.1f}%")

    # Engagement by theme
    print(f"\nEngagement by theme:")
    for theme in VALID_THEMES:
        theme_vids = [v for v in videos if progress.get(v.get("post_id", "")) == theme]
        if not theme_vids:
            continue
        views = [safe_int(v.get("play_count")) for v in theme_vids]
        import statistics
        print(f"  {theme}:")
        print(f"    Videos:      {len(theme_vids)}")
        print(f"    Total views: {sum(views):,}")
        if views:
            print(f"    Median views:{statistics.median(views):,.0f}")


if __name__ == "__main__":
    main()
