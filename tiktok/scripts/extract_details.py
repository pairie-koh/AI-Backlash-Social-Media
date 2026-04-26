"""
Extract structured details from TikTok AI backlash videos.

For each backlash video, extracts:
  - backlash_themes: list of themes (jobs, environment, creative, privacy, etc.)
  - specific_concerns: list of specific issues raised
  - companies_mentioned: list of AI companies/products criticized
  - proposed_actions: list of actions called for (regulate, ban, boycott, etc.)
  - sentiment_intensity: LOW, MEDIUM, HIGH (how strongly negative)

Uses OpenRouter API (Claude Sonnet).
Saves progress incrementally to allow resuming.

Usage:
    export OPENROUTER_API_KEY=...
    python tiktok/scripts/extract_details.py
"""

import csv
import io
import json
import os
import sys
import time
from pathlib import Path

import requests

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILTERED_CSV = DATA_DIR / "tiktok_ai_backlash_filtered.csv"
WHISPER_DIR = DATA_DIR / "whisper_transcripts"
TIKTOK_DIR = DATA_DIR / "tiktok_transcripts"
PROGRESS_FILE = DATA_DIR / "detail_extraction_progress.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "anthropic/claude-sonnet-4"

EXTRACT_PROMPT = """You are extracting structured data about AI backlash from a TikTok video.

Given the video's description and transcript, extract:

1. backlash_themes: Which backlash themes are present? Pick ALL that apply from:
   - "jobs" (job displacement, automation, layoffs, unemployment)
   - "environment" (data centers, energy, water, carbon footprint, climate)
   - "creative" (art theft, AI art/music/writing, voice cloning, copyright)
   - "privacy" (surveillance, tracking, facial recognition, data collection)
   - "education" (cheating, plagiarism, academic integrity, deskilling)
   - "safety" (existential risk, alignment, AI dangerous, loss of control)
   - "misinfo" (deepfakes, misinformation, fake news, scams)
   - "regulation" (calls to regulate, ban, slow down AI)
   - "labor_rights" (workers' rights, unionization against AI, fair compensation)
   - "general" (broad anti-AI sentiment that doesn't fit specific categories)

2. specific_concerns: List each specific concern or criticism raised.
   E.g., "AI will replace graphic designers", "ChatGPT uses too much water",
   "Students can't learn if they use AI for essays", "AI deepfakes of politicians"

3. companies_mentioned: List every AI company, product, or model mentioned.
   Use common names (e.g., "OpenAI", "ChatGPT", "Midjourney", "Google", "Meta", "Microsoft Copilot").

4. proposed_actions: List any actions the creator advocates for.
   E.g., "ban AI in schools", "regulate AI companies", "boycott AI art",
   "tax AI companies", "require AI labeling", "support human artists"

5. sentiment_intensity: How strongly negative is the video?
   - "LOW" — mild concern, questioning, nuanced criticism
   - "MEDIUM" — clear criticism, worry, frustration
   - "HIGH" — strong anger, fear, outrage, call to action

Respond with ONLY valid JSON, no markdown fences:
{{"backlash_themes": [...], "specific_concerns": [...], "companies_mentioned": [...], "proposed_actions": [...], "sentiment_intensity": "..."}}

If a field has no matches, use an empty list [].

---
Description: {description}
Transcript: {transcript}"""


def load_backlash_videos() -> list[dict]:
    """Load only _backlash=YES videos from the filtered CSV."""
    videos = []
    with open(FILTERED_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("_backlash", "").upper() == "YES":
                videos.append(row)
    return videos


def get_transcript(post_id: str, csv_transcript: str = "") -> str:
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


def llm_extract(description: str, transcript: str) -> dict:
    prompt = EXTRACT_PROMPT.format(
        description=description[:500],
        transcript=transcript[:2000],
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    return json.loads(text)


def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    print("Loading backlash videos (_backlash=YES)...")
    videos = load_backlash_videos()
    print(f"  Loaded {len(videos)} videos")

    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"  Resuming: {len(progress)} already extracted")

    remaining = [v for v in videos if v.get("post_id", "") not in progress]
    print(f"  Remaining: {len(remaining)}\n")

    api_calls = 0
    errors = 0

    for i, v in enumerate(remaining):
        pid = v.get("post_id", "")
        description = v.get("description", "") or ""
        transcript = get_transcript(pid, v.get("_transcript", ""))

        if not transcript and not description:
            progress[pid] = {
                "backlash_themes": [],
                "specific_concerns": [],
                "companies_mentioned": [],
                "proposed_actions": [],
                "sentiment_intensity": "LOW",
            }
            continue

        try:
            result = llm_extract(description, transcript)

            for key in ["backlash_themes", "specific_concerns", "companies_mentioned", "proposed_actions"]:
                if key not in result or not isinstance(result[key], list):
                    result[key] = []
            if result.get("sentiment_intensity") not in ("LOW", "MEDIUM", "HIGH"):
                result["sentiment_intensity"] = "MEDIUM"

            progress[pid] = result
            api_calls += 1

        except Exception as e:
            print(f"  Error on {pid}: {e}")
            progress[pid] = {
                "backlash_themes": ["error"],
                "specific_concerns": [],
                "companies_mentioned": [],
                "proposed_actions": [],
                "sentiment_intensity": "MEDIUM",
                "_error": str(e),
            }
            errors += 1
            time.sleep(2)

        if (api_calls + errors) % 25 == 0 and (api_calls + errors) > 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
            print(f"  [{i+1}/{len(remaining)}] API calls: {api_calls}, errors: {errors}")

        time.sleep(0.3)

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

    print(f"\nDone. API calls: {api_calls}, errors: {errors}")
    print(f"Total extracted: {len(progress)}")

    # Summary
    from collections import Counter

    theme_counter = Counter()
    company_counter = Counter()
    sentiment_counter = Counter()
    for data in progress.values():
        for t in data.get("backlash_themes", []):
            theme_counter[t] += 1
        for c in data.get("companies_mentioned", []):
            company_counter[c] += 1
        sentiment_counter[data.get("sentiment_intensity", "MEDIUM")] += 1

    print(f"\nBacklash theme distribution:")
    for t, count in theme_counter.most_common():
        print(f"  {t}: {count}")

    print(f"\nTop 15 companies/products mentioned:")
    for c, count in company_counter.most_common(15):
        print(f"  {c}: {count}")

    print(f"\nSentiment intensity:")
    for s, count in sentiment_counter.most_common():
        print(f"  {s}: {count}")


if __name__ == "__main__":
    main()
