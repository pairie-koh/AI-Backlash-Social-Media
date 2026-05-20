"""
Combined register + theme classifier for the positive YES set.

Reads a filtered CSV (output of filter_by_topic.py with --polarity positive),
processes only _positive == YES rows, and assigns two orthogonal labels per video:

  - theme       : what the positive content is about (career_productivity /
                  companion_affective / creative_tool / education_learning /
                  model_stan / synthetic_media_play / breakthrough_science /
                  general_hype)
  - register    : who is talking and why (brand_official / creator_promo /
                  genuine_sentiment / news_explainer / creative_showcase)

Plus secondary_themes (optional) and an evidence quote.

Writes a new CSV alongside the input with the original columns + classification.
Resume-safe via progress JSON.

Usage:
    export OPENROUTER_API_KEY=...   # or rely on .openrouter_keys file
    python scripts/classify_register_theme.py --input tiktok/data/tiktok_positive_positive_filtered.csv
    python scripts/classify_register_theme.py --input youtube/data/youtube_positive_positive_filtered.csv
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

csv.field_size_limit(2**31 - 1)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
KEYS_FILE = REPO_ROOT / ".openrouter_keys"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEYS = os.environ.get("OPENROUTER_API_KEYS", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")
WORKERS_PER_KEY = int(os.environ.get("WORKERS_PER_KEY", "3"))

DESC_CHARS = int(os.environ.get("DESC_CHARS", "1500"))
TRANSCRIPT_MAX = int(os.environ.get("TRANSCRIPT_MAX", "4000"))

VALID_THEMES = {
    "career_productivity", "companion_affective", "creative_tool",
    "education_learning", "model_stan", "synthetic_media_play",
    "breakthrough_science", "general_hype",
}
VALID_REGISTERS = {
    "brand_official", "creator_promo", "genuine_sentiment",
    "news_explainer", "creative_showcase",
}

PROMPT = """You are classifying a {platform} video that has been verified as expressing positive sentiment about AI. Assign two orthogonal labels:

THEME — what the positive content is about. Pick exactly one PRIMARY theme; optionally add SECONDARY themes if substantively addressed:
- career_productivity: career wins, vibe coding, side hustle, "AI helped my career", productivity, agent demos (Manus, Operator, Claude Computer Use)
- companion_affective: AI girlfriend/boyfriend, character.ai, "Claude is my therapist", AI bestie, parasocial AI bonding
- creative_tool: using AI creative tools to make art/music/video (Midjourney, Suno, Sora, Runway, DALL-E, Stable Diffusion), "made art with AI"
- education_learning: study help, AI tutor, "ChatGPT helped me learn", homework help, academic wins
- model_stan: model praise / capability marvel ("Claude is goated", "GPT cooked", "Sonnet ate", "this is insane", "AI mind blown")
- synthetic_media_play: funny AI videos, AI lip sync, AI animation, deepfake-as-entertainment, viral AI edits where the AI use is the joke/art
- breakthrough_science: medical AI, scientific AI breakthroughs, "AI cures X", lab/research news, AI saving lives
- general_hype: broad pro-AI sentiment without a specific named theme ("AI is the future", "I love AI", "AI is amazing")

REGISTER — who is talking and why. Pick exactly one:
- brand_official: this is an AI company's own channel promoting their product (e.g. @suno, @googledeepmind, @Midjourney, @CharacterAIOfficial, @cursor_ai, @Samsung promoting their own AI features)
- creator_promo: creator selling/promoting AI tools, affiliate marketing, "10 AI tools that will replace your job", course-selling, sponsored AI content, paid promotion of someone else's AI product
- genuine_sentiment: first-person organic personal experience or affective expression. Real person talking about how AI helped them, their own use, their reaction. NOT trying to sell anything.
- news_explainer: mainstream news / explainer / educational content describing AI events without strong personal affect ("GPT-5 was released today", "What is vibe coding")
- creative_showcase: someone using AI to make creative content (deepfake parody, AI music cover, AI animation) where the content IS the point — affect is implicit through the creation, not the message

Rules:
- Both labels are required. If unsure, prefer the closest fit over guessing.
- Use creator_promo (not genuine_sentiment) for any video where the creator is selling, affiliating, course-pushing, or pitching a tool — even if framed as personal experience.
- Use brand_official only when the channel IS the company / product being promoted (i.e. @<company> posting about their own thing).
- Use creative_showcase for videos where AI is the medium and the content (joke/art/song) IS the point — not where the creator is reacting to AI itself.
- The PRIMARY theme is the most central frame; secondary themes can be added if multiple are clearly present.

Output a JSON object on a single line with this EXACT schema and nothing else:
{{"theme": "<theme>", "secondary_themes": ["<theme>", ...], "register": "<register>", "evidence": "<short quote ≤80 chars from description or transcript>"}}

---
Channel: {channel}
Title: {title}
Description: {description}
Transcript: {transcript}
"""


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
    sys.exit("ERROR: Set OPENROUTER_API_KEYS / OPENROUTER_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY")


def llm_call(client, model, prompt, max_tokens=200):
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


def parse_response(raw):
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\"theme\".*?\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def detect_platform(input_path):
    name = input_path.name.lower()
    if "tiktok" in name:
        return "tiktok"
    if "youtube" in name:
        return "youtube"
    sys.exit(f"ERROR: cannot detect platform from filename {name}")


def get_field_config(platform):
    if platform == "tiktok":
        return {
            "id_field": "post_id",
            "title_field": None,
            "transcript_field": "_transcript",
            "channel_field": "profile_username",
        }
    return {
        "id_field": "video_id",
        "title_field": "title",
        "transcript_field": "_transcript_text",
        "channel_field": "youtuber",
    }


def derive_paths(input_path):
    """tiktok_positive_positive_filtered.csv -> tiktok_positive_classified.csv (de-dup the _positive)."""
    stem = input_path.stem
    if stem.endswith("_filtered"):
        stem = stem[:-len("_filtered")]
    stem = stem.replace("_positive_positive", "_positive")
    return (
        input_path.parent / f"{stem}_classified.csv",
        input_path.parent / f"{stem}_classify_progress.json",
    )


def main():
    parser = argparse.ArgumentParser(description="Combined register + theme classifier for positive YES set.")
    parser.add_argument("--input", required=True, help="Filtered CSV from filter_by_topic.py")
    parser.add_argument("--limit", type=int, default=None, help="Test on N rows only")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        candidate = REPO_ROOT / input_path
        input_path = candidate if candidate.exists() else input_path
    if not input_path.exists():
        sys.exit(f"ERROR: input file not found at {input_path}")

    platform = detect_platform(input_path)
    fields = get_field_config(platform)
    output_csv, progress_file = derive_paths(input_path)

    print(f"Platform     : {platform}")
    print(f"Input        : {input_path}")
    print(f"Output       : {output_csv}")
    print(f"Progress file: {progress_file}")
    print()

    print("Loading filtered CSV...")
    yes_rows = []
    all_fieldnames = None
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_fieldnames = list(reader.fieldnames)
        for row in reader:
            if row.get("_positive") == "YES":
                yes_rows.append(row)
    print(f"  Total YES rows: {len(yes_rows)}")

    if args.limit:
        yes_rows = yes_rows[:args.limit]
        print(f"  Limit applied -> {len(yes_rows)} rows")

    already_done = {}
    if progress_file.exists():
        with open(progress_file, encoding="utf-8") as f:
            already_done = json.load(f)
        print(f"  Resuming: {len(already_done)} already classified")

    pending = []
    for r in yes_rows:
        rid = r.get(fields["id_field"], "")
        if rid in already_done:
            continue
        pending.append(r)
    print(f"  Pending: {len(pending)}")

    clients = get_llm_clients()
    n_keys = len(clients)
    n_workers = max(1, n_keys * WORKERS_PER_KEY)
    model = clients[0][1]

    print(f"  Model: {model}")
    print(f"  Keys: {n_keys}, workers: {n_workers}")
    print()

    state_lock = threading.Lock()
    counters = {"api_calls": 0, "errors": 0, "completed": 0, "parse_fail": 0}
    start = time.time()

    def classify_one(idx_r):
        idx, r = idx_r
        rid = r.get(fields["id_field"], "")
        client, model_ = clients[idx % n_keys]

        title = ""
        if fields["title_field"]:
            title = (r.get(fields["title_field"], "") or "")[:300]
        description = (r.get("description", "") or "")[:DESC_CHARS]
        transcript_text = (r.get(fields["transcript_field"], "") or "")[:TRANSCRIPT_MAX]
        channel = (r.get(fields["channel_field"], "") or "")[:60]

        prompt = PROMPT.format(
            platform=platform.capitalize(),
            channel=channel,
            title=title,
            description=description,
            transcript=transcript_text,
        )

        result = None
        try:
            raw = llm_call(client, model_, prompt, max_tokens=200)
            with state_lock:
                counters["api_calls"] += 1
            parsed = parse_response(raw)
            if parsed:
                theme = parsed.get("theme", "")
                if theme in VALID_THEMES:
                    secondary = [t for t in (parsed.get("secondary_themes", []) or []) if t in VALID_THEMES]
                    register = parsed.get("register", "")
                    if register not in VALID_REGISTERS:
                        register = "unknown"
                    result = {
                        "theme": theme,
                        "secondary_themes": secondary,
                        "register": register,
                        "evidence": (parsed.get("evidence", "") or "")[:120],
                    }
            if not result:
                with state_lock:
                    counters["parse_fail"] += 1
                result = {"theme": "unknown", "secondary_themes": [], "register": "unknown", "evidence": ""}
        except Exception as e:
            with state_lock:
                counters["errors"] += 1
            print(f"    Error on {rid}: {e}")
            result = {"theme": "unknown", "secondary_themes": [], "register": "unknown", "evidence": ""}

        with state_lock:
            already_done[rid] = result
            counters["completed"] += 1
            done = counters["completed"]
            if done % 50 == 0 or done == len(pending):
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(already_done, f)
                themes_count = Counter(v["theme"] for v in already_done.values())
                regs_count = Counter(v["register"] for v in already_done.values())
                rate = done / max(time.time() - start, 1e-9)
                eta = (len(pending) - done) / rate if rate > 0 else float("inf")
                top_theme = themes_count.most_common(1)[0] if themes_count else ("none", 0)
                top_reg = regs_count.most_common(1)[0] if regs_count else ("none", 0)
                print(
                    f"    [{done}/{len(pending)}] {rate:.1f}/s | "
                    f"top theme: {top_theme[0]}={top_theme[1]} | "
                    f"top register: {top_reg[0]}={top_reg[1]} | "
                    f"errors: {counters['errors']} parse_fail: {counters['parse_fail']} | "
                    f"ETA: {eta/60:.1f}m"
                )

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(classify_one, (i, r)) for i, r in enumerate(pending)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"    Worker exception: {e}")

    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(already_done, f)

    elapsed = int(time.time() - start)
    print(f"\n  Done in {elapsed//60}m {elapsed%60}s. API: {counters['api_calls']}, errors: {counters['errors']}, parse_fail: {counters['parse_fail']}")

    print(f"\nWriting classified CSV to {output_csv}...")
    new_fieldnames = all_fieldnames + ["_theme", "_secondary_themes", "_register", "_evidence"]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in yes_rows:
            rid = r.get(fields["id_field"], "")
            cls = already_done.get(rid, {"theme": "unknown", "secondary_themes": [], "register": "unknown", "evidence": ""})
            r["_theme"] = cls["theme"]
            r["_secondary_themes"] = json.dumps(cls["secondary_themes"])
            r["_register"] = cls["register"]
            r["_evidence"] = cls["evidence"]
            writer.writerow(r)

    themes_count = Counter(already_done[r.get(fields["id_field"], "")]["theme"]
                           for r in yes_rows if r.get(fields["id_field"], "") in already_done)
    regs_count = Counter(already_done[r.get(fields["id_field"], "")]["register"]
                         for r in yes_rows if r.get(fields["id_field"], "") in already_done)
    print(f"\n=== THEME DISTRIBUTION ===")
    for t, c in themes_count.most_common():
        print(f"  {t:<25} {c:>6} ({c/len(yes_rows)*100:5.1f}%)")
    print(f"\n=== REGISTER DISTRIBUTION ===")
    for rg, c in regs_count.most_common():
        print(f"  {rg:<25} {c:>6} ({c/len(yes_rows)*100:5.1f}%)")


if __name__ == "__main__":
    main()
