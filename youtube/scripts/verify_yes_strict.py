"""
Second-pass verifier for YES-classified videos.

Reads youtube_ai_backlash_filtered.csv, re-classifies only the rows where
_backlash == YES with a stricter prompt that targets the false-positive
patterns found in the audit (marketing tags, clickbait pro-AI content,
tangential AI mentions, etc.). Writes a new CSV with the v2 label.

Usage:
    python youtube/scripts/verify_yes_strict.py
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

csv.field_size_limit(2**31 - 1)

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "youtube_ai_backlash_filtered.csv"
OUTPUT_CSV = DATA_DIR / "youtube_ai_backlash_verified.csv"
PROGRESS_FILE = DATA_DIR / "youtube_verify_v2_progress.json"
KEYS_FILE = REPO_ROOT / ".openrouter_keys"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEYS = os.environ.get("OPENROUTER_API_KEYS", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")
WORKERS_PER_KEY = int(os.environ.get("WORKERS_PER_KEY", "3"))

VERIFY_PROMPT = """A first-pass classifier flagged this YouTube video as expressing AI backlash. Verify whether that label is correct.

Answer YES only if the video genuinely expresses criticism, concern, fear, or negativity about AI as its actual subject or stance — backed by what the creator says, not just keywords in the title.

Valid YES (real backlash):
- Fear of job displacement, layoffs, or automation replacing workers (and the creator agrees this is bad)
- Concern about AI's environmental impact (energy, water, carbon)
- Criticism of AI-generated art, music, writing — art theft, training on artists without consent
- Privacy, surveillance, deepfake, misinformation, or manipulation worries
- AI in education concerns (cheating, deskilling) framed critically
- Calls to regulate, ban, or slow AI; existential / safety / alignment concerns
- Complaints about AI slop, AI making products worse, AI ruining search/internet
- Anti-AI sentiment, "no AI used" as a deliberate stance against AI in art
- Personal stories of being negatively affected by AI

Answer NO (false positive — common patterns to reject):
- "No AI" / "#noAI" used as a marketing/quality badge in non-AI content (longevity videos, vintage music compilations, recipes, wellness, nature footage) where AI is not the topic
- Pro-AI hype, marketing, or product demos with clickbait scary titles ("AI is replacing workers!" but content says AI is your friendly helper)
- Tutorials, reviews, or how-to content using AI tools, even if the title sounds skeptical
- Investment / stock content asking "AI bubble or boom?" that ultimately recommends investing in AI
- AI-generated music, videos, or art uploaded by AI hobbyists (the creator is using AI, not criticizing it)
- Game AI, NPC AI, or fictional robot AI in entertainment (Minecraft mods, FNaF analysis, sci-fi)
- Videos where "AI" appears in title/description but the actual content is about something unrelated (counterfeit gadgets, unrelated news, music videos that matched a keyword in lyrics)
- Neutral interviews or reporting that mentions concerns without endorsing them
- Sponsored content for AI products, even if framed as "honest review"
- Videos that briefly mention AI worry as a small part of a broader topic

Decision rule: if the creator's overall stance is neutral, positive, or off-topic, answer NO — even if the title contains "danger", "ruin", "stop", "hate", "replace", "destroy", etc. The keyword in the title is not enough.

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


def load_yes_rows():
    """Load all rows from the v1 CSV; return (all_rows, yes_indices)."""
    all_rows = []
    yes_idx = []
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            all_rows.append(row)
            if (row.get("_backlash") or "").strip().upper() == "YES":
                yes_idx.append(i)
    return all_rows, yes_idx


def verify_yes(all_rows, yes_idx):
    clients = get_llm_clients()
    n_keys = len(clients)
    n_workers = max(1, n_keys * WORKERS_PER_KEY)
    model = clients[0][1]

    print(f"  Using model: {model}")
    print(f"  Keys: {n_keys}, workers: {n_workers} ({WORKERS_PER_KEY} per key)")
    print(f"  YES rows to verify: {len(yes_idx)}")

    already_done = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            already_done = json.load(f)
        print(f"  Resuming: {len(already_done)} already verified")

    pending = []
    for i in yes_idx:
        vid = all_rows[i].get("video_id", "")
        if vid in already_done:
            all_rows[i]["_backlash_v2"] = already_done[vid]
            continue
        pending.append(i)
    print(f"  Pending: {len(pending)} (skipping {len(yes_idx) - len(pending)} cached)")

    state_lock = threading.Lock()
    counters = {"api_calls": 0, "errors": 0, "completed": 0}
    start = time.time()

    def verify_one(work):
        idx_in_pending, row_idx = work
        v = all_rows[row_idx]
        vid = v.get("video_id", "")
        client, model_ = clients[idx_in_pending % n_keys]

        title = (v.get("title", "") or "")[:300]
        description = (v.get("description", "") or "")[:500]
        transcript_text = (v.get("_transcript_text", "") or "")[:1500]
        prompt = VERIFY_PROMPT.format(
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

        v["_backlash_v2"] = result
        with state_lock:
            already_done[vid] = result
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
                    f"confirmed YES: {yes_count} demoted NO: {no_count} | "
                    f"errors: {counters['errors']} | "
                    f"ETA: {eta/60:.1f}m"
                )

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(verify_one, (i, idx)) for i, idx in enumerate(pending)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"    Worker exception: {e}")

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(already_done, f)

    elapsed = int(time.time() - start)
    print(f"\n  Done in {elapsed//60}m {elapsed%60}s. API calls: {counters['api_calls']}, errors: {counters['errors']}")


def save_output(all_rows):
    fieldnames = [
        "video_id", "url", "title", "description", "date_posted",
        "likes", "views", "num_comments", "video_length",
        "youtuber", "channel_url", "subscribers", "verified",
        "_search_keyword", "_backlash", "_backlash_v2", "_transcript_text",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            row.setdefault("_backlash_v2", "")
            writer.writerow(row)
    print(f"\nSaved {len(all_rows)} rows to {OUTPUT_CSV}")


def main():
    print("Loading v1 filtered CSV...")
    all_rows, yes_idx = load_yes_rows()
    print(f"  Total rows: {len(all_rows)}  YES rows: {len(yes_idx)}")

    print("\n=== Strict v2 verification of YES rows ===")
    verify_yes(all_rows, yes_idx)

    save_output(all_rows)

    confirmed = sum(1 for r in all_rows if r.get("_backlash_v2") == "YES")
    demoted = sum(1 for r in all_rows if r.get("_backlash_v2") == "NO")
    unknown = sum(1 for r in all_rows if r.get("_backlash_v2") == "UNKNOWN")

    def to_int(x):
        try: return int(x or 0)
        except: return 0

    confirmed_views = sum(to_int(r.get("views")) for r in all_rows if r.get("_backlash_v2") == "YES")
    demoted_views = sum(to_int(r.get("views")) for r in all_rows if r.get("_backlash_v2") == "NO")

    print("\n=== VERIFICATION SUMMARY ===")
    print(f"  Confirmed BACKLASH: {confirmed} videos, {confirmed_views:,} views")
    print(f"  Demoted (was YES, now NO): {demoted} videos, {demoted_views:,} views")
    print(f"  Unknown / errors: {unknown}")
    if (confirmed + demoted) > 0:
        print(f"  V1 precision estimate: {confirmed / (confirmed + demoted) * 100:.1f}%")


if __name__ == "__main__":
    main()
