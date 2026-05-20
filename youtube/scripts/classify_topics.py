"""Multi-label topic classification on final-YES YouTube videos.

For each row where _backlash_final == YES, calls an LLM to assign:
  - exactly one PRIMARY topic
  - zero or more SECONDARY topics
from the 8-category fixed taxonomy.

Writes new columns _topic_primary and _topics_v3 (JSON list) back to the
final CSV. Resume-safe via a progress file.

Reuses get_clients / llm_call from verify_yes_strict.

Usage:
    LLM_PROVIDER=openai LLM_MODEL=gpt-5-mini python youtube/scripts/classify_topics.py
    LLM_PROVIDER=openai LLM_MODEL=gpt-5-mini python youtube/scripts/classify_topics.py --limit 30
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_yes_strict import get_clients, llm_call, WORKERS_PER_KEY  # noqa

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "youtube_ai_backlash_final.csv"
OUTPUT_CSV = DATA_DIR / "youtube_ai_backlash_final.csv"
PROGRESS_FILE = DATA_DIR / "youtube_topic_classify_progress.json"

HEAD_CHARS = int(os.environ.get("HEAD_CHARS", "3500"))
TAIL_CHARS = int(os.environ.get("TAIL_CHARS", "1500"))
DESC_CHARS = int(os.environ.get("DESC_CHARS", "1000"))

VALID_TOPICS = {"jobs", "art", "environment", "deepfake", "surveillance", "education", "regulation", "xrisk", "general_neg"}

TOPIC_PROMPT = """You are classifying a YouTube video that has been verified as expressing some form of AI backlash. Assign one or more topic tags describing what the backlash is about, from this fixed taxonomy:

- jobs: job displacement, layoffs, AI replacing workers, "AI taking jobs", career fears
- art: art / music / writing theft, AI training on creative work without consent, anti-AI artist movement, "no AI", "human made", "real artist"
- environment: AI energy use, water consumption, carbon footprint, data center build-out concerns
- deepfake: deepfakes, AI-generated misinformation, AI fake news, AI scams using synthetic identities/voices
- surveillance: AI surveillance, facial recognition, AI tracking, AI-enabled privacy violations
- education: AI cheating in school, AI deskilling students, AI ruining learning
- regulation: explicit calls to regulate, ban, slow, or pause AI; AI policy advocacy that is NOT primarily framed around catastrophe/extinction (use xrisk for that)
- xrisk: AI existential risk, alignment failures, loss of human control, AGI/superintelligence danger, AI causing human extinction or civilizational catastrophe, "AI will kill us all", "AI doom", "Skynet", "paperclip", "rogue AI", treaties to avert AI catastrophe, AI safety in the loss-of-control sense
- general_neg: general anti-AI sentiment without a specific named grievance ("I hate AI", "AI ruining everything", "AI slop", "AI is overhyped"). Do NOT use this for doom/extinction framing — that is xrisk.

Rules:
- Multi-label: a video can have several topics if multiple grievances are substantively addressed.
- The PRIMARY topic is the one the video is mainly about. Exactly one primary.
- SECONDARY topics are substantively addressed but not the main focus. Can be empty.
- Only assign a topic if the actual content addresses it. Don't tag based on keyword/title hints alone.
- Use "general_neg" as primary only when the video is broadly anti-AI without a specific named grievance, OR when general anti-AI is the explicit main frame. Prefer a specific topic when one fits.

Output: a JSON object on a single line with this EXACT schema and nothing else:
{{"primary": "<topic>", "secondary": ["<topic>", ...], "evidence": "<one short quote or phrase justifying the primary topic, ≤80 chars>"}}

---
Title: {title}
Description: {description}
Transcript: {transcript}
"""


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
    m = re.search(r"\{.*?\"primary\".*?\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def sample_transcript(text):
    text = text or ""
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    return text[:HEAD_CHARS] + "\n[...middle cut...]\n" + text[-TAIL_CHARS:]


def classify(rows_to_check, limit=None):
    if limit:
        rows_to_check = rows_to_check[:limit]
    clients = get_clients()
    n_keys = len(clients)
    n_workers = max(1, n_keys * WORKERS_PER_KEY)
    model = clients[0][1]

    print(f"  Model: {model}")
    print(f"  Keys: {n_keys}, workers: {n_workers}")
    print(f"  Rows to classify: {len(rows_to_check)}")

    already_done = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            already_done = json.load(f)
        print(f"  Resuming: {len(already_done)} already done")

    pending = [r for r in rows_to_check if r.get("video_id") not in already_done]
    for r in rows_to_check:
        vid = r.get("video_id", "")
        if vid in already_done:
            entry = already_done[vid]
            r["_topics_v3"] = json.dumps(entry.get("topics", []))
            r["_topic_primary"] = entry.get("primary", "")
    print(f"  Pending: {len(pending)}")

    state_lock = threading.Lock()
    counters = {"api_calls": 0, "errors": 0, "completed": 0, "parse_errors": 0}
    start = time.time()

    def classify_one(idx_v):
        idx, v = idx_v
        vid = v.get("video_id", "")
        client, model_ = clients[idx % n_keys]

        title = (v.get("title") or "")[:300]
        description = (v.get("description") or "")[:DESC_CHARS]
        transcript = sample_transcript(v.get("_transcript_text") or "")
        prompt = TOPIC_PROMPT.format(title=title, description=description, transcript=transcript)

        result = None
        for attempt in range(3):
            try:
                raw = llm_call(client, model_, prompt, max_tokens=300)
                result = parse_response(raw)
                with state_lock:
                    counters["api_calls"] += 1
                if result is None:
                    with state_lock:
                        counters["parse_errors"] += 1
                break
            except Exception as e:
                err = str(e)[:120]
                if attempt == 2:
                    with state_lock:
                        counters["errors"] += 1
                    print(f"    Error on {vid}: {err}")
                else:
                    time.sleep(1.5 * (attempt + 1))

        if result and result.get("primary") in VALID_TOPICS:
            primary = result["primary"]
            secondary = [t for t in (result.get("secondary") or []) if t in VALID_TOPICS and t != primary]
            topics = [primary] + secondary
            evidence = (result.get("evidence") or "")[:200]
        else:
            primary = ""
            topics = []
            evidence = ""

        v["_topics_v3"] = json.dumps(topics)
        v["_topic_primary"] = primary

        with state_lock:
            already_done[vid] = {"primary": primary, "topics": topics, "evidence": evidence}
            counters["completed"] += 1
            done = counters["completed"]
            if done % 100 == 0 or done == len(pending):
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(already_done, f)
                rate = done / max(time.time() - start, 1e-9)
                eta = (len(pending) - done) / rate if rate > 0 else 0
                print(f"    [{done}/{len(pending)}] {rate:.1f}/s | parse_err: {counters['parse_errors']} | err: {counters['errors']} | ETA {eta/60:.1f}m")

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(classify_one, (i, v)) for i, v in enumerate(pending)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"    Worker exception: {e}")

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(already_done, f)

    elapsed = int(time.time() - start)
    print(f"\n  Done in {elapsed//60}m {elapsed%60}s. API calls: {counters['api_calls']}, errors: {counters['errors']}, parse errors: {counters['parse_errors']}")


def save_output(all_rows):
    fields = list(all_rows[0].keys())
    if "_topic_primary" not in fields:
        pos = fields.index("_transcript_text") if "_transcript_text" in fields else len(fields)
        fields.insert(pos, "_topic_primary")
        fields.insert(pos + 1, "_topics_v3")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            r.setdefault("_topic_primary", "")
            r.setdefault("_topics_v3", "")
            writer.writerow(r)
    print(f"\nSaved {len(all_rows)} rows back to {OUTPUT_CSV}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, help="Process only first N rows (smoke test)")
    args = p.parse_args()

    print(f"Loading {INPUT_CSV.name}...")
    all_rows = list(csv.DictReader(INPUT_CSV.open("r", encoding="utf-8", newline="")))
    yes_rows = [r for r in all_rows if (r.get("_backlash_final") or "").upper() == "YES"]
    print(f"  total rows: {len(all_rows)}  final-YES rows: {len(yes_rows)}")

    print("\n=== Topic classification ===")
    classify(yes_rows, limit=args.limit)

    save_output(all_rows)

    primary_counts = Counter(r.get("_topic_primary", "") for r in all_rows if r.get("_topic_primary"))
    print("\n=== PRIMARY TOPIC DISTRIBUTION (final-YES) ===")
    n_classified = sum(primary_counts.values())
    for t, c in primary_counts.most_common():
        print(f"  {t:>15}  {c:>5}  ({c/max(n_classified,1)*100:>4.1f}%)")


if __name__ == "__main__":
    main()
