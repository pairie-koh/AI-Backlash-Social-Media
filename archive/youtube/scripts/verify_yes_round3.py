"""Round 3 — final precision pass on V2 YES rows (YouTube).

Re-checks every _backlash_v2 == YES row using head + middle + tail of the
transcript instead of [:1500]. V1 and V2 both saw only ~20% of the typical
YouTube transcript, which let clickbait-pivot videos through (intro sounds
critical, outro pivots to product/investment promotion).

Reuses VERIFY_PROMPT, parse_response, and client setup from verify_yes_strict.
Only the input shape changes.

Usage:
    python youtube/scripts/verify_yes_round3.py
    python youtube/scripts/verify_yes_round3.py --limit 50
"""

import argparse
import csv
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

csv.field_size_limit(2**31 - 1)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_yes_strict import (  # noqa: E402
    VERIFY_PROMPT,
    parse_response,
    get_clients,
    llm_call,
    WORKERS_PER_KEY,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "youtube_ai_backlash_verified.csv"
OUTPUT_CSV = DATA_DIR / "youtube_ai_backlash_round3.csv"
PROGRESS_FILE = DATA_DIR / "youtube_round3_progress.json"

HEAD_CHARS = int(os.environ.get("HEAD_CHARS", "4000"))
MIDDLE_CHARS = int(os.environ.get("MIDDLE_CHARS", "3000"))
TAIL_CHARS = int(os.environ.get("TAIL_CHARS", "2000"))
DESC_CHARS = int(os.environ.get("DESC_CHARS", "1500"))


def sample_transcript(text: str) -> str:
    """Return head + middle + tail concatenated with separators when sampling."""
    text = text or ""
    full_budget = HEAD_CHARS + MIDDLE_CHARS + TAIL_CHARS
    if len(text) <= full_budget:
        return text
    head = text[:HEAD_CHARS]
    mid_start = (len(text) - MIDDLE_CHARS) // 2
    middle = text[mid_start:mid_start + MIDDLE_CHARS]
    tail = text[-TAIL_CHARS:]
    return (
        f"{head}\n"
        f"[...middle of transcript, sampled...]\n"
        f"{middle}\n"
        f"[...end of transcript...]\n"
        f"{tail}"
    )


def load_yes_v2_rows():
    rows = []
    all_rows = []
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            all_rows.append(row)
            if (row.get("_backlash_v2") or "").strip().upper() == "YES":
                rows.append(row)
    return all_rows, rows


def verify(rows_to_check, limit=None):
    if limit:
        rows_to_check = rows_to_check[:limit]
    clients = get_clients()
    n_keys = len(clients)
    n_workers = max(1, n_keys * WORKERS_PER_KEY)
    model = clients[0][1]

    print(f"  Model: {model}")
    print(f"  Keys: {n_keys}, workers: {n_workers}")
    print(f"  V2 YES rows to recheck: {len(rows_to_check)}")
    print(f"  Sample budget: head={HEAD_CHARS} mid={MIDDLE_CHARS} tail={TAIL_CHARS} desc={DESC_CHARS}")

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
            r["_backlash_v3"] = entry["label"] if isinstance(entry, dict) else entry
            r["_score_v3"] = entry.get("score") if isinstance(entry, dict) else ""
    print(f"  Pending: {len(pending)}")

    state_lock = threading.Lock()
    counters = {"api_calls": 0, "errors": 0, "completed": 0}
    start = time.time()

    def verify_one(idx_v):
        idx, v = idx_v
        vid = v.get("video_id", "")
        client, model_ = clients[idx % n_keys]

        title = (v.get("title") or "")[:300]
        description = (v.get("description") or "")[:DESC_CHARS]
        transcript = sample_transcript(v.get("_transcript_text") or "")
        prompt = VERIFY_PROMPT.format(title=title, description=description, transcript=transcript)

        label, score = "UNKNOWN", None
        for attempt in range(3):
            try:
                raw = llm_call(client, model_, prompt)
                label, score = parse_response(raw)
                with state_lock:
                    counters["api_calls"] += 1
                break
            except Exception as e:
                err = str(e)[:120]
                if attempt == 2:
                    with state_lock:
                        counters["errors"] += 1
                    print(f"    Error on {vid}: {err}")
                else:
                    time.sleep(1.5 * (attempt + 1))

        v["_backlash_v3"] = label
        v["_score_v3"] = "" if score is None else score

        with state_lock:
            already_done[vid] = {"label": label, "score": score}
            counters["completed"] += 1
            done = counters["completed"]
            if done % 50 == 0 or done == len(pending):
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(already_done, f)
                yes_n = sum(1 for x in already_done.values() if (x.get("label") if isinstance(x, dict) else x) == "YES")
                no_n = sum(1 for x in already_done.values() if (x.get("label") if isinstance(x, dict) else x) == "NO")
                rate = done / max(time.time() - start, 1e-9)
                eta = (len(pending) - done) / rate if rate > 0 else 0
                print(
                    f"    [{done}/{len(pending)}] {rate:.1f}/s | "
                    f"confirmed YES: {yes_n} demoted NO: {no_n} | "
                    f"errors: {counters['errors']} | ETA {eta/60:.1f}m"
                )

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(verify_one, (i, v)) for i, v in enumerate(pending)]
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
        "_search_keyword", "_backlash", "_backlash_v2", "_score_v2",
        "_backlash_v3", "_score_v3",
        "_transcript_text",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            row.setdefault("_backlash_v3", "")
            row.setdefault("_score_v3", "")
            writer.writerow(row)
    print(f"\nSaved {len(all_rows)} rows to {OUTPUT_CSV}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, help="Process only first N rows (smoke test)")
    args = p.parse_args()

    print(f"Loading {INPUT_CSV.name}...")
    all_rows, target_rows = load_yes_v2_rows()
    print(f"  total rows: {len(all_rows)}  V2 YES rows: {len(target_rows)}")

    print("\n=== Round 3 (full-content recheck on V2 YES) ===")
    verify(target_rows, limit=args.limit)

    save_output(all_rows)

    v2_yes = sum(1 for r in all_rows if (r.get("_backlash_v2") or "").upper() == "YES")
    confirmed = sum(1 for r in all_rows if r.get("_backlash_v3") == "YES")
    demoted = sum(1 for r in all_rows if r.get("_backlash_v3") == "NO")
    unknown = sum(1 for r in all_rows if r.get("_backlash_v3") == "UNKNOWN")
    print("\n=== SUMMARY ===")
    print(f"  V2 YES (input set):  {v2_yes}")
    print(f"  V3 confirmed YES:    {confirmed}")
    print(f"  V3 demoted to NO:    {demoted}")
    print(f"  V3 unknown:          {unknown}")
    if confirmed + demoted > 0:
        v2_fp_rate = demoted / (confirmed + demoted) * 100
        print(f"  V2 false-positive rate (per V3 judge): {v2_fp_rate:.1f}%")
        print(f"  Final-set precision est. (assuming V3 ~equal to ground truth): {confirmed/(confirmed+demoted)*100:.1f}%")
    print(f"\nFinal label rule: YES iff _backlash_v2 == YES AND _backlash_v3 == YES")


if __name__ == "__main__":
    main()
