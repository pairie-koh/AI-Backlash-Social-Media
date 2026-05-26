"""Round-3 diagnostic — run on a focused sample and print per-row reasoning.

Picks the V2 YES rows most likely to be false positives (longest transcripts
— V2 only saw the first 1500 chars, so heavy truncation = highest FP risk),
runs the round-3 verifier, and prints title / V2 / V3 / reasoning side-by-side
so you can eyeball whether the demotions are real.

Defaults to 25 rows. Pre-seeds two known FPs from the prior audit
(1U1HMqtam90, hIDWmuWv8SY) so you can sanity-check those first.

Usage:
    python youtube/scripts/test_round3_sample.py
    python youtube/scripts/test_round3_sample.py --n 50
    python youtube/scripts/test_round3_sample.py --ids 1U1HMqtam90,hIDWmuWv8SY
    python youtube/scripts/test_round3_sample.py --random            # unbiased sample
    python youtube/scripts/test_round3_sample.py --no-known-fps      # don't pre-seed
"""

import argparse
import csv
import io
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

csv.field_size_limit(2**31 - 1)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_yes_strict import VERIFY_PROMPT, parse_response, get_clients, llm_call  # noqa: E402
from verify_yes_round3 import sample_transcript, DESC_CHARS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "youtube_ai_backlash_verified.csv"
OUT_JSON = DATA_DIR / "test_round3_sample_results.json"

KNOWN_FPS = ["1U1HMqtam90", "hIDWmuWv8SY"]  # judge_v3=NO from audit_v2_results.json


def pick_sample(rows, n, random_mode, ids, include_known_fps):
    if ids:
        wanted = set(i.strip() for i in ids if i.strip())
        return [r for r in rows if r.get("video_id") in wanted]
    if random_mode:
        random.seed(42)
        return random.sample(rows, min(n, len(rows)))
    rows_sorted = sorted(rows, key=lambda r: len(r.get("_transcript_text") or ""), reverse=True)
    picked = []
    if include_known_fps:
        seen = set()
        for r in rows:
            if r.get("video_id") in KNOWN_FPS:
                picked.append(r)
                seen.add(r.get("video_id"))
        for r in rows_sorted:
            if r.get("video_id") not in seen and len(picked) < n:
                picked.append(r)
    else:
        picked = rows_sorted[:n]
    return picked[:n]


def truncate(s, n):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def run_one(client, model, row):
    title = (row.get("title") or "")[:300]
    description = (row.get("description") or "")[:DESC_CHARS]
    transcript = sample_transcript(row.get("_transcript_text") or "")
    prompt = VERIFY_PROMPT.format(title=title, description=description, transcript=transcript)
    raw = llm_call(client, model, prompt)
    label, score = parse_response(raw)
    return label, score, raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--ids", help="Comma-sep video_ids to test specifically")
    p.add_argument("--random", action="store_true", help="Random sample instead of longest-transcript")
    p.add_argument("--no-known-fps", action="store_true", help="Don't pre-seed known FPs from audit")
    args = p.parse_args()

    print(f"Loading {INPUT_CSV.name}...")
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))
    yes_v2 = [r for r in all_rows if (r.get("_backlash_v2") or "").strip().upper() == "YES"]
    print(f"  total: {len(all_rows)}  V2 YES: {len(yes_v2)}")

    ids_list = args.ids.split(",") if args.ids else None
    sample = pick_sample(yes_v2, args.n, args.random, ids_list, not args.no_known_fps)
    print(f"  Sampled: {len(sample)} rows  (mode: {'ids' if ids_list else 'random' if args.random else 'longest-transcript'})")

    clients = get_clients()
    n_keys = len(clients)
    print(f"  Model: {clients[0][1]}  Keys: {n_keys}\n")

    lock = threading.Lock()
    results = [None] * len(sample)

    def worker(idx_v):
        idx, v = idx_v
        client, model_ = clients[idx % n_keys]
        try:
            label, score, raw = run_one(client, model_, v)
            return idx, v, label, score, raw, None
        except Exception as e:
            return idx, v, "ERROR", None, "", str(e)[:200]

    print("Running round 3 in parallel...")
    start = time.time()
    with ThreadPoolExecutor(max_workers=n_keys * 3) as ex:
        futures = [ex.submit(worker, (i, r)) for i, r in enumerate(sample)]
        for fut in as_completed(futures):
            idx, v, label, score, raw, err = fut.result()
            with lock:
                results[idx] = (v, label, score, raw, err)
    print(f"Done in {int(time.time()-start)}s\n")

    # Pretty per-row report
    sep = "─" * 100
    demoted_count = 0
    confirmed_count = 0
    error_count = 0
    out_records = []
    for i, item in enumerate(results, 1):
        v, label, score, raw, err = item
        v2_score = v.get("_score_v2") or "?"
        tlen = len(v.get("_transcript_text") or "")
        keyword = v.get("_search_keyword") or ""
        views = v.get("views") or "?"
        title = v.get("title") or ""
        vid = v.get("video_id") or ""
        is_known_fp = vid in KNOWN_FPS
        flag_known = "  [known-FP]" if is_known_fp else ""
        if err:
            error_count += 1
            verdict_tag = f"ERROR: {err}"
        elif label == "NO":
            demoted_count += 1
            verdict_tag = f"!! DEMOTED to NO  (V2→V3 caught a likely FP)"
        elif label == "YES":
            confirmed_count += 1
            verdict_tag = "confirmed YES"
        else:
            verdict_tag = f"UNKNOWN ({label})"

        print(sep)
        print(f"[{i}/{len(results)}] {truncate(title, 90)}{flag_known}")
        print(f"  vid: {vid}  |  keyword: {keyword!r}  |  views: {views}  |  transcript_len: {tlen}")
        print(f"  V2: YES (score {v2_score})    V3: {label} (score {score})    → {verdict_tag}")
        if raw:
            reasoning = raw.split("REASONING:", 1)[-1].split("SCORE:", 1)[0].strip()
            print(f"  V3 reasoning: {truncate(reasoning, 500)}")

        out_records.append({
            "video_id": vid,
            "title": title[:200],
            "keyword": keyword,
            "views": views,
            "transcript_len": tlen,
            "v2_score": v2_score,
            "v3_label": label,
            "v3_score": score,
            "v3_raw": raw[:1500] if raw else "",
            "known_fp": is_known_fp,
            "error": err,
        })

    print(sep)
    n_decided = confirmed_count + demoted_count
    print(f"\n=== RESULT on this {len(results)}-row sample ===")
    print(f"  V3 confirmed YES: {confirmed_count}")
    print(f"  V3 demoted NO:    {demoted_count}    ← V2 false-positive estimate")
    print(f"  V3 errors/unknown: {error_count + (len(results) - n_decided - error_count)}")
    if n_decided:
        print(f"  V2 FP rate on this sample: {demoted_count/n_decided*100:.1f}%")

    OUT_JSON.write_text(json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull responses saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
