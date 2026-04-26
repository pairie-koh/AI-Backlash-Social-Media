"""
Trigger Bright Data collection for TikTok + YouTube simultaneously.

Reads keywords from KEYWORDS.md, fires triggers for both dataset IDs
(no per-keyword limit), polls until snapshots are ready, and saves
results to the per-platform raw JSON files.

Usage:
    python scripts/collect_brightdata.py
"""

import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
KEYWORDS_FILE = ROOT / "KEYWORDS.md"
MANIFEST_FILE = ROOT / "scripts" / "brightdata_snapshots.json"

PLATFORMS = {
    "tiktok": {
        "dataset_id": "gd_lu702nij2f790tmv9h",
        "raw_json": ROOT / "tiktok" / "data" / "raw" / "tiktok_raw.json",
        "id_field": "post_id",
        "input_field": "search_keyword",
    },
    "youtube": {
        "dataset_id": "gd_lk56epmy2i5g7lzu0k",
        "raw_json": ROOT / "youtube" / "data" / "raw" / "youtube_raw.json",
        "id_field": "video_id",
        "input_field": "keyword",
    },
}

BRIGHTDATA_API_TOKEN = os.environ.get("BRIGHTDATA_API_TOKEN", "")
if not BRIGHTDATA_API_TOKEN:
    raise SystemExit("Set BRIGHTDATA_API_TOKEN env var before running.")
BASE_URL = "https://api.brightdata.com/datasets/v3"
HEADERS = {
    "Authorization": f"Bearer {BRIGHTDATA_API_TOKEN}",
    "Content-Type": "application/json",
}

POLL_INTERVAL = 30
MAX_WAIT = 120 * 60  # 2 hours — no per-input cap means slower scrapes

# Snapshots from earlier aborted runs — included so their data isn't wasted.
SEED_SNAPSHOTS = {
    "sd_mofg7wmo14xjxusx79": {"platform": "youtube", "keyword": "AI taking jobs"},
    "sd_mofg7zmc1eenr3sad7": {"platform": "youtube", "keyword": "AI replacing workers"},
    "sd_mofga5ev1enjmb6cx3": {"platform": "tiktok", "keyword": "AI taking jobs"},
}


def load_keywords():
    text = KEYWORDS_FILE.read_text(encoding="utf-8")
    pattern = re.compile(r"^\|\s*\d+\s*\|\s*`([^`]+)`", re.MULTILINE)
    return pattern.findall(text)


def trigger(platform, keyword):
    cfg = PLATFORMS[platform]
    try:
        resp = requests.post(
            f"{BASE_URL}/trigger",
            headers=HEADERS,
            json=[{cfg["input_field"]: keyword}],
            params={
                "dataset_id": cfg["dataset_id"],
                "format": "json",
                "type": "discover_new",
                "discover_by": "keyword",
            },
            timeout=30,
        )
    except Exception as e:
        print(f"    ERROR [{platform}/{keyword!r}]: {e}")
        return None

    if resp.status_code != 200:
        print(f"    ERROR [{platform}/{keyword!r}]: {resp.status_code} {resp.text[:200]}")
        return None
    return resp.json().get("snapshot_id")


def fire_all_triggers(keywords):
    jobs = {}
    total = len(keywords) * len(PLATFORMS)
    n = 0
    for keyword in keywords:
        for platform in PLATFORMS:
            n += 1
            sid = trigger(platform, keyword)
            if sid:
                jobs[sid] = {"platform": platform, "keyword": keyword}
                print(f"    [{n}/{total}] {platform}/{keyword!r} -> {sid}")
            time.sleep(0.4)
    return jobs


def save_manifest(jobs):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)
    print(f"  Manifest saved -> {MANIFEST_FILE.relative_to(ROOT)}")


def fetch_snapshot(snapshot_id):
    try:
        progress = requests.get(
            f"{BASE_URL}/progress/{snapshot_id}", headers=HEADERS, timeout=30,
        ).json()
    except Exception as e:
        return "poll_error", str(e)
    status = progress.get("status")
    if status == "ready":
        if progress.get("records", 0) == 0:
            return "ready", []
        try:
            data_resp = requests.get(
                f"{BASE_URL}/snapshot/{snapshot_id}",
                headers=HEADERS, params={"format": "json"}, timeout=180,
            )
            if data_resp.status_code == 200:
                return "ready", data_resp.json()
            return "download_error", []
        except Exception as e:
            return "download_error", str(e)
    return status, []


def merge_into_raw(platform, results):
    cfg = PLATFORMS[platform]
    raw_json = cfg["raw_json"]
    id_field = cfg["id_field"]
    raw_json.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    deduped = []
    if raw_json.exists():
        try:
            existing = json.load(open(raw_json, encoding="utf-8"))
            for r in existing:
                vid = r.get(id_field) or r.get("url") or r.get("id", "")
                if vid and vid not in seen:
                    seen.add(vid)
                    deduped.append(r)
        except (json.JSONDecodeError, IOError):
            pass
    for r in results:
        vid = r.get(id_field) or r.get("url") or r.get("id", "")
        if vid and vid not in seen:
            seen.add(vid)
            deduped.append(r)
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    return len(deduped)


def main():
    resume = len(sys.argv) > 1 and sys.argv[1] == "resume"

    if resume:
        if not MANIFEST_FILE.exists():
            print(f"ERROR: no manifest at {MANIFEST_FILE}")
            sys.exit(1)
        jobs = json.load(open(MANIFEST_FILE, encoding="utf-8"))
        print(f"RESUME mode — loaded {len(jobs)} snapshots from manifest")
    else:
        keywords = load_keywords()
        print(f"Loaded {len(keywords)} keywords from KEYWORDS.md")
        print(f"Firing {len(keywords) * len(PLATFORMS)} triggers (TikTok + YouTube)...\n")

        jobs = fire_all_triggers(keywords)
        print(f"\n  Triggered {len(jobs)} jobs.")

        if SEED_SNAPSHOTS:
            jobs.update(SEED_SNAPSHOTS)
            print(f"  + {len(SEED_SNAPSHOTS)} seed snapshots from prior runs")

        save_manifest(jobs)

    print(f"\nPolling every {POLL_INTERVAL}s (max {MAX_WAIT // 60}m)...\n")
    pending = dict(jobs)
    start_time = time.time()

    while pending and (time.time() - start_time) < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed = int(time.time() - start_time)
        newly_done = []
        results_by_platform = {p: [] for p in PLATFORMS}

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(fetch_snapshot, sid): sid for sid in list(pending.keys())}
            for fut in as_completed(futures):
                sid = futures[fut]
                meta = pending[sid]
                platform, keyword = meta["platform"], meta["keyword"]
                try:
                    status, payload = fut.result()
                except Exception as e:
                    print(f"    poll error {sid}: {e}")
                    continue

                if status == "ready":
                    newly_done.append(sid)
                    if isinstance(payload, list) and payload:
                        for r in payload:
                            r["_search_keyword"] = keyword
                        results_by_platform[platform].extend(payload)
                        print(f"    [{elapsed}s] {platform}/{keyword!r}: {len(payload)} records")
                    else:
                        print(f"    [{elapsed}s] {platform}/{keyword!r}: 0 records")
                elif status in ("failed",):
                    newly_done.append(sid)
                    print(f"    [{elapsed}s] {platform}/{keyword!r}: FAILED")

        for sid in newly_done:
            del pending[sid]

        for platform, results in results_by_platform.items():
            if results:
                n = merge_into_raw(platform, results)
                print(f"      -> {platform}: {n} unique videos in raw")

        if newly_done:
            done = len(jobs) - len(pending)
            print(f"    --- {done}/{len(jobs)} complete, {len(pending)} pending ---\n")

    if pending:
        print(f"\nWARNING: {len(pending)} jobs still pending after {MAX_WAIT // 60}m")
        print("  Re-run with the manifest to resume polling.")

    elapsed = int(time.time() - start_time)
    print(f"\nDone in {elapsed // 60}m {elapsed % 60}s")
    for platform, cfg in PLATFORMS.items():
        if cfg["raw_json"].exists():
            n = len(json.load(open(cfg["raw_json"], encoding="utf-8")))
            print(f"  {platform}: {n} unique videos in {cfg['raw_json'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
