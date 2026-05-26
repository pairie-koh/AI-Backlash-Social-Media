"""Quality audit: independent third-pass judge on samples from V2 verified CSV.

Buckets:
  - YES_v2 (V2 said YES) — precision check
  - V1_NO (V1 said NO, never re-checked) — recall check
  - DEMOTED (V1 said YES, V2 said NO) — was V2 right to drop them?

For each row we ask a fresh LLM (different prompt phrasing, no awareness of prior labels)
to classify YES/NO. Then compare to the existing label and report agreement.
"""

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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "youtube_ai_backlash_verified.csv"
OUT_JSON = DATA_DIR / "audit_v2_results.json"
KEYS_FILE = REPO_ROOT / ".openrouter_keys"

LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")
SAMPLE_PER_BUCKET = int(os.environ.get("SAMPLE_PER_BUCKET", "30"))
SEED = 11

JUDGE_PROMPT = """You are a strict evaluator. A YouTube video below was surfaced by an AI-related search keyword. Your job: decide whether the video itself genuinely expresses backlash, criticism, fear, or concern about AI as its actual stance.

Reply YES only if the creator's actual position (judged from title + description + transcript) is critical of AI, worried about its impact, or against AI for one of these reasons:
- Job displacement / automation taking work, framed as bad
- Environmental impact (energy, water, carbon)
- Art theft, training without consent, harm to artists
- Privacy, surveillance, deepfakes, manipulation, misinformation worries
- AI in education concerns (cheating, deskilling) framed critically
- Calls to regulate, ban, slow AI; safety / alignment / existential concerns
- Complaints about AI slop, AI making products worse, AI ruining the internet
- Anti-AI stance ("no AI", "human-made") used as principled rejection

Reply NO if the video:
- Is pro-AI, neutral, or off-topic, even if title contains scary AI keywords
- Uses "no AI" as a marketing badge in unrelated content (longevity, vintage, recipes, nature)
- Is a tutorial / review / how-to using AI tools
- Is investment / business content recommending AI
- Is AI-generated content uploaded by an AI hobbyist
- Is fiction, gaming, sci-fi, or NPC AI
- Briefly mentions AI worry as a small part of a broader topic
- Is clickbait that uses scary AI words but content is unrelated or pro-AI

Be ruthless. The keyword the video matched is not evidence of backlash; only the creator's actual stance is.

Respond with exactly one word: YES or NO

---
Search keyword: {keyword}
Title: {title}
Description: {description}
Transcript: {transcript}"""


def load_clients():
    from openai import OpenAI
    keys = [
        ln.strip() for ln in KEYS_FILE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    return [
        (OpenAI(base_url="https://openrouter.ai/api/v1", api_key=k), LLM_MODEL)
        for k in keys
    ]


def llm_judge(client, model, row):
    title = (row.get("title") or "")[:300]
    desc = (row.get("description") or "")[:500]
    transcript = (row.get("_transcript_text") or "")[:1500]
    keyword = row.get("_search_keyword") or ""
    prompt = JUDGE_PROMPT.format(
        keyword=keyword, title=title, description=desc, transcript=transcript,
    )
    resp = client.chat.completions.create(
        model=model, max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    txt = resp.choices[0].message.content.strip().upper()
    if "YES" in txt:
        return "YES"
    if "NO" in txt:
        return "NO"
    return "UNKNOWN"


def main():
    random.seed(SEED)
    print(f"Loading {INPUT_CSV.name}...")
    rows = list(csv.DictReader(INPUT_CSV.open("r", encoding="utf-8", newline="")))
    print(f"  total: {len(rows)}")

    yes_v2 = [r for r in rows if (r.get("_backlash_v2") or "").strip().upper() == "YES"]
    demoted = [
        r for r in rows
        if (r.get("_backlash") or "").strip().upper() == "YES"
        and (r.get("_backlash_v2") or "").strip().upper() == "NO"
    ]
    v1_no = [r for r in rows if (r.get("_backlash") or "").strip().upper() == "NO"]

    print(f"  YES_v2:  {len(yes_v2)}")
    print(f"  DEMOTED: {len(demoted)}  (V1=YES, V2=NO)")
    print(f"  V1_NO:   {len(v1_no)}")

    buckets = {
        "yes_v2":  random.sample(yes_v2,  min(SAMPLE_PER_BUCKET, len(yes_v2))),
        "demoted": random.sample(demoted, min(SAMPLE_PER_BUCKET, len(demoted))),
        "v1_no":   random.sample(v1_no,   min(SAMPLE_PER_BUCKET, len(v1_no))),
    }

    clients = load_clients()
    print(f"  judges: {len(clients)} OpenRouter keys, model={LLM_MODEL}")

    results = {}
    lock = threading.Lock()

    def grade(bucket, idx, row):
        client, model = clients[idx % len(clients)]
        for attempt in range(3):
            try:
                verdict = llm_judge(client, model, row)
                return bucket, row, verdict, None
            except Exception as e:
                last_err = str(e)[:120]
                time.sleep(1.5 * (attempt + 1))
        return bucket, row, "ERROR", last_err

    tasks = []
    for bucket, sample in buckets.items():
        for i, row in enumerate(sample):
            tasks.append((bucket, i, row))

    out = {b: [] for b in buckets}

    print(f"\nGrading {len(tasks)} samples in parallel...")
    start = time.time()
    with ThreadPoolExecutor(max_workers=len(clients) * 2) as ex:
        futures = [ex.submit(grade, b, i, r) for b, i, r in tasks]
        for fut in as_completed(futures):
            bucket, row, verdict, err = fut.result()
            v1 = (row.get("_backlash") or "").strip().upper()
            v2 = (row.get("_backlash_v2") or "").strip().upper()
            entry = {
                "video_id": row.get("video_id"),
                "title": (row.get("title") or "")[:120],
                "keyword": row.get("_search_keyword"),
                "views": row.get("views"),
                "channel": row.get("youtuber"),
                "v1": v1,
                "v2": v2,
                "judge_v3": verdict,
                "err": err,
            }
            with lock:
                out[bucket].append(entry)

    print(f"Graded in {int(time.time()-start)}s\n")

    # Summary
    summary = {}
    for bucket, entries in out.items():
        graded = [e for e in entries if e["judge_v3"] in ("YES", "NO")]
        v3_yes = sum(1 for e in graded if e["judge_v3"] == "YES")
        v3_no = sum(1 for e in graded if e["judge_v3"] == "NO")
        errors = sum(1 for e in entries if e["judge_v3"] == "ERROR")
        # Agreement with most-recent label (v2 if non-empty else v1)
        def latest(e):
            return e["v2"] if e["v2"] in ("YES", "NO") else e["v1"]
        agree = sum(1 for e in graded if e["judge_v3"] == latest(e))
        summary[bucket] = {
            "n_total": len(entries),
            "n_graded": len(graded),
            "v3_YES": v3_yes,
            "v3_NO": v3_no,
            "errors": errors,
            "agreement_with_latest_label": f"{agree}/{len(graded)} ({agree/max(1,len(graded))*100:.1f}%)",
        }

    print("=== AUDIT SUMMARY ===")
    for bucket, s in summary.items():
        print(f"\n[{bucket}] n={s['n_total']}, errors={s['errors']}")
        print(f"  Judge V3 said YES: {s['v3_YES']}")
        print(f"  Judge V3 said NO:  {s['v3_NO']}")
        print(f"  Agreement w/ latest label: {s['agreement_with_latest_label']}")

    OUT_JSON.write_text(json.dumps({"summary": summary, "samples": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull samples saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
