"""Validate the new V2 (verify_yes_strict.py) on the 30 demoted audit samples.

Compares its output against my eyeball judgments on the 13 V3-rescued ones from
the prior audit. The other 17 had V2=NO and V3=NO so are taken as ground-truth NO.
"""

import csv
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

csv.field_size_limit(2**31 - 1)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Reuse the new V2 prompt + parser from verify_yes_strict.py
sys.path.insert(0, str(Path(__file__).parent))
from verify_yes_strict import VERIFY_PROMPT, parse_response, get_clients, llm_call

DATA = Path(__file__).resolve().parent.parent / "data"
AUDIT = DATA / "audit_v2_results.json"
INPUT_CSV = DATA / "youtube_ai_backlash_filtered.csv"

# My eyeball judgments on the 13 V3-rescued samples.
# Source: my earlier review of the demoted bucket from audit_v2_results.json.
EYEBALL = {
    # Clear YES (V2 wrongly demoted)
    "—": "YES",  # placeholder; populated by video_id below
}

# Map by video_id and bucket. Anything not in EYEBALL_BY_VID is treated as
# "ground-truth NO" since both V2 and V3 said NO on it.
EYEBALL_BY_VID = {
    # The 13 V3-rescued — my eyeball read:
    # YES (V2 wrongly demoted)
    # We don't have the IDs hardcoded; build from titles below. Filled at runtime.
}


def load_audit_demoted():
    data = json.load(open(AUDIT, encoding="utf-8"))
    return data["samples"]["demoted"]


# Eyeball labels keyed by title prefix (resilient to ID changes)
EYEBALL_BY_TITLE_PREFIX = {
    "Microsoft Ruined Windows": "YES",
    "AI Is Replacing Jobs Faster": "AMBIG",
    "The END of Actors? AI Deepfakes": "NO",
    "Hot WGA Sag Iatse Teamsters": "AMBIG",
    "Stupid AI Responses": "YES",
    "Labor minister addresses concerns": "YES",
    "Every Company Exposed for Spying": "AMBIG",
    "Your AI Is Rewriting Your Thoughts": "YES",
    "D-Pop": "AMBIG",
    "Пузырь ИИ": "YES",
    "AI Deepfakes & Fake News in Michigan": "YES",
    "Can AI really Take all Jobs": "AMBIG",
    "The AI Power Crisis": "YES",
}


def eyeball_label(title):
    for prefix, lab in EYEBALL_BY_TITLE_PREFIX.items():
        if title.startswith(prefix):
            return lab
    return "NO"  # everything else (V2=NO, V3=NO) is ground-truth NO


def main():
    demoted = load_audit_demoted()
    print(f"Loaded {len(demoted)} demoted samples from audit")

    # Build full row lookup from V1 CSV
    idx = {}
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            idx[row["video_id"]] = row

    rows = [idx[e["video_id"]] for e in demoted if e["video_id"] in idx]
    print(f"Resolved {len(rows)} full rows from V1 CSV")

    clients = get_clients()
    n_keys = len(clients)
    print(f"Keys: {n_keys}, model: {clients[0][1]}")

    results = []
    lock = threading.Lock()

    def worker(idx_v):
        i, v = idx_v
        client, model = clients[i % n_keys]
        title = (v.get("title") or "")[:300]
        desc = (v.get("description") or "")[:500]
        transcript = (v.get("_transcript_text") or "")[:1500]
        prompt = VERIFY_PROMPT.format(title=title, description=desc, transcript=transcript)
        for attempt in range(3):
            try:
                raw = llm_call(client, model, prompt)
                label, score = parse_response(raw)
                return v, label, score, raw, None
            except Exception as e:
                last = str(e)[:120]
                time.sleep(1.5 * (attempt + 1))
        return v, "ERROR", None, "", last

    print("\nRunning new V2 on validation set...")
    start = time.time()
    with ThreadPoolExecutor(max_workers=n_keys * 3) as ex:
        futures = [ex.submit(worker, (i, r)) for i, r in enumerate(rows)]
        for fut in as_completed(futures):
            v, label, score, raw, err = fut.result()
            with lock:
                results.append({"row": v, "label": label, "score": score, "raw": raw, "err": err})
    print(f"Done in {int(time.time()-start)}s\n")

    # Compare to eyeball
    by_id = {r["row"]["video_id"]: r for r in results}
    rescued_seen = []
    other_seen = []
    for e in demoted:
        title = e.get("title") or ""
        ground = eyeball_label(title)
        v_res = by_id.get(e["video_id"])
        if not v_res:
            continue
        new_v2 = v_res["label"]
        score = v_res["score"]
        record = {
            "title": title[:80],
            "ground": ground,
            "new_v2": new_v2,
            "score": score,
            "v2_old": "NO",  # by definition (demoted)
            "v3_audit": e["judge_v3"],
        }
        if e["judge_v3"] == "YES":
            rescued_seen.append(record)
        else:
            other_seen.append(record)

    print("=" * 90)
    print("RESCUED BUCKET (V3 said YES; my eyeball labels)")
    print("=" * 90)
    print(f"{'eye':6} {'newV2':6} {'score':6}  title")
    correct_yes = wrong_yes = ambig = 0
    for r in rescued_seen:
        print(f"{r['ground']:6} {r['new_v2']:6} {str(r['score']):6}  {r['title']}")
        if r["ground"] == "YES" and r["new_v2"] == "YES":
            correct_yes += 1
        elif r["ground"] == "YES" and r["new_v2"] == "NO":
            wrong_yes += 1
        elif r["ground"] == "AMBIG":
            ambig += 1
    print()
    print(f"  Of clear-YES eyeball ({correct_yes + wrong_yes} videos): newV2 caught {correct_yes}, missed {wrong_yes}")
    print(f"  Ambiguous ({ambig}): newV2 verdict can go either way")

    print()
    print("=" * 90)
    print("CONSENSUS-NO BUCKET (V2=NO, V3=NO — should ALL be NO)")
    print("=" * 90)
    correct_no = wrong_no = 0
    for r in other_seen:
        if r["new_v2"] == "NO":
            correct_no += 1
        else:
            wrong_no += 1
            print(f"  ⚠ NEW-V2 SAYS YES (false positive on consensus-NO): {r['title']} | score={r['score']}")
    print(f"  newV2 = NO on {correct_no}/{len(other_seen)} consensus-NO samples (false-positive rate {wrong_no/max(1,len(other_seen))*100:.1f}%)")

    # One-shot precision/recall summary
    print()
    print("=" * 90)
    print("ESTIMATED PRECISION/RECALL on this 30-sample validation set")
    print("=" * 90)
    # True ground-truth labels for the 30:
    true_yes = [r for r in rescued_seen + other_seen if r["ground"] == "YES"]
    true_no = [r for r in rescued_seen + other_seen if r["ground"] == "NO"]
    ambiguous = [r for r in rescued_seen + other_seen if r["ground"] == "AMBIG"]

    tp = sum(1 for r in true_yes if r["new_v2"] == "YES")
    fn = sum(1 for r in true_yes if r["new_v2"] == "NO")
    fp = sum(1 for r in true_no if r["new_v2"] == "YES")
    tn = sum(1 for r in true_no if r["new_v2"] == "NO")

    print(f"  Ground truth: YES={len(true_yes)}  NO={len(true_no)}  AMBIG={len(ambiguous)}")
    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    if tp + fp > 0:
        print(f"  Precision (no false positives): {tp/(tp+fp)*100:.1f}%")
    if tp + fn > 0:
        print(f"  Recall:                         {tp/(tp+fn)*100:.1f}%")

    # Save full output for inspection
    out = DATA / "validate_new_v2_results.json"
    out.write_text(json.dumps(
        [{"video_id": r["row"]["video_id"], "title": (r["row"].get("title") or "")[:120],
          "label": r["label"], "score": r["score"], "raw": r["raw"][:400]}
         for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull responses saved to {out}")


if __name__ == "__main__":
    main()
