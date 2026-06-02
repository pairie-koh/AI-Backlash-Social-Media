"""Tier 2 sub-classification via the Message Batches API.

Same prompts / taxonomies as classify_subtopics.py, but submitted as a single
batch so we are not bound by the 50 req/min realtime rate limit, and at 50%
lower cost. Resume-safe: rows already in the sidecars are skipped; a submitted
batch id is cached so re-running just polls + harvests.

Usage:
    python analysis/classify_subtopics_batch.py            # submit + poll + write
    python analysis/classify_subtopics_batch.py --submit   # submit only, print id
    python analysis/classify_subtopics_batch.py --harvest   # poll cached id + write
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drilldown_common import load_tiktok, REPO
from classify_subtopics import (PROMPT, menu_text, THEME_DESC, evidence_of,
                                BACKLASH_SUBTAX, POSITIVE_SUBTAX, OUT_RE, MODEL)

BATCH_STATE = REPO / "tiktok/data/final/.subtopics_batch.json"


def get_client():
    key = None
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return anthropic.Anthropic(api_key=key)


def build_prompt(r, theme_key, subtax):
    theme = r[theme_key]
    return PROMPT.format(
        platform="TikTok", theme=theme, theme_desc=THEME_DESC.get(theme, theme),
        menu=menu_text(subtax[theme]),
        description=(r.get("description") or "")[:600],
        hashtags=(r.get("hashtags") or "")[:200],
        keyword=r.get("_search_keyword") or "",
        evidence=evidence_of(r) or "(none)",
    )


def sidecar_path(corpus):
    return REPO / f"tiktok/data/final/tiktok_{corpus}_subtopics.csv"


def done_ids(corpus):
    p = sidecar_path(corpus)
    ids = set()
    if p.exists():
        with open(p, encoding="utf-8", newline="") as f:
            for d in csv.DictReader(f):
                ids.add(d["post_id"])
    return ids


def collect_todo():
    """Returns list of (custom_id, corpus, post_id, theme, prompt)."""
    b, p = load_tiktok(only_2026=False)
    todo = []
    for corpus, rows, theme_key, subtax in [
        ("backlash", b, "_topic_primary", BACKLASH_SUBTAX),
        ("positive", p, "_theme", POSITIVE_SUBTAX),
    ]:
        done = done_ids(corpus)
        pref = "b" if corpus == "backlash" else "p"
        for r in rows:
            pid = r.get("post_id")
            if not pid or pid in done or r.get(theme_key) not in subtax:
                continue
            todo.append((f"{pref}_{pid}", corpus, pid, r[theme_key],
                         build_prompt(r, theme_key, subtax)))
    return todo


def submit(client):
    todo = collect_todo()
    print(f"submitting {len(todo):,} requests (model={MODEL})")
    if not todo:
        return None
    requests = [
        Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
            model=MODEL, max_tokens=40,
            messages=[{"role": "user", "content": prompt}]))
        for cid, _, _, _, prompt in todo
    ]
    batch = client.messages.batches.create(requests=requests)
    BATCH_STATE.write_text(json.dumps({"id": batch.id, "n": len(todo)}))
    print(f"submitted batch {batch.id}  (state cached -> {BATCH_STATE.name})")
    return batch.id


def harvest(client, poll=True):
    if not BATCH_STATE.exists():
        print("no cached batch; run --submit first")
        return
    bid = json.loads(BATCH_STATE.read_text())["id"]
    while True:
        batch = client.messages.batches.retrieve(bid)
        rc = batch.request_counts
        print(f"  {batch.processing_status}: succeeded={rc.succeeded} "
              f"errored={rc.errored} processing={rc.processing}")
        if batch.processing_status == "ended" or not poll:
            break
        time.sleep(20)
    if batch.processing_status != "ended":
        print("batch not ended yet; re-run --harvest later")
        return

    rows = {"backlash": [], "positive": []}
    n_ok = n_err = 0
    for res in client.messages.batches.results(bid):
        cid = res.custom_id
        corpus = "backlash" if cid.startswith("b_") else "positive"
        pid = cid[2:]
        if res.result.type != "succeeded":
            n_err += 1
            continue
        text = res.result.message.content[0].text
        m = OUT_RE.search(text)
        sub = m.group(1).lower() if m else "other"
        conf = m.group(2).lower() if m else "low"
        rows[corpus].append((pid, sub, conf))
        n_ok += 1

    for corpus, recs in rows.items():
        if not recs:
            continue
        p = sidecar_path(corpus)
        # Need theme per pid to write the theme column; rebuild lookup.
        b, pos = load_tiktok(only_2026=False)
        src = {r["post_id"]: r for r in (b if corpus == "backlash" else pos)}
        key = "_topic_primary" if corpus == "backlash" else "_theme"
        write_header = not p.exists()
        with open(p, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["post_id", "theme", "subtopic", "confidence"])
            for pid, sub, conf in recs:
                theme = src.get(pid, {}).get(key, "")
                w.writerow([pid, theme, sub, conf])
        print(f"[{corpus}] appended {len(recs):,} rows -> {p.name}")
    print(f"harvest done: ok={n_ok:,} errored={n_err:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--harvest", action="store_true")
    args = ap.parse_args()
    client = get_client()
    if args.harvest:
        harvest(client)
    elif args.submit:
        submit(client)
    else:
        submit(client)
        harvest(client)


if __name__ == "__main__":
    main()
