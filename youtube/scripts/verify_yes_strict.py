"""V2 strict verifier — few-shot + chain-of-thought + scoring.

Re-classifies V1 YES rows from youtube_ai_backlash_filtered.csv. Replaces the
old V2 (which used long reject-pattern lists that overshot and killed real
backlash on energy / news / mockery / non-English content).

Approach:
- Few-shot exemplars (3 YES, 3 NO) chosen from this corpus
- Model reasons step-by-step before answering
- Outputs SCORE (1-10) + ANSWER (YES/NO); threshold YES at score >= 7
- Default to NO under uncertainty (precision-first)

Usage:
    python youtube/scripts/verify_yes_strict.py
    python youtube/scripts/verify_yes_strict.py --ids-file <path>      # only process specific video_ids
    python youtube/scripts/verify_yes_strict.py --limit 50             # smoke test
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "youtube_ai_backlash_filtered.csv"
OUTPUT_CSV = DATA_DIR / "youtube_ai_backlash_verified.csv"
PROGRESS_FILE = DATA_DIR / "youtube_verify_v2_progress.json"
KEYS_FILE = REPO_ROOT / ".openrouter_keys"

OPENROUTER_API_KEYS = os.environ.get("OPENROUTER_API_KEYS", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")
WORKERS_PER_KEY = int(os.environ.get("WORKERS_PER_KEY", "3"))
SCORE_THRESHOLD = int(os.environ.get("SCORE_THRESHOLD", "6"))


VERIFY_PROMPT = """You are evaluating whether a YouTube video genuinely expresses backlash about AI. The video was found by searching for an AI-related keyword, but the keyword alone is NOT evidence of backlash — only the creator's actual stance matters.

Real backlash means the creator is critical, worried, or against AI for substantive reasons:
- Job displacement / labor concerns (and the creator agrees this is bad)
- Environmental / energy / data center concerns
- Art theft, training without consent, harm to creative workers
- Privacy, surveillance, deepfakes, misinformation, manipulation
- AI in education concerns (cheating, deskilling) framed critically
- Calls to regulate, ban, slow AI; safety / alignment / existential concerns
- Complaints about AI slop, AI making products worse, AI ruining the internet
- Anti-AI stance ("no AI", "human-made") used as principled rejection
- Mockery, satire, or ridicule of AI failures (this IS criticism)
- News reporting that frames AI concerns as legitimate (not just "balanced both-sides")

NOT backlash: pro-AI hype, neutral product reviews, AI tutorials, AI-generated content uploaded by AI fans, "no AI" as a marketing badge in unrelated content (e.g. WW2 docs, longevity tips), investment hype that recommends buying AI stocks, fictional/game AI, brief AI mentions inside an unrelated topic.

Important clarifications:
- News interviews and political segments where a speaker substantively expresses AI concerns ARE backlash, not "balanced reporting." If a politician, scientist, expert, or journalist takes a clear concerned stance — even framed as an interview answer — that counts as YES.
- Sparse data (short transcript, missing description) is NOT itself a reason to demote. Judge by what the available content actually says. If the available signal is unambiguously critical, give a high score even if the transcript is short.
- Mockery, satire, and ridicule of AI failures ARE backlash (the creator's stance is still critical).

When the stance is genuinely unclear from available content, default to NO — precision matters more than recall. But never demote purely because of data volume.

---

Here are six examples to anchor your judgment:

EXAMPLE 1
Title: Canada's AI Advisor: Mass Job Loss Is Coming (UBI) | Yoshua Bengio
Description: "There is huge commercial pressure to build AIs with greater and greater agency to replace human labor." — Yoshua Bengio, Turing Award winner.
Transcript: There's huge commercial pressure to build AIs with greater and greater agency to replace human labor, but we're not ready. We still don't have the scientific answers, nor the societal guard rails. I'm the most cited computer scientist in the world, and you'd think that people would heed my warnings.
REASONING: Top AI scientist explicitly warning that AI will displace labor and that society isn't ready. Substantive concern, primary topic.
SCORE: 10/10
ANSWER: YES

EXAMPLE 2
Title: WATCH THIS! IF YOU HATE A.i!
Description: This video is a summation of several pleasentries on my hate for Artificial intelligence.
Transcript: This video is dedicated to my undying hatred. Hatred for artificial intelligence. Tell me one significant thing the artificial intelligence have done for human beings.
REASONING: Explicit, sustained personal anti-AI stance. Title and content match. Mockery + rejection.
SCORE: 9/10
ANSWER: YES

EXAMPLE 3
Title: There WILL be an Anti-AI PROTEST in London - What you need to know
Description: AI should be regulated in this Country more.
Transcript: Tomorrow there will be an AI protest in London. They think AI strips away creativity, security, and connection. And I completely agree with this.
REASONING: Creator is organizing/promoting an anti-AI protest and explicitly endorses its stance. Clear backlash.
SCORE: 10/10
ANSWER: YES

EXAMPLE 4
Title: Senator addresses concerns around AI deepfake misinformation
Description: Senator Smith has addressed concerns around AI-driven disinformation, calling for action.
Transcript: yeah I get all the upsides and the industry itself is very exciting but one part that worries me and a lot of people, ChatGPT is going to feed into all kinds of disinformation, I've fallen for it hundreds of times this year, so I mean you can't just ban it can you, are there upsides here, has the horse bolted, what do you do.
REASONING: Politician in an interview substantively expressing personal worry about AI misinformation, with concrete examples of being fooled. The interview format doesn't make it neutral — the speaker is taking a clear concerned stance. This is backlash, not balanced reporting.
SCORE: 8/10
ANSWER: YES

EXAMPLE 5
Title: Why AI Will Break the Power Grid
Description:
Transcript: AI data centers are now consuming more electricity than entire countries. Microsoft's water usage doubled in three years just to cool servers. We can't keep building this infrastructure indefinitely.
REASONING: No description and a short transcript, but content is unambiguous: framing AI's electricity and water footprint as unsustainable. Sparse data does not justify demotion when the available signal is clearly critical.
SCORE: 8/10
ANSWER: YES

EXAMPLE 6
Title: 5 AI Tools That Will Replace Jobs in 2026 – Use Them Before It's Too Late!
Description: By 2026, AI will change everything — even your job! In this video, we reveal 5 AI tools that are already replacing tasks and creating massive opportunities.
Transcript: By 2026, these AI tools will replace millions of jobs. But if you use them first, you'll win big. Welcome to TechShock, your shortcut to mastering AI. Tool one: Chat GPT can write emails, articles, even books for you.
REASONING: Title is fear-bait but content is a pro-AI tools tutorial. Creator's stance is enthusiastic, "use them before it's too late" = sales pitch, not critique.
SCORE: 2/10
ANSWER: NO

EXAMPLE 7
Title: The Raid That Saved Millions? - Operation Gunnerside (No AI WW2 Documentary)
Description: Discover the untold story of Operation Gunnerside, the audacious WWII commando raid.
Transcript: On a frozen February night in 1943, Norway's snowy silence was pierced by the blare of alarms, barking guard dogs and spotlights slashing through the darkness. The world's largest heavy water plant had just been sabotaged.
REASONING: "No AI" is a marketing badge signaling traditional human-made content. Video itself is a WWII documentary; AI is not the topic at all.
SCORE: 1/10
ANSWER: NO

EXAMPLE 8
Title: S2G Podcast: The Case for Automating American Factories With Formic
Description: From CapEx to OpEx: The Hardware-as-a-Service Opportunity. Conversation with the CEO of Formic.
Transcript: very excited to be here today with Simon Fared, CEO and co-founder of Formic. The idea of physical AI and robotics and where this is going is super super cool.
REASONING: Despite the keyword match, this is a pro-automation investment podcast. Host and guest are excited about robotics/AI replacing manual labor. Pro-AI stance.
SCORE: 2/10
ANSWER: NO

---

Now classify the following video. Use the same REASONING / SCORE / ANSWER format. Reason in 1-3 sentences before scoring.

Title: {title}
Description: {description}
Transcript: {transcript}"""


SCORE_RE = re.compile(r"SCORE\s*:\s*(\d{1,2})\s*/\s*10", re.IGNORECASE)
ANSWER_RE = re.compile(r"ANSWER\s*:\s*(YES|NO)", re.IGNORECASE)


def parse_response(text: str):
    """Returns (label: 'YES'|'NO'|'UNKNOWN', score: int|None)."""
    score = None
    m = SCORE_RE.search(text)
    if m:
        try:
            score = int(m.group(1))
        except ValueError:
            score = None
    m = ANSWER_RE.search(text)
    explicit = m.group(1).upper() if m else None
    if explicit:
        if score is not None and score < SCORE_THRESHOLD and explicit == "YES":
            return "NO", score  # threshold override (precision-first)
        return explicit, score
    # Fallback: derive from score
    if score is not None:
        return ("YES" if score >= SCORE_THRESHOLD else "NO"), score
    # Last resort: scan text
    upper = text.upper()
    if "ANSWER" in upper and "YES" in upper.split("ANSWER", 1)[1][:30]:
        return "YES", score
    if "ANSWER" in upper and "NO" in upper.split("ANSWER", 1)[1][:30]:
        return "NO", score
    return "UNKNOWN", score


def _load_openrouter_keys():
    if OPENROUTER_API_KEYS.strip():
        return [k.strip() for k in OPENROUTER_API_KEYS.split(",") if k.strip()]
    if KEYS_FILE.exists():
        return [
            ln.strip() for ln in KEYS_FILE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    if OPENROUTER_API_KEY:
        return [OPENROUTER_API_KEY]
    return []


def _load_openai_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    p = REPO_ROOT / ".openai_key"
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                return ln
    return ""


def get_clients():
    """Returns list of (client, model). For OpenAI: 7 slots sharing one client (for parallelism).
    For OpenRouter: one (client, model) per key."""
    from openai import OpenAI
    provider = os.environ.get("LLM_PROVIDER", "openrouter").lower()

    if provider == "openai":
        key = _load_openai_key()
        if not key:
            print("ERROR: LLM_PROVIDER=openai but no OPENAI_API_KEY env / .openai_key file found")
            sys.exit(1)
        model = os.environ.get("LLM_MODEL")
        if not model or model == "anthropic/claude-sonnet-4":
            model = "gpt-5"
        client = OpenAI(api_key=key)
        # Duplicate slots to keep n_keys * WORKERS_PER_KEY parallelism comparable
        return [(client, model)] * 7

    keys = _load_openrouter_keys()
    if not keys:
        print("ERROR: no OpenRouter keys found")
        sys.exit(1)
    return [
        (OpenAI(base_url="https://openrouter.ai/api/v1", api_key=k), LLM_MODEL)
        for k in keys
    ]


def llm_call(client, model, prompt, max_tokens=400):
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    # GPT-5 / o-series reasoning models require max_completion_tokens.
    # Disable internal reasoning so the prompt's CoT structure does the work
    # (otherwise the model burns the token budget on hidden reasoning and returns empty).
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["max_completion_tokens"] = max_tokens
        kwargs["reasoning_effort"] = "minimal"
    else:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def load_yes_rows(ids_filter=None):
    """Load V1 YES rows. If ids_filter is provided, restrict to those video_ids."""
    rows = []
    all_rows = []
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)
            if (row.get("_backlash") or "").strip().upper() != "YES":
                continue
            if ids_filter and row.get("video_id") not in ids_filter:
                continue
            rows.append(row)
    return all_rows, rows


def verify(rows_to_check, all_rows, limit=None):
    if limit:
        rows_to_check = rows_to_check[:limit]
    clients = get_clients()
    n_keys = len(clients)
    n_workers = max(1, n_keys * WORKERS_PER_KEY)
    model = clients[0][1]

    print(f"  Model: {model}")
    print(f"  Keys: {n_keys}, workers: {n_workers}")
    print(f"  YES rows to verify: {len(rows_to_check)}")
    print(f"  YES threshold: score >= {SCORE_THRESHOLD}/10")

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
            r["_backlash_v2"] = entry["label"] if isinstance(entry, dict) else entry
            r["_score_v2"] = (entry.get("score") if isinstance(entry, dict) else "")
    print(f"  Pending: {len(pending)}")

    state_lock = threading.Lock()
    counters = {"api_calls": 0, "errors": 0, "completed": 0}
    start = time.time()

    def verify_one(idx_v):
        idx, v = idx_v
        vid = v.get("video_id", "")
        client, model_ = clients[idx % n_keys]

        title = (v.get("title") or "")[:300]
        description = (v.get("description") or "")[:500]
        transcript = (v.get("_transcript_text") or "")[:1500]
        prompt = VERIFY_PROMPT.format(title=title, description=description, transcript=transcript)

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
                    label, score, raw = "UNKNOWN", None, ""
                else:
                    time.sleep(1.5 * (attempt + 1))

        v["_backlash_v2"] = label
        v["_score_v2"] = "" if score is None else score

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
        "_transcript_text",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            row.setdefault("_backlash_v2", "")
            row.setdefault("_score_v2", "")
            writer.writerow(row)
    print(f"\nSaved {len(all_rows)} rows to {OUTPUT_CSV}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids-file", help="File with one video_id per line — restrict to these")
    p.add_argument("--limit", type=int, help="Process only first N rows (smoke test)")
    args = p.parse_args()

    ids_filter = None
    if args.ids_file:
        ids_filter = set(
            ln.strip() for ln in Path(args.ids_file).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        )
        print(f"ID filter: {len(ids_filter)} video_ids from {args.ids_file}")

    print("Loading V1 CSV...")
    all_rows, target_rows = load_yes_rows(ids_filter=ids_filter)
    print(f"  total rows: {len(all_rows)}  target rows: {len(target_rows)}")

    print("\n=== V2 strict verification (few-shot + CoT) ===")
    verify(target_rows, all_rows, limit=args.limit)

    save_output(all_rows)

    confirmed = sum(1 for r in all_rows if r.get("_backlash_v2") == "YES")
    demoted = sum(1 for r in all_rows if r.get("_backlash_v2") == "NO")
    unknown = sum(1 for r in all_rows if r.get("_backlash_v2") == "UNKNOWN")
    print("\n=== SUMMARY ===")
    print(f"  Confirmed YES: {confirmed}")
    print(f"  Demoted NO:    {demoted}")
    print(f"  Unknown:       {unknown}")
    if confirmed + demoted > 0:
        print(f"  V1 precision estimate: {confirmed/(confirmed+demoted)*100:.1f}%")


if __name__ == "__main__":
    main()
