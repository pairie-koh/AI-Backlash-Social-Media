"""Tier 2: theme-specific sub-category classification (TikTok).

For every classified video, assign ONE sub-category drawn from a short menu
that is specific to that video's already-assigned top-level theme. This is a
second pass on top of the existing _topic_primary (backlash) / _theme
(positive) labels — it does not re-litigate the top-level theme.

Design notes:
  - Source rows come through drilldown_common.load_tiktok (canonical filter),
    so sub-topics line up exactly with the published main bars.
  - We classify ALL dates (not just 2026) to maximise n per sub-cell; the
    plotting step filters to 2026.
  - Output is a SIDECAR keyed by post_id (we never mutate the final CSVs):
        tiktok/data/final/tiktok_backlash_subtopics.csv
        tiktok/data/final/tiktok_positive_subtopics.csv
    Columns: post_id, theme, subtopic, confidence
  - Resume-safe: rows already in the sidecar are skipped on re-run.

Usage:
    python analysis/classify_subtopics.py --corpus backlash --limit 15   # test
    python analysis/classify_subtopics.py --corpus backlash              # full
    python analysis/classify_subtopics.py --corpus positive
    MODEL=claude-sonnet-4-6 python analysis/classify_subtopics.py ...     # upgrade
"""

import argparse
import csv
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drilldown_common import load_tiktok, REPO

csv.field_size_limit(2**31 - 1)

MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
WORKERS = int(os.environ.get("WORKERS", "8"))

# ---------------------------------------------------------------------------
# Sub-taxonomies. Each entry: code -> one-line definition shown to the model.
# Keep menus short (4-6) and mutually exclusive. "other" is always allowed.
# ---------------------------------------------------------------------------
BACKLASH_SUBTAX = {
    "art": {
        "style_mimicry": "AI copying / imitating a specific human artist's style or work",
        "creative_job_loss": "illustrators, designers, or studios losing commissions/work to AI",
        "music_voice_theft": "AI cloning musicians, songs, or voices without permission",
        "consent_training": "models trained on artists' work without consent / copyright grievance",
        "noai_activism": "#noAI / #SupportHumanArtists movement, 'human made not AI' identity content",
    },
    "deepfake": {
        "political_election": "deepfakes about politicians, elections, or political events",
        "scam_fraud": "impersonation scams, fake endorsements, financial fraud via AI",
        "nonconsensual_imagery": "non-consensual sexual/abusive deepfakes of real people",
        "celebrity_impersonation": "fake videos of celebrities not tied to fraud or politics",
        "misinfo_general": "general fake news / AI misinformation eroding trust in media",
    },
    "jobs": {
        "white_collar_tech": "office, clerical, coding, or tech jobs displaced by AI",
        "service_blue_collar": "customer service, retail, drivers, manual/service jobs displaced",
        "layoffs_news": "reporting specific company layoffs attributed to AI",
        "personal_loss": "creator says AI took their own job / career",
        "all_jobs_panic": "generalized fear that AI will take everyone's jobs",
    },
    "general_neg": {
        "ai_slop": "complaints about AI slop / low-quality AI content flooding the internet",
        "product_degradation": "a product got worse (search, chatbots, customer service) because of AI",
        "hate_quit": "'I hate AI' / 'why I quit AI' personal rejection",
        "bubble_skepticism": "AI is overhyped / a bubble / overrated",
        "antiai_identity": "#antiAI as a general identity/movement, not a specific complaint",
    },
    "environment": {
        "data_centers": "data centers, server farms, or GPU farms (siting, footprint, local impact)",
        "electricity_grid": "electricity demand, power grid strain, energy consumption",
        "water_usage": "water consumption / cooling",
        "carbon_climate": "carbon emissions / climate change framing",
        "general_resource": "generic 'AI is bad for the environment' without a specific resource",
    },
    "surveillance": {
        "facial_recognition": "facial recognition specifically",
        "tracking_spying": "general AI tracking / spying / 'AI is watching'",
        "data_privacy": "personal data being harvested / used to train AI",
        "state_corporate": "government or corporate surveillance programs",
    },
    "education": {
        "student_cheating": "students using AI to cheat / plagiarize",
        "learning_harm": "AI degrading learning, critical thinking, or literacy",
        "school_bans": "banning or policing ChatGPT in schools",
        "teacher_impact": "impact on teachers / teaching",
    },
    "regulation": {
        "pro_regulation": "explicit calls to regulate, ban, or legislate AI",
        "corporate_critique": "criticism of AI companies/labs (OpenAI bad, etc.)",
        "safety_oversight": "demands for safety standards / oversight",
        "econ_bubble": "economic / bubble framing for why AI needs reining in",
    },
    "xrisk": {
        "extinction_doom": "AI will kill / wipe out humanity",
        "loss_of_control": "AI is out of control / can't be stopped / takes over",
        "agi_superintelligence": "AGI or superintelligence framed as the threat",
        "personal_dread": "personal fear/dread, 'why I quit AI' out of existential worry",
    },
}

POSITIVE_SUBTAX = {
    "career_productivity": {
        "vibe_coding": "building apps/software with AI (Cursor, Copilot, 'vibe coding')",
        "agent_automation": "AI agents / automations / workflows doing tasks",
        "chatgpt_workflow": "ChatGPT/productivity tips, prompts, hacks for work",
        "income_side_hustle": "making money, side hustles, raises attributed to AI",
        "job_landed": "got a job or promotion thanks to AI skills",
        "tool_roundup": "round-ups / lists of AI tools for productivity",
    },
    "creative_tool": {
        "music_suno": "making music with AI (Suno, Udio)",
        "image_art": "making images/art (Midjourney, 'AI made me an artist')",
        "video_gen": "AI video generation (Sora, Runway, Veo)",
        "design_build": "designing or building a creative artifact with AI ('built this with AI')",
        "pro_ai_identity": "'I proudly use AI' creative identity / defense of AI art",
    },
    "synthetic_media_play": {
        "face_swap": "face swaps / face filters for fun",
        "lip_sync": "AI lip sync / talking photos",
        "animation": "AI animation / cartoons",
        "viral_edit": "viral AI edits / transitions",
        "funny_skit": "funny AI skits, memes, absurd AI clips",
    },
    "companion_affective": {
        "romantic": "AI girlfriend / boyfriend / romantic partner",
        "friendship": "AI as a friend / 'bestie'",
        "emotional_support": "AI for emotional support, coping, therapy",
        "character_roleplay": "character.ai / roleplay companions",
    },
    "education_learning": {
        "homework_help": "AI helping with homework / assignments",
        "study_aid": "AI for studying / exam prep",
        "concept_explainer": "AI explaining concepts better than other sources",
        "skill_learning": "using AI to learn a new skill or language",
    },
    "general_hype": {
        "future_optimism": "'AI is the future' broad optimism",
        "love_ai": "'I love AI' general enthusiasm",
        "mind_blown": "amazement / 'mind blown' at AI capabilities",
    },
    "model_stan": {
        "claude_stan": "praising Claude specifically",
        "chatgpt_stan": "praising ChatGPT / OpenAI specifically",
        "other_model_stan": "praising another named model (Gemini, Grok, etc.)",
        "generic_amazement": "model praise without a clear named model",
    },
    "breakthrough_science": {
        "medical_health": "medical / health AI breakthroughs",
        "research_discovery": "scientific research / discovery via AI",
        "future_promise": "promise of future scientific breakthroughs",
    },
}


def menu_text(subtax):
    return "\n".join(f"  - {code}: {desc}" for code, desc in subtax.items())


PROMPT = """You are sub-classifying a short {platform} video that has already been labelled with the top-level theme "{theme}" ({theme_desc}).

Pick the ONE sub-category that best fits this specific video. Choose "other" only if none fit.

Sub-categories:
{menu}
  - other: none of the above fit

Video metadata:
description: {description}
hashtags: {hashtags}
seed search keyword: {keyword}
evidence/transcript snippet: {evidence}

Respond with ONLY this format (no extra text):
subtopic: <code>
confidence: <high|medium|low>"""

THEME_DESC = {
    "art": "AI backlash about art / creative theft",
    "deepfake": "AI backlash about deepfakes / misinformation",
    "jobs": "AI backlash about jobs / displacement",
    "general_neg": "general anti-AI sentiment",
    "environment": "AI backlash about environment / energy",
    "surveillance": "AI backlash about surveillance / privacy",
    "education": "AI backlash about education",
    "regulation": "calls to regulate AI",
    "xrisk": "existential-risk backlash",
    "career_productivity": "positive: career / productivity wins",
    "creative_tool": "positive: creative-tool use",
    "synthetic_media_play": "positive: synthetic-media play",
    "companion_affective": "positive: AI companionship",
    "education_learning": "positive: education / learning",
    "general_hype": "positive: general hype",
    "model_stan": "positive: model fandom",
    "breakthrough_science": "positive: breakthrough science",
}

OUT_RE = re.compile(r"subtopic:\s*([a-z_]+).*?confidence:\s*(high|medium|low)", re.S | re.I)


def evidence_of(r):
    ev = (r.get("_evidence") or "").strip()
    if ev:
        return ev[:400]
    return (r.get("_transcript") or "").strip()[:400]


def classify_row(client, r, theme_key, subtax):
    theme = r[theme_key]
    prompt = PROMPT.format(
        platform="TikTok",
        theme=theme,
        theme_desc=THEME_DESC.get(theme, theme),
        menu=menu_text(subtax[theme]),
        description=(r.get("description") or "")[:600],
        hashtags=(r.get("hashtags") or "")[:200],
        keyword=r.get("_search_keyword") or "",
        evidence=evidence_of(r) or "(none)",
    )
    msg = client.messages.create(
        model=MODEL, max_tokens=40,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    m = OUT_RE.search(text)
    if not m:
        return "other", "low"
    sub = m.group(1).lower()
    if sub not in subtax[theme] and sub != "other":
        sub = "other"
    return sub, m.group(2).lower()


def run(corpus, limit):
    if corpus == "backlash":
        rows, _ = load_tiktok(only_2026=False)
        theme_key, subtax = "_topic_primary", BACKLASH_SUBTAX
    else:
        _, rows = load_tiktok(only_2026=False)
        theme_key, subtax = "_theme", POSITIVE_SUBTAX

    # Only themes we have a menu for (all of them, but guard anyway).
    rows = [r for r in rows if r.get(theme_key) in subtax and r.get("post_id")]

    out_path = REPO / f"tiktok/data/final/tiktok_{corpus}_subtopics.csv"
    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8", newline="") as f:
            for d in csv.DictReader(f):
                done[d["post_id"]] = d
    todo = [r for r in rows if r["post_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{corpus}] total={len(rows):,} done={len(done):,} todo_now={len(todo):,} "
          f"model={MODEL}")

    api_key = None
    envp = REPO / ".env"
    if envp.exists():
        for line in envp.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY"):
                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    client = anthropic.Anthropic(api_key=api_key)

    lock = threading.Lock()
    results = []
    errors = [0]

    def work(r):
        try:
            sub, conf = classify_row(client, r, theme_key, subtax)
            return (r["post_id"], r[theme_key], sub, conf)
        except Exception as e:
            with lock:
                errors[0] += 1
            if errors[0] <= 3:
                print("  error:", str(e)[:120])
            return None

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res:
                results.append(res)
            if i % 100 == 0:
                print(f"  ...{i}/{len(todo)}")

    # Append to sidecar.
    write_header = not out_path.exists()
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["post_id", "theme", "subtopic", "confidence"])
        for row in results:
            w.writerow(row)
    print(f"[{corpus}] wrote {len(results):,} rows -> {out_path.name} "
          f"(errors={errors[0]})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["backlash", "positive"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="print sample classifications")
    args = ap.parse_args()
    res = run(args.corpus, args.limit)
    if args.show and res:
        # Re-load to pair with descriptions for display.
        rows, _ = load_tiktok(only_2026=False) if args.corpus == "backlash" \
            else (None, None)
        if args.corpus == "positive":
            _, rows = load_tiktok(only_2026=False)
        by_id = {r["post_id"]: r for r in rows}
        print("\n--- sample classifications ---")
        for pid, theme, sub, conf in res[:25]:
            desc = (by_id.get(pid, {}).get("description") or "")[:70].replace("\n", " ")
            print(f"  [{theme} -> {sub} ({conf})] {desc}")


if __name__ == "__main__":
    main()
