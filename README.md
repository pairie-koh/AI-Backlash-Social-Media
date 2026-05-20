# AI Sentiment on Social Media (TikTok + YouTube)

Measuring public sentiment about AI on TikTok and YouTube — both the **backlash** corpus (jobs, environment, creative theft, privacy, deepfakes, safety) and a parallel **positive-sentiment** corpus (creative tools, productivity, synthetic-media play, education, AI companions). Collected via Bright Data, classified with Claude Sonnet, audited with Whisper transcripts for TikTok.

## Research Questions

1. **How much AI-related content exists** on TikTok and YouTube, and what share is backlash vs. positive?
2. **What themes dominate** each side? (jobs, environment, creative, privacy vs. creative tools, productivity, companions, etc.)
3. **How does the mix differ by platform?** (TikTok is artist-led; YouTube is news/jobs/productivity-led)
4. **What is the share of edge-of-discourse content** — existential risk on the backlash side, breakthrough science on the positive side?
5. **Which themes break through to the broadest audience?**

## Pipeline

```
1. COLLECT (Bright Data API)
   → TikTok + YouTube raw JSON via parallel keyword lists
     (KEYWORDS.md for backlash, KEYWORDS_positive.md for positive)

2. PASS 1: LLM FILTER (Claude Sonnet via OpenRouter)
   → YES/NO: Does this video express backlash / positive sentiment?

3. PASS 2: STRICT RE-CLASSIFIER (Claude Sonnet, chain-of-thought)
   → 0–10 confidence score on the YES classification

4. PASS 3: MANUAL-AGENT AUDIT (score=6 and score=7 buckets only)
   → Per-chunk verdicts saved in {tiktok,youtube}/data/score{6,7}_audit/

5. TRANSCRIBE (OpenAI Whisper, TikTok only)
   → Audio → text for TikTok videos. YouTube transcripts come inline
     from Bright Data and live in the CSV _transcript_text column.

6. THEME / TOPIC CLASSIFICATION (Claude Sonnet)
   → Backlash: art | deepfake | jobs | environment | surveillance |
              general_neg | xrisk | education | regulation
   → Positive: creative_tool | career_productivity | synthetic_media_play |
              education_learning | companion_affective | model_stan |
              general_hype | breakthrough_science

7. EXPORT → Clean CSVs (one per platform × sentiment)
```

## Project Structure

```
AI-Backlash-Social-Media/
├── KEYWORDS.md                              # backlash keywords
├── KEYWORDS_positive.md                     # positive-sentiment keywords
├── README.md
│
├── tiktok/
│   ├── data/
│   │   ├── tiktok_ai_backlash_filtered.csv          # Pass 1 output
│   │   ├── tiktok_ai_backlash_verified.csv          # Pass 2 output
│   │   ├── tiktok_ai_backlash_final.csv             # Pass 3 (audited) — use this
│   │   ├── tiktok_positive_positive_filtered.csv    # positive Pass 1
│   │   ├── tiktok_positive_classified.csv           # positive themes — use this
│   │   ├── whisper_transcripts/                     # 13,131 Whisper outputs
│   │   ├── score6_audit/  score7_audit/             # per-chunk audit verdicts
│   │   ├── raw/                                     # raw Bright Data (gitignored)
│   │   └── *_progress.json / *.log                  # run state
│   └── scripts/
│       ├── filter_by_topic.py                       # Pass 1 + Pass 2 backlash
│       ├── classify_topics.py                       # theme classification
│       ├── verify_yes_strict.py / verify_yes_round3.py
│       ├── whisper_transcribe.py
│       └── test_round3_sample.py
│
├── youtube/
│   ├── data/
│   │   ├── youtube_ai_backlash_*.csv                # NOT IN GIT (>100 MB each)
│   │   ├── youtube_positive_*.csv                   # NOT IN GIT (>100 MB)
│   │   ├── score6_audit/  score7_audit/             # per-chunk audit verdicts
│   │   ├── raw/                                     # raw Bright Data (gitignored)
│   │   └── *_progress.json / *.log
│   └── scripts/
│       ├── filter_by_topic.py
│       ├── classify_topics.py
│       ├── verify_yes_strict.py / verify_yes_round3.py
│       ├── audit_v2_quality.py / validate_new_v2.py
│       └── test_round3_sample.py
│
├── scripts/                                  # cross-platform / collection
│   ├── collect_brightdata.py                 # Bright Data trigger
│   ├── classify_register_theme.py            # positive theme classifier
│   ├── smoke_xrisk.py                        # xrisk audit smoke test
│   └── brightdata_snapshots*.json            # snapshot IDs
│
├── _pilot_batches/  _audit_batches/  _meta_audit_batches/
│                                             # JSON batches from the audit pipeline
│
├── final_data/
│   ├── score6_audit_aggregate.json
│   └── score7_audit_aggregate.json
│
└── analysis/                                 # write-ups + figures
    ├── descriptive_summary.md                # full 12-month overview
    ├── methodology.md                        # filter/audit pipeline details
    ├── research_idea.md
    ├── positive_vs_backlash_memo.md          # 12-month memo
    ├── positive_vs_backlash_2026_memo.md     # ← 2026-only memo (current)
    ├── x_thread_draft.md                     # X thread copy
    ├── plot_2026_x_share.py                  # makes the two 2026 graphs
    ├── plot_positive_vs_backlash.py          # makes the 12-month composite
    ├── plot_overall_over_time.py
    ├── plot_topics_over_time.py
    ├── plot_engagement_by_topic.py
    ├── backlash_topics_2026.png              # ← key graph (backlash)
    ├── positive_themes_2026.png              # ← key graph (positive)
    ├── positive_vs_backlash_summary.png      # 12-month composite (superseded)
    ├── datacenter_callout.png                # data-center share callout
    ├── overall_over_time.png
    ├── topics_over_time.png
    └── engagement_by_topic.png
```

## Files NOT in this git repo

Five YouTube CSVs exceed GitHub's 100 MB per-file hard limit and live outside git (Drive/Zenodo to be linked here):

| File | Size |
|---|---|
| `youtube/data/youtube_ai_backlash_filtered.csv` | 419 MB |
| `youtube/data/youtube_ai_backlash_verified.csv` | 419 MB |
| `youtube/data/youtube_ai_backlash_final.csv` | 419 MB |
| `youtube/data/youtube_positive_positive_filtered.csv` | 369 MB |
| `youtube/data/youtube_positive_classified.csv` | 207 MB |

Plus raw Bright Data scrapes (1.5–1.7 GB each) in `*/data/raw/`. Only `_final.csv` (backlash) and `_classified.csv` (positive) are needed for downstream analysis; the others are pipeline intermediates that can be regenerated from raw.

## Usage

### Setup

```bash
export OPENROUTER_API_KEY=...      # for Claude Sonnet classification
export OPENAI_API_KEY=...          # for Whisper transcription (TikTok)
```

### Backlash pipeline

```bash
# 1. Collect (uses brightdata_snapshots*.json for snapshot IDs)
python scripts/collect_brightdata.py

# 2. Filter + verify (per platform)
python tiktok/scripts/filter_by_topic.py
python tiktok/scripts/verify_yes_strict.py
python youtube/scripts/filter_by_topic.py
python youtube/scripts/verify_yes_strict.py

# 3. Theme classification
python tiktok/scripts/classify_topics.py
python youtube/scripts/classify_topics.py

# 4. Transcribe TikTok audio
python tiktok/scripts/whisper_transcribe.py
```

### Positive-sentiment pipeline

Same steps, using `KEYWORDS_positive.md` and the positive variants of the filter / classify scripts (`*_positive_*` output files).

### Generate the 2026 graphs

```bash
python analysis/plot_2026_x_share.py
# → analysis/backlash_topics_2026.png
# → analysis/positive_themes_2026.png
```

## Headline numbers (2026 only, TikTok + YouTube pooled)

- **Backlash:** 3,668 videos. Top topics: jobs 20%, deepfakes 19%, art 18%, environment 12%, general anti-AI 10%. Existential risk only **4.3%**.
- **Positive:** 7,099 videos (~2× the backlash volume). Top themes: career/productivity 34%, creative tools 29%, synthetic-media play 13%, education 7%. Breakthrough science only **2.1%**.
- **Data centers:** 23% of YouTube backlash videos explicitly mention a data center, server farm, or GPU farm — more than double the share classified as "environment."

See `analysis/positive_vs_backlash_2026_memo.md` for the full write-up.

## Cost Estimate (full pipeline, both corpora)

| Component | Estimate |
|---|---|
| Bright Data (TikTok + YouTube, both corpora) | ~$60–80 |
| LLM classification (filter + verify + theme passes) | ~$60–100 |
| Whisper transcription (TikTok, both corpora) | ~$25–40 |
| **Total** | **~$145–220** |
