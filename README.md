# AI Backlash on Social Media

Measuring public backlash to AI on TikTok and YouTube — job displacement, environmental impact, creative theft, privacy, education, safety, and more.

## Research Questions

1. **How much AI backlash content exists** on TikTok and YouTube?
2. **What themes dominate** the backlash? (jobs, environment, creative, privacy, etc.)
3. **How has backlash evolved over time?** (temporal trends over the past 12 months)
4. **Which AI companies/products** receive the most criticism?
5. **How intense is the sentiment?** (mild concern vs. outrage)

## Pipeline

```
1. COLLECT (Bright Data API)
   → TikTok + YouTube raw JSON via 65 keyword searches

2. PASS 1: KEYWORD FILTER (regex, free)
   → Keep only videos mentioning AI-related terms

3. PASS 2: LLM BACKLASH CLASSIFIER (Claude Sonnet)
   → YES/NO: Does this video express criticism/concern about AI?

4. TRANSCRIBE (OpenAI Whisper, TikTok only)
   → Audio → text for TikTok videos without captions

5. THEME CLASSIFICATION (Claude Sonnet)
   → JOBS | ENVIRONMENT | CREATIVE | PRIVACY | EDUCATION | SAFETY | MISINFO | GENERAL

6. DETAIL EXTRACTION (Claude Sonnet, structured JSON)
   → backlash_themes, specific_concerns, companies_mentioned,
     proposed_actions, sentiment_intensity

7. EXPORT → Clean CSVs + transcripts
```

## Project Structure

```
AI-Backlash-Social-Media/
├── tiktok/
│   ├── data/
│   │   ├── raw/                          # tiktok_raw.json
│   │   ├── whisper_transcripts/          # Whisper transcriptions
│   │   └── tiktok_transcripts/           # Platform captions
│   └── scripts/
│       ├── filter_by_topic.py            # Pass 1 + Pass 2
│       ├── classify_backlash_theme.py    # Theme classification
│       ├── extract_details.py            # Structured extraction
│       └── whisper_transcribe.py         # Audio → text
│
├── youtube/
│   ├── data/
│   │   └── raw/                          # youtube_raw.json
│   └── scripts/
│       └── filter_by_topic.py            # Pass 1 + Pass 2
│
├── final_data/                           # Clean exports
├── replication/                          # Analysis scripts
├── trends/                               # Temporal analysis
├── KEYWORDS.md                           # 65 search keywords
└── README.md
```

## Usage

### 1. Collect data (Bright Data)

Trigger Bright Data collection using the keywords in `KEYWORDS.md`.

### 2. Filter & classify (TikTok)

```bash
export OPENROUTER_API_KEY=...

# Pass 1 (keyword) + Pass 2 (LLM backlash classifier)
python tiktok/scripts/filter_by_topic.py

# Classify backlash theme
python tiktok/scripts/classify_backlash_theme.py

# Extract structured details
python tiktok/scripts/extract_details.py

# Transcribe audio (requires OPENAI_API_KEY)
export OPENAI_API_KEY=...
python tiktok/scripts/whisper_transcribe.py
```

### 3. Filter & classify (YouTube)

```bash
python youtube/scripts/filter_by_topic.py
```

## Backlash Themes

| Theme | Description |
|-------|-------------|
| JOBS | Job displacement, automation, layoffs, unemployment |
| ENVIRONMENT | Data centers, energy, water, carbon footprint |
| CREATIVE | Art theft, AI art/music/writing, voice cloning |
| PRIVACY | Surveillance, tracking, facial recognition |
| EDUCATION | Cheating, plagiarism, academic integrity |
| SAFETY | Existential risk, alignment, regulation calls |
| MISINFO | Deepfakes, misinformation, fake news, scams |
| GENERAL | Broad anti-AI sentiment, tech backlash |

## Cost Estimate

| Component | Estimate |
|-----------|----------|
| Bright Data (TikTok + YouTube) | ~$30-40 |
| LLM classification (3 passes) | ~$30-50 |
| Whisper transcription (TikTok) | ~$12-20 |
| **Total** | **~$70-110** |
