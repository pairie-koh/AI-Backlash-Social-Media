# How We Built the AI-Backlash Video Corpus

*Pairie Koh · 2026-04-28*

## Search and collection

We assembled a list of **68 AI-related keywords** spanning the major grievance categories — job displacement, environmental cost, art and music theft, deepfakes and misinformation, surveillance and privacy, AI in education, regulation and bans, and general anti-AI sentiment ("I hate AI", "AI slop", "AI is ruining everything", `#noAI`, `#antiAI`, `#stopAIart`, "real artist not AI", and similar). The keyword list (`KEYWORDS.md`) was deliberately designed to be **high-recall**: we wanted to over-collect candidates and let the downstream filters do the precision work. Many keywords are themselves backlash-y in framing ("AI killed my career", "I quit AI", "stop AI art"), but several are neutral seeds that surface both pro- and anti-AI content ("AI regulation", "AI data centers", "AI surveillance"), which lets the same pipeline produce both the YES and NO sides of each topic.

We ran the full keyword × platform sweep on **Bright Data's YouTube and TikTok scrapers** (dataset IDs `gd_lk56epmy2i5g7lzu0k` and `gd_lu702nij2f790tmv9h`), collecting 139 snapshots over a window ending 2026-04-26 with no per-keyword cap (`limit_per_input: NONE`). The raw scrape returned **23,865 YouTube videos and 8,108 TikToks** (31,973 total), each tagged with the keyword that surfaced it. Standard fields were kept: title, description, transcript or auto-generated captions, view/like/comment counts, channel metadata, post date, region (TikTok), and verification status.

## Identification — three filter passes, recall first then precision

Because the keyword sweep over-collects, the corpus needed substantial filtering. We used a deliberately staged three-pass design that prioritizes recall in the early passes and precision in the late ones:

**Pass 1 — Keyword-level relevance filter (V1).** A Sonnet-class LLM read each video's description and transcript (truncated for cost) and answered a single YES/NO question: does this video express criticism, concern, or backlash about AI? The prompt enumerated the legitimate backlash categories and the obvious non-backlash patterns (pro-AI hype, neutral tutorials, AI-generated content uploaded by AI fans, clickbait that pivots to product promotion). This pass is intentionally permissive — its job is to drop the obviously-off-topic content (gaming reviews, cooking videos, AI tool tutorials with neutral framing) and keep anything plausibly relevant.

**Pass 2 — Strict re-classifier with chain-of-thought (V2).** Every V1 YES video was re-evaluated by a stricter prompt with eight few-shot examples (four YES, four NO) that anchor the harder edge cases — clickbait fear-bait titles followed by sales pivots, "no AI" used as a marketing badge in unrelated content, balanced news framing with no creator stance, "adapt to AI" optimism dressed as concern. The model produced a brief reasoning trace, a 1–10 confidence score, and a final YES/NO. The threshold for YES was set at score ≥ 6, biased toward precision: when the stance was genuinely unclear, the prompt instructed the model to default to NO. This pass demoted thousands of V1 YES rows that were actually pro-AI tutorials, finance pitches using AI-fear hooks, or both-sides explainers.

**Pass 3 — Manual agent audit at the borderline (V3).** Even with a strict prompt, the lowest-confidence buckets (score = 6 and score = 7) had a much higher false-positive rate than the high-confidence buckets — eyeball samples showed roughly 50–65% FP at score 6, ~20–40% at score 7, and under 10% at score 8 and above. Rather than trust the LLM at these borderlines, we **chunked the score 6 and 7 rows and dispatched parallel reviewer agents** to read each video's title, description, full transcript head-and-tail, and search keyword, applying a single common-sense test:

> *"Would you cite this video as evidence of AI backlash sentiment in a research paper studying public attitudes toward AI?"*

If the answer was no — because the video was a finance/career pitch using AI fear as a hook, a marketing badge in unrelated content, a balanced explainer with no personal stance, a comedy bit using AI as a plot device, an AI tool tutorial, or a "use AI smartly" optimism piece — the row was flagged for demotion. Each agent wrote a short reason for every verdict, preserving the audit trail. Across both platforms this manual pass reviewed 4,038 borderline videos and demoted 1,629 false positives that the V2 LLM had let through (1,388 YouTube + 241 TikTok). Higher-confidence buckets (score 8+) were left as V2 classified them since spot-checks showed those buckets to already be at acceptable precision.

## Result

The final corpus is **5,937 YouTube and 2,416 TikTok videos**, with each row carrying a `_backlash_final` column reflecting all three passes. Estimated residual false-positive rate, validated by independent spot-check, is approximately **5% on each platform** — within the precision range typical for LLM-curated cultural-discourse corpora. Every drop and keep decision from the manual audit is preserved in `final_data/score6_audit_aggregate.json` and `final_data/score7_audit_aggregate.json` along with a one-line reason per video, so the labeling can be inspected, contested, or re-applied with a different cutoff.
