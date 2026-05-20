# AI Backlash on YouTube and TikTok — Descriptive Summary

*Pairie Koh · 2026-04-28*

## 1. Dataset

Bright Data scraped 31,973 videos across 68 AI-related keywords (collection completed 2026-04-26): 23,865 YouTube videos and 8,108 TikToks. After a three-pass filter (V1 LLM keyword filter → V2 strict re-classifier with chain-of-thought → manual agent audit of the lowest-confidence buckets), the final-YES set — videos that genuinely express AI backlash sentiment — contains:

- **5,937 YouTube videos** (24.9% of scraped)
- **2,416 TikTok videos** (29.8% of scraped)

Estimated residual false-positive rate is ~5% on each platform, validated by spot-check.

## 2. Sampling caveat — read this before any temporal claim

**Bright Data's keyword search over-represents recent posts.** Monthly volume on the YES set:

| | 2024 avg | 2025-Q4 avg | 2026-03 | 2026-04 (partial) |
|---|---|---|---|---|
| YouTube YES | ~50/mo | ~325/mo | 542 | 1,080 |
| TikTok YES | ~10/mo | ~135/mo | 279 | 650 |

The April 2026 spike — five times the prior month on YouTube — reflects the scrape window, not real-world growth in posting. Raw counts cannot be used for temporal trends without a date-stratified denominator.

**However**, the ratio of final-YES to total-scraped is **stable at 25–33% across the entire 2.5-year window** for both platforms. The share of AI-related content that is backlash isn't dramatically shifting; the volume of AI-related content on the platforms is.

**Implication for analysis**: use YES *rate* (final-YES ÷ total scraped per period), not raw counts, for any time-trend claim. Footnote or drop 2026-04 (partial month, ends 2026-04-26).

## 3. Topic mix

Keyword categories, % of final-YES:

| Topic | YouTube | TikTok |
|---|---|---|
| General anti-AI ("hate", "ruining", "slop") | 23.0% | 21.8% |
| Art / creative theft | 14.5% | **23.6%** |
| Deepfakes / misinformation | 15.1% | 10.6% |
| Jobs / displacement | 13.8% | 7.2% |
| Environment / energy | 9.2% | 9.5% |
| Surveillance / privacy | 8.1% | 5.4% |
| Education (cheating, deskilling) | 3.0% | 3.5% |
| Regulation calls | 2.6% | 4.1% |

**The platform difference is structural.** TikTok backlash is artist-led — art theft is the single largest category (24%), driven by an active `#noai` / `#antiai` / `#stopaiart` creative community. YouTube backlash skews toward news, expert commentary, and labor concerns (deepfakes 15%, jobs 14%).

## 4. Reach

| | YouTube | TikTok |
|---|---|---|
| Total views/plays in YES set | **1.04B** | **857M** |
| Median views/plays per video | 805 | 3,972 |
| 90th percentile | 318,891 | 604,800 |
| 99th percentile | 3,150,843 | 8,200,000 |
| Max | 19,535,285 | 18,600,000 |
| Total likes / diggs | 35.2M | 90.9M |

Combined reach is roughly **1.9 billion views** of AI-backlash content. The distribution is heavy-tailed — the median video has fewer than 1k views on YouTube but the top of the distribution reaches viral scale (Hinton interviews, Diary of a CEO ⨯ Mo Gawdat, John Oliver "Last Week Tonight" on AI, "AI is ruining the internet").

## 5. Creator structure — long-tail, not concentrated

| | YouTube | TikTok |
|---|---|---|
| Unique creators | 4,579 | 2,171 |
| Mean videos per creator | 1.30 | 1.11 |
| Top 1% of creators' share of videos | 9.0% | 4.7% |
| Verified creators | 27.0% | 10.8% |

This is the most striking structural finding: **AI backlash on both platforms is a grassroots, long-tail phenomenon, not a few-loud-voices phenomenon.** Most backlash videos come from creators who post one such video. The top 1% of creators contribute fewer than 10% of videos. This shape rules out a "professional anti-AI influencer industry" hypothesis and supports a "broad cultural reaction" hypothesis.

Verified-account share (27% YT, 11% TT) is much lower than for institutional or news content, consistent with the grassroots reading.

## 6. Geography (TikTok)

| Region | Share of YES |
|---|---|
| US | 50.8% |
| UK | 14.1% |
| Australia | 4.9% |
| Canada | 3.4% |
| Philippines | 2.9% |
| Germany | 2.6% |
| France | 2.1% |

About 73% of the TikTok YES set comes from anglophone countries (US/UK/AU/CA). Western backlash narratives dominate the sample. This may reflect Bright Data's English-keyword-driven discovery rather than absolute country-level prevalence.

## 7. Headline takeaways

1. **AI backlash content on YouTube and TikTok is a real phenomenon at meaningful scale** — ~8,400 backlash videos with 1.9B combined views over the last 2-3 years.
2. **The mix of grievances differs by platform.** YouTube is news/jobs/deepfake-heavy; TikTok is artist-led.
3. **It is grassroots, not concentrated** — long-tail creator distribution, mostly unverified accounts.
4. **The cleanest temporal signal is the YES *rate*, not raw counts**, because the scrape over-represents recent posts. Stable ~25–33% YES rate across 2.5 years is itself an interesting finding: the *share* of AI-keyword content that is backlash has not exploded — but the volume on the platforms has.

## 8. Next analyses (priority order)

1. **Topic × time, using YES rate as denominator.** Which kinds of backlash track which real-world events? (Hollywood strikes, GPT-4o launch, Sora 2, NYT v. OpenAI, Microsoft/Meta layoff waves.)
2. **Engagement per topic.** Median and total views by topic — which grievances actually break through to attention?
3. **Score=10 subset.** Strong-stance videos (~200 across both platforms) as a qualitative anchor for the most explicit anti-AI rhetoric.
4. **Top-20 creators by reach.** Who are the loudest voices on each platform? (Even if the long tail dominates, the top voices set discourse.)
5. **Comment-to-view ratio as a controversy proxy.** High ratio = the video is sparking debate, not just being watched.
6. **Hashtag co-occurrence on TikTok.** `#noai` co-traveling hashtags map the artist-resistance community structure.
7. **Cross-platform topic gap.** What does TikTok talk about that YouTube barely covers, and vice versa?

## 9. Files

- `youtube/data/youtube_ai_backlash_final.csv` — 23,865 rows, `_backlash_final` column carries the audited verdict
- `tiktok/data/tiktok_ai_backlash_final.csv` — 8,108 rows, same schema
- `final_data/score6_audit_aggregate.json` — manual audit, score=6 rows
- `final_data/score7_audit_aggregate.json` — manual audit, score=7 rows
- `*/data/score{6,7}_audit/chunk_NN_verdicts.json` — per-chunk reasoning preserved
