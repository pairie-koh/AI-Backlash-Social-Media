# Field Notes — AI Backlash & Positive Sentiment on TikTok and YouTube (2026)

**Your name + location:** Pairie Koh — [City, Country]

**Research headline:** When the AI debate leaves Twitter, where do existential risk and breakthrough science actually go?

## The question

AI discourse on Twitter is dominated by existential-risk and breakthrough-science narratives, but we didn't know whether those frames travel. What does AI sentiment actually look like on the video platforms where most of the public encounters AI — and which themes carry both the backlash and the enthusiasm?

## What we built / did

Throughout 2026, Professor Andy Hall and I scraped AI-related TikTok and YouTube videos via Bright Data using parallel backlash and positive-sentiment keyword lists, then classified each video with Claude Sonnet in three passes: a YES/NO sentiment filter, a strict re-classifier with chain-of-thought confidence scores, and a theme assignment across nine backlash and eight positive themes. TikTok audio was transcribed with Whisper for audit.

## What we found

We classified 3,668 backlash videos and 7,099 positive videos. Five themes accounted for ~80% of all backlash content — jobs, deepfakes/misinformation, art/creative theft, environment, and general anti-AI. The platform split is sharp: TikTok backlash is art-dominated (29%, carried by the #noAI and #stopAIart artist community), while YouTube backlash leans jobs (24%) plus deepfakes (20%) — together nearly half the platform. Data centers are mentioned in 18% of backlash content and are emerging as the operational target of AI opposition. Positive content concentrates on career/productivity wins, creative-tool demos, synthetic-media play, and education — TikTok skews career (28%), synthetic-media play (22%), creative tools (18%), and AI companions (8%); YouTube positives are two-thirds career-productivity (37%) plus creative tools (34%). Big caveat: a large share of positive content is promotional (affiliate marketing, sponsored tool reviews, course funnels), so it likely overstates organic enthusiasm. The most surprising result: existential risk is only **4.3%** of backlash and breakthrough science only **2.1%** of positive sentiment — the two narratives that dominate elite AI Twitter are nearly invisible on the platforms where the broader public actually watches AI content.

## Where we're going

Next we'll build a filter to separate promotional from organic positive content so we can recover a genuine sentiment baseline, and track whether the data-center backlash translates into local political mobilization.

## One image or chart

`analysis/backlash_topics_2026.png` and `analysis/positive_themes_2026.png`, side-by-side. *Theme shares for AI backlash (left) and AI positive sentiment (right) on TikTok and YouTube in 2026. Notice how art, jobs, deepfakes, and data centers dominate the backlash while existential risk barely registers — and how career/productivity and creative-tool content dominate the positive side while breakthrough-science framing is essentially absent.*

## Anything we should link to?

- Repo: https://github.com/pairie-koh/AI-Backlash-Social-Media
- 2026 memo: `analysis/positive_vs_backlash_2026_memo.md`
- Methodology: `analysis/methodology.md`
