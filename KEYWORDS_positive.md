# AI Positive Sentiment Keywords

Search keywords for collecting TikTok and YouTube videos expressing enthusiasm, gratitude, or advocacy for AI. Mirrors `KEYWORDS.md` in *structure* (parallel categories, same per-category counts where retained) but uses **organic TikTok/YouTube vernacular** (Jan 2026) rather than press-release / corporate-deck phrasing — model-stan content (Claude goated, GPT cooked, Sora insane), AI-companion content (girlfriend, therapist, character.ai), vibe-coding / agent-demo content, viral prompt-share content, and first-person affective content (AI changed my life, AI helped me cope).

**Total: 62 keywords** across 9 categories.

> **Methodology note — read before sweeping.**
>
> 1. **Environment / climate dimension is excluded by design.** The backlash list has 6 environment keywords (data centers, energy, water, carbon, climate, grid) but the positive twin is press-release / Nvidia-keynote content with no grassroots vernacular. We are not collecting positive environment data — the environment dimension drops out of head-to-head comparisons. Backlash environment data still exists; analyze it stand-alone.
>
> 2. **Promo-vs-organic contamination.** Positive AI discourse has two flavors: promotional/creator-economy content (affiliate marketing, course funnels, brand pumping) and organic personal sentiment (first-person experience, model-stan, affective use). The backlash side has no promotional analogue. This list leans toward organic but contamination is unavoidable on some keywords (esp. `#8 AI side hustle`, `#10 Midjourney art`).
>
> 3. **Mitigation lives at the filter stage, not collection.** The Pass 2 LLM classifier should label each video on a `register` axis — `promotional / genuine_sentiment / news / mixed` — and downstream analysis should report results both unfiltered AND restricted to `genuine_sentiment`. Apply the same `register` label to the backlash corpus on a re-run for symmetry. The promo-vs-organic ratio is itself an interesting finding (asymmetric supply chains for pro vs. anti AI speech).
>
> 4. **Synthetic Media (Cat 6) will be news-light but vernacular-light too.** "Funny AI video" / "AI lip sync" pull viral creative content but the construct is fuzzy ("playful AI use" is not the same as "positive AI sentiment"). The LLM filter handles this; we accept the noise.

---

## 1. Job Augmentation & Builder Wins (9 keywords)
| # | Keyword | Rationale |
|---|---------|-----------|
| 1 | `AI helped my career` | First-person positive narrative — mirror of "AI killed my career" |
| 2 | `vibe coding` | Dominant developer-positive vernacular (2025–26) — mirror of "AI taking jobs" framing |
| 3 | `AI hiring boom` | News framing — mirror of "AI layoffs" |
| 4 | `AI made my job easier` | First-person augmentation — mirror of "will AI take my job" |
| 5 | `ChatGPT made me productive` | Brand-specific positive — mirror of "ChatGPT replacing jobs" |
| 6 | `built this with AI` | Maker/builder voice — mirror of "AI replacing workers" |
| 7 | `Cursor changed my life` | First-person developer-tool love — mirror of "AI replace me" |
| 8 | `AI side hustle` | Hustle/monetization vernacular (heavy promo bleed; LLM filter must label) — mirror of "AI job loss" discourse |
| 9 | `agent did my job for me` | Agent-demo register (Manus / Operator / Claude Computer Use) — mirror of "AI automation jobs" |

## 2. Creative Tools & Generative Showcase (8 keywords)
| # | Keyword | Rationale |
|---|---------|-----------|
| 10 | `Midjourney art` | Tool-name showcase — mirror of "AI art theft" |
| 11 | `AI made me an artist` | Emotional vernacular inverse — mirror of "AI stealing art" |
| 12 | `Sora is insane` | Viral video-model reaction — mirror of "AI vs artists" |
| 13 | `Suno cooked` | Model-stan music register — mirror of "stop AI art" |
| 14 | `Suno AI music` | Direct music-tool mirror of "AI music theft" |
| 15 | `AI voice cover` | Creative voice use — mirror of "AI voice cloning danger" |
| 16 | `Runway AI` | Major video-AI tool brand — mirror of "AI copyright" |
| 17 | `made art with AI` | First-person creator pride — mirror of "protect artists from AI" |

## 3. AI Companions & Personal Use (5 keywords)
| # | Keyword | Rationale |
|---|---------|-----------|
| 18 | `AI girlfriend` | Companion content — vernacular intimate-AI inversion of "AI watching me" |
| 19 | `character ai` | Companion-app brand — mirror of "AI facial recognition" |
| 20 | `Claude is my therapist` | AI as trusted confidant — mirror of "AI spying" |
| 21 | `my AI bestie` | Affective companion register — mirror of "AI surveillance" |
| 22 | `talking to ChatGPT instead of therapy` | Mental-health positive vernacular — mirror of "AI tracking" |

## 4. Education & Learning Wins (5 keywords)
| # | Keyword | Rationale |
|---|---------|-----------|
| 23 | `ChatGPT helped me study` | Brand-specific, very high volume — mirror of "ChatGPT cheating" |
| 24 | `AI explained it better` | Pedagogy positive — mirror of "ban ChatGPT school" |
| 25 | `ChatGPT got me through finals` | Academic positive vernacular — mirror of "AI ruining education" |
| 26 | `AI homework help` | Vernacular dual-valence (LLM filter sorts) — mirror of "AI cheating school" |
| 27 | `AI helped me learn` | First-person learning positive — mirror of "AI plagiarism" |

## 5. Personal Affect & Life Change (5 keywords)
| # | Keyword | Rationale |
|---|---------|-----------|
| 28 | `AI changed my life` | Personal positive impact — mirror of "AI dangerous" |
| 29 | `AI is the future` | Strongest pro-AI counter-policy framing — mirror of "ban AI" |
| 30 | `AI helped me cope` | Affective positive — mirror of "AI out of control" |
| 31 | `AI is amazing` | Broad positive sentiment — mirror of "AI regulation" |
| 32 | `I'm cooked without AI` | Vernacular dependency framing — mirror of "stop AI" |

## 6. Synthetic Media & Creative Edits (5 keywords)
| # | Keyword | Rationale |
|---|---------|-----------|
| 33 | `funny AI video` | Creative use of same tech — mirror of "AI deepfake" |
| 34 | `AI lip sync` | Creative voice/face edits — mirror of "deepfake danger" |
| 35 | `AI animation` | Generative-animation showcase content — mirror of "AI scam" |
| 36 | `AI face swap funny` | Playful synthetic media — mirror of "AI fake news" |
| 37 | `AI edit viral` | Viral creative AI use — mirror of "AI misinformation" |

## 7. Model-Stan & Capability Praise (7 keywords)
| # | Keyword | Rationale |
|---|---------|-----------|
| 38 | `Claude is goated` | Model-stan praise — mirror of "AI slop" (THE dominant pro-AI vernacular) |
| 39 | `AI mind blown` | Engagement reaction — mirror of "AI spam" |
| 40 | `GPT cooked` | Model-stan brand register — mirror of "AI garbage" |
| 41 | `Sonnet ate` | Specific-model fan register — mirror of "Google AI ruined search" |
| 42 | `this prompt is insane` | Viral prompt-share register — mirror of "AI making internet worse" |
| 43 | `AI is incredible` | Broad positive sentiment — mirror of "AI ruining everything" |
| 44 | `AI is so good now` | Quality-affirmation framing — mirror of "AI generated content problem" |

## 8. Everyday Wins & Brand Love (8 keywords)
| # | Keyword | Rationale |
|---|---------|-----------|
| 45 | `AI saved me hours` | Productivity gratitude — mirror of "AI customer service sucks" |
| 46 | `AI saved me time` | Universal everyday win — mirror of "can't talk to a human" |
| 47 | `ChatGPT getting smarter` | Product-improvement discourse — mirror of "ChatGPT getting worse" |
| 48 | `Claude is the best` | Company/model-specific praise — mirror of "OpenAI bad" |
| 49 | `Cursor is amazing` | Developer tool love (real usage register, not vendor copy) — mirror of "Copilot sucks" |
| 50 | `AI chatbot helpful` | Direct mirror of "AI chatbot useless" |
| 51 | `AI underrated` | Direct mirror of "AI overrated" |
| 52 | `AI is the real deal` | Anti-skeptic vernacular — mirror of "AI bubble" |

## 9. Pro-AI Identity & Hashtags (10 keywords)
| # | Keyword | Rationale |
|---|---------|-----------|
| 53 | `#proAI` | Direct hashtag mirror of "#antiAI" |
| 54 | `#AIenthusiast` | Pro-AI community tag — mirror of "#noAI" |
| 55 | `#FutureWithAI` | Affirmative movement tag — mirror of "#SupportHumanArtists" |
| 56 | `proudly use AI` | Authenticity inversion — mirror of "human made not AI" |
| 57 | `#AIart` | Contested space — captures both sides (intentionally duplicates backlash list) |
| 58 | `built with AI` | Affirmative badge content — mirror of "no AI used" |
| 59 | `team AI` | Community identity — mirror of "real artist not AI" |
| 60 | `AI underhyped` | Direct hype mirror of "AI overhyped" |
| 61 | `why I love AI` | Personal-conversion narrative — mirror of "why I quit AI" |
| 62 | `I love AI` | Raw sentiment mirror of "I hate AI" |

---

## What this list is and isn't

**Is:** A vernacular-tuned positive-sentiment seed list designed to recall organic Jan-2026 pro-AI content on TikTok/YouTube — model-stan, AI-companion, vibe-coding, agent-demo, prompt-share, and first-person affective registers.

**Isn't:** A construct-validated comparison set. It's a 1:1 *structural* mirror of the backlash list with two known asymmetries: the environment dimension is dropped (no organic positive vernacular exists), and promotional content will bleed into several keywords. Both asymmetries are documented and pushed to the filter stage rather than the collection stage.

**Compared with the backlash list:** 62 keywords vs. 68. Cat 2 (Environment) excluded. Otherwise category structure parallels the backlash side; per-category counts identical where retained.

---

## Collection Parameters

- **Platform**: TikTok + YouTube (via Bright Data) — same dataset IDs as backlash sweep
- **Time filter**: Last 12 months (April 2025 – April 2026) — must match backlash window exactly
- **Limit per keyword**: NONE (matches backlash sweep — no `limit_per_input`)
- **Expected raw yield**: ~9,000–22,000 videos per platform
- **Deduplication**: By `post_id` (TikTok) / `video_id` (YouTube)

## Estimated Bright Data Cost

- 62 keywords × ~200 results ≈ ~12,400 records per platform
- ~24,800 total records × $0.0015/record = **~$37**
- May run higher than backlash (~$48 actual) on richer keywords (`vibe coding`, `character ai`, `AI girlfriend`, `Claude is goated`)

## To run

`scripts/collect_brightdata.py:28` hard-codes `KEYWORDS_FILE = ROOT / "KEYWORDS.md"`. Two options:

1. **Swap files temporarily** — rename `KEYWORDS.md` → `KEYWORDS_backlash.md`, this file → `KEYWORDS.md`, run, swap back. Brittle.
2. **Parametrize** (preferred) — add a `--keywords` CLI arg, write snapshot manifest to a per-sweep path (`brightdata_snapshots_positive.json`), and write raw outputs to `tiktok/data/raw/tiktok_positive_raw.json` / `youtube/data/raw/youtube_positive_raw.json` so the two corpora don't collide.

I'd recommend option 2 before firing the sweep tomorrow.
