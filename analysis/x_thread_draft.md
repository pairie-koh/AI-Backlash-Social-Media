# X thread draft — AI sentiment on TikTok + YouTube (2026 cut)

Numbers come from `plot_2026_x_share.py`. 2026-only to sidestep the recency-skew in the Bright Data scrape (the share of pro/anti content is stable across years; the volume isn't).

---

**Tweet 1 — hook + main composite**
We scraped ~50K AI-related TikTok + YouTube videos and classified them with Claude. Restricted to 2026 posts, two surprises:

1) Pro-AI content outnumbers anti-AI ~2:1
2) Almost none of the backlash is about existential risk

[attach: ai_sentiment_2026_x.png]

---

**Tweet 2 — what backlash *is* actually about**
Of 3,668 backlash videos in 2026, the top complaints are:

• jobs / displacement — 20%
• deepfakes / misinfo — 19%
• art / creative theft — 18%
• environment / energy — 12%
• general anti-AI — 10%

Existential risk: 4.3% (158 videos). The discourse that dominates AI Twitter is barely visible on the platforms where most people actually are.

---

**Tweet 3 — data centers**
On YouTube, **23% of all backlash videos in 2026 explicitly mention a data center, server farm, or GPU farm** — twice the share that even get classified as "environment" content (11%).

Data centers are leaking into jobs, general anti-AI, and politics content. They're a cross-cutting hook.

On TikTok: 8.9% — much lower, because TikTok backlash is artist-led (art theft is the dominant theme).

[attach: datacenter_2026.png]

---

**Tweet 4 — positive side**
And on the pro-AI side, 7,099 videos in 2026:

• career / productivity — 34%
• creative tools — 29%
• synthetic-media play — 13%
• education — 7%
• AI companion — 5.5%

Breakthrough-science / "AI cures cancer" content: just 2.1%. The labs' favorite story arc is also barely on the platforms.

[attach: positive_themes_2026.png]

---

**Tweet 5 — caveats / what's next**
Caveats:
- Bright Data over-represents recent posts, which is why this cut is 2026-only.
- Positive corpus skews promotional (especially YouTube). Restricting to genuine-sentiment subset would shift career/productivity down.
- English keywords → anglophone-dominant sample.

Working on a full write-up. Code + numbers are in the repo.

---

## Numbers cheat-sheet (for replies)

| | TikTok | YouTube | Pooled |
|---|---|---|---|
| Backlash videos, 2026 | 1,263 | 2,405 | 3,668 |
| Positive videos, 2026 | 2,393 | 4,706 | 7,099 |
| Pos:neg ratio | 1.9x | 2.0x | 1.9x |
| Existential risk share of backlash | — | — | 4.3% |
| Data-center mention share of backlash | 8.9% | 23.3% | — |
| Environment-topic share of backlash | 12.7% | 11.0% | — |
| Breakthrough-science share of positive | — | — | 2.1% |
