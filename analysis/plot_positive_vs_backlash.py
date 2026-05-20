"""Five-panel summary graphic comparing AI backlash vs AI positive-sentiment corpora.

Panels:
  A: Volume comparison (positive vs backlash, by platform)        — "there's more pro-AI than anti-AI content"
  B: Theme distribution side-by-side, all 12 months               — xrisk and breakthrough_science highlighted as small slices
  C: Engagement by topic, median views                            — backlash + positive on same axes
  D: YouTube topic share over time (backlash) — stacked area      — "art-led -> jobs/deepfake"
  E: 2026-only YouTube theme distribution (positive vs backlash)  — avoids temporal-skew confound

Outputs: analysis/positive_vs_backlash_summary.png (multi-panel)
         analysis/datacenter_callout.png (small standalone)
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(2**31 - 1)

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "analysis"

# ---------- color palettes ----------
BACKLASH_COLOR = "#c0392b"
POSITIVE_COLOR = "#2980b9"

BACKLASH_TOPIC_COLORS = {
    "jobs":         "#d62728",
    "art":          "#9467bd",
    "deepfake":     "#ff7f0e",
    "environment":  "#2ca02c",
    "surveillance": "#1f77b4",
    "regulation":   "#17becf",
    "education":    "#bcbd22",
    "general_neg":  "#7f7f7f",
    "xrisk":        "#000000",
}
POSITIVE_THEME_COLORS = {
    "creative_tool":        "#9467bd",
    "career_productivity":  "#d62728",
    "synthetic_media_play": "#ff7f0e",
    "education_learning":   "#bcbd22",
    "companion_affective":  "#e377c2",
    "model_stan":           "#17becf",
    "general_hype":         "#7f7f7f",
    "breakthrough_science": "#000000",
}

BACKLASH_ORDER = ["art", "deepfake", "jobs", "surveillance", "environment", "general_neg", "xrisk", "education", "regulation"]
POSITIVE_ORDER = ["creative_tool", "career_productivity", "synthetic_media_play", "education_learning",
                  "companion_affective", "general_hype", "model_stan", "breakthrough_science"]


# ---------- data loading ----------
def safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0


def median(xs):
    return sorted(xs)[len(xs) // 2] if xs else 0


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def year_of(r):
    for f in ("date_posted", "date", "published_date", "upload_date"):
        v = r.get(f) or ""
        if v and len(v) >= 4 and v[:4].isdigit():
            return v[:4]
    return None


def load_backlash():
    tt = [r for r in load_csv(REPO / "tiktok/data/tiktok_ai_backlash_final.csv")
          if (r.get("_backlash_final") or "").upper() == "YES"]
    yt = [r for r in load_csv(REPO / "youtube/data/youtube_ai_backlash_final.csv")
          if (r.get("_backlash_final") or "").upper() == "YES"]
    return tt, yt


def load_positive():
    tt = [r for r in load_csv(REPO / "tiktok/data/tiktok_positive_classified.csv")
          if (r.get("_theme") or "") and r.get("_theme") != "unknown"]
    yt = [r for r in load_csv(REPO / "youtube/data/youtube_positive_classified.csv")
          if (r.get("_theme") or "") and r.get("_theme") != "unknown"]
    return tt, yt


# ---------- main figure ----------
def main():
    print("Loading...")
    tt_b, yt_b = load_backlash()
    tt_p, yt_p = load_positive()
    print(f"  backlash: TT={len(tt_b)} YT={len(yt_b)}  positive: TT={len(tt_p)} YT={len(yt_p)}")

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.30)

    # ---- A: VOLUME ----
    ax = fig.add_subplot(gs[0, 0])
    platforms = ["TikTok", "YouTube"]
    backlash_counts = [len(tt_b), len(yt_b)]
    positive_counts = [len(tt_p), len(yt_p)]
    x = np.arange(len(platforms))
    w = 0.35
    ax.bar(x - w/2, backlash_counts, w, label=f"AI backlash (n={sum(backlash_counts):,})",
           color=BACKLASH_COLOR, alpha=0.9)
    ax.bar(x + w/2, positive_counts, w, label=f"AI positive (n={sum(positive_counts):,})",
           color=POSITIVE_COLOR, alpha=0.9)
    for i, (b, p) in enumerate(zip(backlash_counts, positive_counts)):
        ax.text(i - w/2, b + 200, f"{b:,}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + w/2, p + 200, f"{p:,}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(platforms, fontsize=11)
    ax.set_ylabel("Videos classified", fontsize=10)
    ax.set_title("A. Volume — pro-AI content outnumbers anti-AI ~2:1", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    # ---- B: Combined theme distribution ----
    # B1 — backlash topic shares
    ax = fig.add_subplot(gs[0, 1])
    b_counter = Counter((r.get("_topic_primary") or "") for r in tt_b + yt_b)
    total_b = sum(b_counter.values())
    topics = [t for t in BACKLASH_ORDER if b_counter[t] > 0]
    shares = [b_counter[t] / total_b * 100 for t in topics]
    colors = [BACKLASH_TOPIC_COLORS[t] for t in topics]
    bars = ax.barh(topics, shares, color=colors, alpha=0.9)
    ax.invert_yaxis()
    for bar, share, t in zip(bars, shares, topics):
        ax.text(share + 0.4, bar.get_y() + bar.get_height()/2, f"{share:.1f}%", va="center", fontsize=9)
        if t == "xrisk":
            ax.text(share + 4.5, bar.get_y() + bar.get_height()/2,
                    "← existential-risk only 4.4%",
                    va="center", fontsize=9, color="#c0392b", fontweight="bold")
    ax.set_xlabel("Share of backlash videos (%)", fontsize=10)
    ax.set_title(f"B. AI backlash topics (n={total_b:,}, both platforms, 12 months)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(right=max(shares) * 1.45)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)

    # ---- C: Positive theme distribution ----
    ax = fig.add_subplot(gs[1, 0])
    p_counter = Counter((r.get("_theme") or "") for r in tt_p + yt_p)
    total_p = sum(p_counter.values())
    themes = [t for t in POSITIVE_ORDER if p_counter[t] > 0]
    shares = [p_counter[t] / total_p * 100 for t in themes]
    colors = [POSITIVE_THEME_COLORS[t] for t in themes]
    bars = ax.barh(themes, shares, color=colors, alpha=0.9)
    ax.invert_yaxis()
    for bar, share, t in zip(bars, shares, themes):
        ax.text(share + 0.4, bar.get_y() + bar.get_height()/2, f"{share:.1f}%", va="center", fontsize=9)
        if t == "breakthrough_science":
            ax.text(share + 4.5, bar.get_y() + bar.get_height()/2,
                    "← breakthrough-science only 1.9%",
                    va="center", fontsize=9, color="#c0392b", fontweight="bold")
    ax.set_xlabel("Share of positive videos (%)", fontsize=10)
    ax.set_title(f"C. AI positive themes (n={total_p:,}, both platforms, 12 months)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(right=max(shares) * 1.45)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)

    # ---- D: engagement (median views) ----
    ax = fig.add_subplot(gs[1, 1])
    eng_data = []
    for r in yt_b + tt_b:
        topic = r.get("_topic_primary") or ""
        if topic in BACKLASH_TOPIC_COLORS:
            views = safe_int(r.get("views") or r.get("play_count") or r.get("playcount"))
            if views > 0:
                eng_data.append(("backlash", topic, views))
    for r in yt_p + tt_p:
        theme = r.get("_theme") or ""
        if theme in POSITIVE_THEME_COLORS:
            views = safe_int(r.get("views") or r.get("play_count") or r.get("playcount"))
            if views > 0:
                eng_data.append(("positive", theme, views))

    by_combo = defaultdict(list)
    for kind, t, v in eng_data:
        by_combo[(kind, t)].append(v)

    rows = []
    for (kind, t), vs in by_combo.items():
        if len(vs) >= 30:
            rows.append((kind, t, median(vs), len(vs)))
    rows.sort(key=lambda x: -x[2])
    rows = rows[:14]

    labels = [f"{('[neg] ' if k=='backlash' else '[pos] ')}{t}" for k, t, _, _ in rows]
    meds = [r[2] for r in rows]
    colors = [BACKLASH_COLOR if r[0] == "backlash" else POSITIVE_COLOR for r in rows]
    ax.barh(labels, meds, color=colors, alpha=0.85)
    ax.invert_yaxis()
    for i, (k, t, m, n) in enumerate(rows):
        ax.text(m + max(meds)*0.01, i, f"{int(m):,} (n={n})", va="center", fontsize=8)
    ax.set_xlabel("Median views per video", fontsize=10)
    ax.set_title("D. Median engagement by theme (top 14, both platforms pooled)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(right=max(meds) * 1.30)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)

    # ---- E: YouTube backlash topic share over time ----
    ax = fig.add_subplot(gs[2, 0])
    yt_b_by_year = defaultdict(lambda: defaultdict(int))
    for r in yt_b:
        y = year_of(r)
        if y in {"2022", "2023", "2024", "2025", "2026"}:
            t = r.get("_topic_primary") or ""
            if t in BACKLASH_TOPIC_COLORS:
                yt_b_by_year[y][t] += 1

    years = ["2022", "2023", "2024", "2025", "2026"]
    topics_for_stack = ["art", "deepfake", "jobs", "surveillance", "environment", "general_neg", "xrisk", "education", "regulation"]
    arr = np.zeros((len(topics_for_stack), len(years)))
    for j, y in enumerate(years):
        total = sum(yt_b_by_year[y].values()) or 1
        for i, t in enumerate(topics_for_stack):
            arr[i, j] = yt_b_by_year[y][t] / total * 100
    ax.stackplot(years, arr, labels=topics_for_stack,
                 colors=[BACKLASH_TOPIC_COLORS[t] for t in topics_for_stack], alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Upload year", fontsize=10)
    ax.set_ylabel("Share of YT backlash content (%)", fontsize=10)
    ax.set_title("E. YouTube backlash: art-led → jobs+deepfake-led",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)

    # ---- F: YouTube positive topic share over time ----
    ax = fig.add_subplot(gs[2, 1])
    yt_p_by_year = defaultdict(lambda: defaultdict(int))
    for r in yt_p:
        y = year_of(r)
        if y in {"2022", "2023", "2024", "2025", "2026"}:
            t = r.get("_theme") or ""
            if t in POSITIVE_THEME_COLORS:
                yt_p_by_year[y][t] += 1
    themes_for_stack = ["creative_tool", "career_productivity", "synthetic_media_play",
                        "education_learning", "companion_affective", "general_hype",
                        "model_stan", "breakthrough_science"]
    arr = np.zeros((len(themes_for_stack), len(years)))
    for j, y in enumerate(years):
        total = sum(yt_p_by_year[y].values()) or 1
        for i, t in enumerate(themes_for_stack):
            arr[i, j] = yt_p_by_year[y][t] / total * 100
    ax.stackplot(years, arr, labels=themes_for_stack,
                 colors=[POSITIVE_THEME_COLORS[t] for t in themes_for_stack], alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Upload year", fontsize=10)
    ax.set_ylabel("Share of YT positive content (%)", fontsize=10)
    ax.set_title("F. YouTube positive: creative-led → career+productivity-led",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)

    # ---- G: 2026-only YouTube side-by-side ----
    ax = fig.add_subplot(gs[3, :])
    yt_b_26 = [r for r in yt_b if year_of(r) == "2026"]
    yt_p_26 = [r for r in yt_p if year_of(r) == "2026"]
    b_26 = Counter(r.get("_topic_primary","") for r in yt_b_26)
    p_26 = Counter(r.get("_theme","") for r in yt_p_26)

    b_total = sum(b_26.values()) or 1
    p_total = sum(p_26.values()) or 1
    b_pairs = [(t, b_26[t] / b_total * 100) for t in BACKLASH_ORDER if b_26[t] > 0]
    p_pairs = [(t, p_26[t] / p_total * 100) for t in POSITIVE_ORDER if p_26[t] > 0]
    b_pairs.sort(key=lambda x: -x[1])
    p_pairs.sort(key=lambda x: -x[1])

    n = max(len(b_pairs), len(p_pairs))
    ypos = np.arange(n)
    # left half = backlash, right half = positive
    for i, (t, share) in enumerate(b_pairs):
        ax.barh(i, -share, color=BACKLASH_TOPIC_COLORS.get(t, "#888"), alpha=0.9)
        ax.text(-share - 1.5, i, f"{t} {share:.1f}%", va="center", ha="right", fontsize=9)
    for i, (t, share) in enumerate(p_pairs):
        ax.barh(i, share, color=POSITIVE_THEME_COLORS.get(t, "#888"), alpha=0.9)
        ax.text(share + 1.5, i, f"{t} {share:.1f}%", va="center", ha="left", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([-40, -30, -20, -10, 0, 10, 20, 30, 40])
    ax.set_xticklabels(["40%", "30%", "20%", "10%", "0", "10%", "20%", "30%", "40%"])
    ax.text(-25, -1.2, f"BACKLASH (YouTube 2026, n={len(yt_b_26):,})",
            ha="center", fontsize=11, fontweight="bold", color=BACKLASH_COLOR)
    ax.text(25, -1.2, f"POSITIVE (YouTube 2026, n={len(yt_p_26):,})",
            ha="center", fontsize=11, fontweight="bold", color=POSITIVE_COLOR)
    ax.set_title("G. 2026-only YouTube cut (avoids the recency-skew confound)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    plt.suptitle("AI Sentiment on Social Media — Positive vs. Backlash\nTikTok + YouTube, Apr 2025 – Apr 2026",
                 fontsize=15, fontweight="bold", y=0.995)
    out = OUT_DIR / "positive_vs_backlash_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    # ---------- Data center callout ----------
    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 4.5))
    dc_pat = re.compile(r"\b(data ?cent(?:er|re)s?|gpu farm|server farm|data ?campus)\b", re.I)

    def desc_text(r, plat):
        parts = [r.get("description", "") or ""]
        if plat == "yt":
            parts.append(r.get("title", "") or "")
            parts.append(r.get("_transcript_text", "") or "")
        else:
            parts.append(r.get("_transcript", "") or "")
        return " ".join(parts)

    tt_dc = sum(1 for r in tt_b if dc_pat.search(desc_text(r, "tt")))
    yt_dc = sum(1 for r in yt_b if dc_pat.search(desc_text(r, "yt")))
    tt_env = sum(1 for r in tt_b if r.get("_topic_primary") == "environment")
    yt_env = sum(1 for r in yt_b if r.get("_topic_primary") == "environment")

    cats = ["TikTok\n(n=2,416)", "YouTube\n(n=5,937)"]
    env_pct = [tt_env / len(tt_b) * 100, yt_env / len(yt_b) * 100]
    dc_pct = [tt_dc / len(tt_b) * 100, yt_dc / len(yt_b) * 100]
    x = np.arange(len(cats))
    w = 0.35
    ax2.bar(x - w/2, env_pct, w, label="_topic_primary = environment",
            color="#2ca02c", alpha=0.85)
    ax2.bar(x + w/2, dc_pct, w, label="any 'data center / server farm / GPU farm' mention",
            color="#1f4f1f", alpha=0.95)
    for i, (e, d) in enumerate(zip(env_pct, dc_pct)):
        ax2.text(i - w/2, e + 0.2, f"{e:.1f}%", ha="center", fontsize=10)
        ax2.text(i + w/2, d + 0.2, f"{d:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, fontsize=11)
    ax2.set_ylabel("Share of backlash videos (%)", fontsize=10)
    ax2.set_title("Data-center / server-farm mentions in AI backlash content",
                  fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_axisbelow(True)
    plt.tight_layout()
    out2 = OUT_DIR / "datacenter_callout.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
