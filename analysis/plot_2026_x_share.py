"""2026-only X-shareable summary — two standalone graphs.

Restricts both corpora to videos posted in 2026 to sidestep the recency-skew
issue in the scrape.

Output:
  analysis/backlash_topics_2026.png   — backlash topic mix, both platforms pooled
  analysis/positive_themes_2026.png   — positive theme mix, both platforms pooled
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(2**31 - 1)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "analysis"

BACKLASH = "#c0392b"
POSITIVE = "#2980b9"

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

BACKLASH_LABELS = {
    "art": "art / creative theft",
    "deepfake": "deepfakes / misinfo",
    "jobs": "jobs / displacement",
    "surveillance": "surveillance / privacy",
    "environment": "environment / energy",
    "general_neg": "general anti-AI",
    "xrisk": "existential risk",
    "education": "education",
    "regulation": "regulation calls",
}
POSITIVE_LABELS = {
    "creative_tool": "creative tools",
    "career_productivity": "career / productivity",
    "synthetic_media_play": "synthetic media play",
    "education_learning": "education / learning",
    "companion_affective": "AI companion",
    "model_stan": "model fandom",
    "general_hype": "general hype",
    "breakthrough_science": "breakthrough science",
}

def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def year_of(r):
    for f in ("date_posted", "create_time", "date", "published_date", "upload_date"):
        v = r.get(f) or ""
        if v and len(v) >= 4 and v[:4].isdigit():
            return v[:4]
    return None


def is_2026(r):
    return year_of(r) == "2026"


def load():
    tt_b = [r for r in load_csv(REPO / "tiktok/data/tiktok_ai_backlash_final.csv")
            if (r.get("_backlash_final") or "").upper() == "YES"]
    yt_b = [r for r in load_csv(REPO / "youtube/data/youtube_ai_backlash_final.csv")
            if (r.get("_backlash_final") or "").upper() == "YES"]
    tt_p = [r for r in load_csv(REPO / "tiktok/data/tiktok_positive_classified.csv")
            if (r.get("_theme") or "") and r.get("_theme") != "unknown"]
    yt_p = [r for r in load_csv(REPO / "youtube/data/youtube_positive_classified.csv")
            if (r.get("_theme") or "") and r.get("_theme") != "unknown"]
    return tt_b, yt_b, tt_p, yt_p


def horiz_share(ax, counter, total, order, labels, colors, title, highlight=None, highlight_note=""):
    items = [(t, counter[t] / total * 100) for t in order if counter[t] > 0]
    items.sort(key=lambda x: -x[1])
    names = [labels.get(t, t) for t, _ in items]
    shares = [s for _, s in items]
    cmap = [colors[t] for t, _ in items]
    bars = ax.barh(names, shares, color=cmap, alpha=0.9)
    ax.invert_yaxis()
    for bar, share, (t, _) in zip(bars, shares, items):
        ax.text(share + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{share:.1f}%", va="center", fontsize=10)
        if highlight and t == highlight:
            ax.text(share + 4.5, bar.get_y() + bar.get_height() / 2,
                    highlight_note, va="center", fontsize=10,
                    color=BACKLASH, fontweight="bold")
    ax.set_xlim(right=max(shares) * 1.50)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    print("Loading...")
    tt_b, yt_b, tt_p, yt_p = load()

    b26 = [r for r in tt_b + yt_b if is_2026(r)]
    p26 = [r for r in tt_p + yt_p if is_2026(r)]
    print(f"  2026 only: backlash n={len(b26):,}  positive n={len(p26):,}")

    b_counter = Counter((r.get("_topic_primary") or "") for r in b26)
    p_counter = Counter((r.get("_theme") or "") for r in p26)
    xrisk_share = b_counter["xrisk"] / sum(b_counter.values()) * 100
    science_share = p_counter["breakthrough_science"] / sum(p_counter.values()) * 100

    # ---------- backlash topics ----------
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    horiz_share(
        ax1, b_counter, sum(b_counter.values()),
        list(BACKLASH_TOPIC_COLORS.keys()),
        BACKLASH_LABELS, BACKLASH_TOPIC_COLORS,
        f"AI backlash topics, 2026 (TikTok + YouTube, n={sum(b_counter.values()):,})",
        highlight="xrisk",
        highlight_note=f"← existential risk only {xrisk_share:.1f}%",
    )
    ax1.set_xlabel("Share of backlash videos (%)", fontsize=11)
    plt.tight_layout()
    out1 = OUT / "backlash_topics_2026.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")

    # ---------- positive themes ----------
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    horiz_share(
        ax2, p_counter, sum(p_counter.values()),
        list(POSITIVE_THEME_COLORS.keys()),
        POSITIVE_LABELS, POSITIVE_THEME_COLORS,
        f"AI positive themes, 2026 (TikTok + YouTube, n={sum(p_counter.values()):,})",
        highlight="breakthrough_science",
        highlight_note=f"← breakthrough science only {science_share:.1f}%",
    )
    ax2.set_xlabel("Share of positive videos (%)", fontsize=11)
    plt.tight_layout()
    out2 = OUT / "positive_themes_2026.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")

    print()
    print("=== 2026 numbers ===")
    print(f"  backlash total : {sum(b_counter.values()):,}")
    print(f"  positive total : {sum(p_counter.values()):,}")
    print(f"  xrisk share    : {xrisk_share:.1f}%  ({b_counter['xrisk']:,} videos)")
    print(f"  science share  : {science_share:.1f}%  ({p_counter['breakthrough_science']:,} videos)")


if __name__ == "__main__":
    main()
