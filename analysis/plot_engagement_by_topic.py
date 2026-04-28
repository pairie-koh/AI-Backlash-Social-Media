"""Plot engagement metrics by primary topic for YouTube and TikTok.

Four-panel grid:
  - Top-left:    YouTube total reach by topic (sum of views)
  - Top-right:   YouTube median views per video by topic
  - Bottom-left: TikTok total reach by topic (sum of plays)
  - Bottom-right:TikTok median plays per video by topic

Total-reach bars are sorted by reach. Median bars are sorted by median.
Reads _topic_primary from the *_final.csv files.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv.field_size_limit(2**31 - 1)

REPO = Path(__file__).resolve().parent.parent
YCSV = REPO / "youtube" / "data" / "youtube_ai_backlash_final.csv"
TCSV = REPO / "tiktok" / "data" / "tiktok_ai_backlash_final.csv"
OUT = REPO / "analysis" / "engagement_by_topic.png"

TOPICS = ["jobs", "art", "deepfake", "environment", "surveillance", "regulation", "education", "general_neg"]
TOPIC_COLORS = {
    "jobs":         "#d62728",
    "art":          "#9467bd",
    "deepfake":     "#ff7f0e",
    "environment":  "#2ca02c",
    "surveillance": "#1f77b4",
    "regulation":   "#17becf",
    "education":    "#bcbd22",
    "general_neg":  "#7f7f7f",
}


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0


def median(xs):
    return sorted(xs)[len(xs) // 2] if xs else 0


def aggregate(csv_path, view_col):
    by_topic = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("_backlash_final", "") or "").upper() != "YES":
                continue
            topic = r.get("_topic_primary", "") or ""
            if topic not in TOPICS:
                continue
            by_topic[topic].append(safe_int(r.get(view_col)))
    return by_topic


def main():
    ydat = aggregate(YCSV, "views")
    tdat = aggregate(TCSV, "play_count")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for row_i, (plat, dat) in enumerate([("YouTube", ydat), ("TikTok", tdat)]):
        totals = [(t, sum(dat[t]), len(dat[t]), median(dat[t])) for t in TOPICS]
        totals.sort(key=lambda x: -x[1])
        topics_sorted = [x[0] for x in totals]
        reach_sorted = [x[1] for x in totals]
        n_sorted = [x[2] for x in totals]
        med_sorted = [x[3] for x in totals]
        colors = [TOPIC_COLORS[t] for t in topics_sorted]

        # Left panel: total reach
        ax = axes[row_i, 0]
        bars = ax.barh(topics_sorted, [r / 1e6 for r in reach_sorted], color=colors, alpha=0.9)
        ax.invert_yaxis()
        for bar, n, total in zip(bars, n_sorted, reach_sorted):
            ax.text(bar.get_width() + max(reach_sorted) / 1e6 * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{total/1e6:.0f}M ({n} videos)", va="center", fontsize=9)
        ax.set_xlabel("Total views (millions)", fontsize=10)
        ax.set_title(f"{plat} — total reach by topic", fontsize=11, fontweight="bold")
        ax.set_xlim(right=max(reach_sorted) / 1e6 * 1.25)
        ax.grid(True, alpha=0.3, axis="x")

        # Right panel: median views, sorted by median
        medsort = sorted(zip(topics_sorted, med_sorted, n_sorted), key=lambda x: -x[1])
        topics_m = [x[0] for x in medsort]
        meds = [x[1] for x in medsort]
        ns = [x[2] for x in medsort]
        colors_m = [TOPIC_COLORS[t] for t in topics_m]

        ax = axes[row_i, 1]
        bars = ax.barh(topics_m, meds, color=colors_m, alpha=0.9)
        ax.invert_yaxis()
        for bar, n, m in zip(bars, ns, meds):
            ax.text(bar.get_width() + max(meds) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{m:,} (n={n})", va="center", fontsize=9)
        ax.set_xlabel("Median views per video", fontsize=10)
        ax.set_title(f"{plat} — median views per video by topic", fontsize=11, fontweight="bold")
        ax.set_xlim(right=max(meds) * 1.30)
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("AI Backlash Engagement by Topic", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
