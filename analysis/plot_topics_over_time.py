"""Plot monthly final-YES backlash volume by primary topic for YouTube and TikTok.

Two stacked area panels (YouTube top, TikTok bottom) showing how the
8-category topic mix evolves month-over-month. Reads _topic_primary
from the *_final.csv files (set by classify_topics.py).

Excludes pre-2024 and 2026-04 (partial month).
"""

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

csv.field_size_limit(2**31 - 1)

REPO = Path(__file__).resolve().parent.parent
YCSV = REPO / "youtube" / "data" / "youtube_ai_backlash_final.csv"
TCSV = REPO / "tiktok" / "data" / "tiktok_ai_backlash_final.csv"
OUT = REPO / "analysis" / "topics_over_time.png"

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


def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def month_start(d):
    return d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def aggregate_by_topic(csv_path, date_col):
    by_month_topic = defaultdict(lambda: defaultdict(int))
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            d = parse_iso(r.get(date_col))
            if not d or d.year < 2024 or (d.year == 2026 and d.month >= 4):
                continue
            if (r.get("_backlash_final", "") or "").upper() != "YES":
                continue
            topic = r.get("_topic_primary", "") or ""
            if topic not in TOPICS:
                continue
            by_month_topic[month_start(d)][topic] += 1
    return by_month_topic


def main():
    ydat = aggregate_by_topic(YCSV, "date_posted")
    tdat = aggregate_by_topic(TCSV, "create_time")
    months = sorted(set(ydat.keys()) | set(tdat.keys()))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for ax, dat, plat in [(axes[0], ydat, "YouTube"), (axes[1], tdat, "TikTok")]:
        series = {topic: [dat[m][topic] for m in months] for topic in TOPICS}
        ax.stackplot(months, *[series[t] for t in TOPICS],
                     labels=TOPICS, colors=[TOPIC_COLORS[t] for t in TOPICS], alpha=0.85)
        ax.set_title(f"{plat} — AI-backlash videos per month, by primary topic",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of videos", fontsize=10)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.92, ncol=2)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axes[1].set_xlabel("Month", fontsize=10)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
