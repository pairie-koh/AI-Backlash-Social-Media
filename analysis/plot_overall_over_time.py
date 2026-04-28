"""Plot monthly count of final-YES backlash videos for YouTube and TikTok.

Reads *_ai_backlash_final.csv on each platform, aggregates final-YES videos
by month, and writes a single line chart to analysis/overall_over_time.png.

Excludes pre-2024 (sparse) and 2026-04 (partial month).
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
OUT = REPO / "analysis" / "overall_over_time.png"


def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def month_start(d):
    return d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def aggregate_monthly(csv_path, date_col):
    counts = defaultdict(int)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            d = parse_iso(r.get(date_col))
            if not d or d.year < 2024 or (d.year == 2026 and d.month >= 4):
                continue
            if (r.get("_backlash_final", "") or "").upper() == "YES":
                counts[month_start(d)] += 1
    return counts


def main():
    ym = aggregate_monthly(YCSV, "date_posted")
    tm = aggregate_monthly(TCSV, "create_time")
    months = sorted(set(ym.keys()) | set(tm.keys()))
    y_yes = [ym[m] for m in months]
    t_yes = [tm[m] for m in months]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(months, y_yes, marker="o", linewidth=2.2, markersize=5, label="YouTube", color="#cc2c2c")
    ax.plot(months, t_yes, marker="s", linewidth=2.2, markersize=5, label="TikTok", color="#2a72c0")
    ax.fill_between(months, y_yes, alpha=0.15, color="#cc2c2c")
    ax.fill_between(months, t_yes, alpha=0.15, color="#2a72c0")

    ax.set_ylabel("Number of AI-backlash videos posted", fontsize=11)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_title("AI Backlash Videos Over Time — YouTube and TikTok",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
