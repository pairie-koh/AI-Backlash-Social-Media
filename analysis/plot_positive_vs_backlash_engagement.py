"""Engagement-weighted positive vs. backlash comparison, full 12-month window.

Sister to Panel A of plot_positive_vs_backlash.py (which is video-count weighted).
Here each bar is total engagement (YouTube views + TikTok plays) for the corpus.

Output: analysis/positive_vs_backlash_engagement.png
"""

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(2**31 - 1)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "analysis"

BACKLASH_COLOR = "#c0392b"
POSITIVE_COLOR = "#2980b9"


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


AI_TEAM_PAT = re.compile(
    r"(#ai[\s_-]?team\b|ai team x|aiteam2026|aiteam membership|"
    r"partnership ai team|datang.*ai team|aiteam[\s🇮]?‍?)",
    re.I,
)
HASHTAG_PIGGYBACK_PATS = [
    re.compile(r"only good vibes.*2026.*#ai", re.I),
]


def _row_blob(r):
    parts = [r.get("description", "") or "", r.get("hashtags", "") or "",
             r.get("title", "") or ""]
    return " ".join(parts).lower()


def _is_corpus_pollution(r):
    blob = _row_blob(r)
    if AI_TEAM_PAT.search(blob):
        return True
    for pat in HASHTAG_PIGGYBACK_PATS:
        if pat.search(blob):
            return True
    return False


SECONDARY_PROMOTE_THEMES = {"model_stan", "general_hype"}


def _promote_secondary(r):
    if (r.get("_theme") or "") not in SECONDARY_PROMOTE_THEMES:
        return
    raw = (r.get("_secondary_themes") or "").strip()
    if not raw or raw == "[]":
        return
    try:
        arr = json.loads(raw)
    except Exception:
        return
    if arr and arr[0]:
        r["_theme"] = arr[0]


def load():
    tt_b = [r for r in load_csv(REPO / "tiktok/data/final/tiktok_backlash_final.csv")
            if (r.get("_backlash_final") or "").upper() == "YES"]
    yt_b = [r for r in load_csv(REPO / "youtube/data/final/youtube_backlash_final.csv")
            if (r.get("_backlash_final") or "").upper() == "YES"]

    tt_p_raw = load_csv(REPO / "tiktok/data/final/tiktok_positive_final.csv")
    yt_p_raw = load_csv(REPO / "youtube/data/final/youtube_positive_final.csv")

    n_dropped = 0
    tt_p = []
    for r in tt_p_raw:
        if _is_corpus_pollution(r):
            n_dropped += 1
            continue
        _promote_secondary(r)
        if (r.get("_theme") or "") and r.get("_theme") != "unknown":
            tt_p.append(r)
    yt_p = []
    for r in yt_p_raw:
        if _is_corpus_pollution(r):
            n_dropped += 1
            continue
        _promote_secondary(r)
        if (r.get("_theme") or "") and r.get("_theme") != "unknown":
            yt_p.append(r)
    print(f"  cleanup: dropped {n_dropped} corpus-pollution rows from positive corpus")

    return tt_b, yt_b, tt_p, yt_p


def total_engagement(rows):
    return sum(safe_int(r.get("views") or r.get("play_count")) for r in rows)


def fmt_billions(x):
    if x >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    return f"{x:,}"


def main():
    print("Loading...")
    tt_b, yt_b, tt_p, yt_p = load()
    print(f"  videos: TT_b={len(tt_b):,}  YT_b={len(yt_b):,}  "
          f"TT_p={len(tt_p):,}  YT_p={len(yt_p):,}")

    eng = {
        ("TikTok",  "backlash"): total_engagement(tt_b),
        ("TikTok",  "positive"): total_engagement(tt_p),
        ("YouTube", "backlash"): total_engagement(yt_b),
        ("YouTube", "positive"): total_engagement(yt_p),
    }
    for k, v in eng.items():
        print(f"  {k[0]:<8} {k[1]:<8} engagement = {v:>16,}")

    backlash_total = eng[("TikTok", "backlash")] + eng[("YouTube", "backlash")]
    positive_total = eng[("TikTok", "positive")] + eng[("YouTube", "positive")]
    print(f"  TOTAL backlash engagement = {backlash_total:,}")
    print(f"  TOTAL positive engagement = {positive_total:,}")
    print(f"  positive : backlash ratio = {positive_total / backlash_total:.2f}x")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    platforms = ["TikTok", "YouTube"]
    backlash_vals = [eng[(p, "backlash")] for p in platforms]
    positive_vals = [eng[(p, "positive")] for p in platforms]

    x = np.arange(len(platforms))
    w = 0.35
    ax.bar(x - w/2, backlash_vals, w,
           label=f"AI backlash (total = {fmt_billions(backlash_total)})",
           color=BACKLASH_COLOR, alpha=0.9)
    ax.bar(x + w/2, positive_vals, w,
           label=f"AI positive (total = {fmt_billions(positive_total)})",
           color=POSITIVE_COLOR, alpha=0.9)

    ymax = max(backlash_vals + positive_vals)
    for i, (b, p) in enumerate(zip(backlash_vals, positive_vals)):
        ax.text(i - w/2, b + ymax * 0.01, fmt_billions(b),
                ha="center", fontsize=11, fontweight="bold")
        ax.text(i + w/2, p + ymax * 0.01, fmt_billions(p),
                ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(platforms, fontsize=12)
    ax.set_ylabel("Total engagement (YouTube views + TikTok plays)", fontsize=11)
    ax.set_title(
        f"AI positive vs. backlash — engagement-weighted\n"
        f"TikTok + YouTube, Apr 2025 – Apr 2026  "
        f"(positive captures {positive_total / (positive_total + backlash_total) * 100:.1f}% "
        f"of total AI-related engagement)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    ax.set_ylim(top=ymax * 1.15)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = OUT / "positive_vs_backlash_engagement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
