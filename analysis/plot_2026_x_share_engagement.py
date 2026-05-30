"""2026-only X-shareable summary, ENGAGEMENT-WEIGHTED.

Sister script to plot_2026_x_share.py. Same layout, but each topic's share is
the fraction of total views/plays it captured (not the fraction of videos).
On YouTube engagement = views; on TikTok engagement = play_count.

Output:
  analysis/backlash_topics_2026_engagement.png
  analysis/positive_themes_2026_engagement.png
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0


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


# AI Team is a Malaysian entertainment group; their content matches our scrape
# keyword on "AI" but has nothing to do with artificial intelligence.
AI_TEAM_PAT = re.compile(
    r"(#ai[\s_-]?team\b|ai team x|aiteam2026|aiteam membership|"
    r"partnership ai team|datang.*ai team|aiteam[\s🇮]?‍?)",
    re.I,
)

# Same archetype: a video unrelated to AI that bolts #ai onto its hashtag stack
# to ride the algorithm. The CutieLoop "Only good vibes 2026" pattern is the
# canonical example — 24M views of generic motivation content tagged #ai.
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


# Themes where the classifier over-fires and the secondary is usually more
# accurate. model_stan: any model-name mention got stanned, even tutorials.
# general_hype: catchall for vague positivity, often mislabeled tutorials.
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

    # Cleanup pass on positive rows: drop corpus pollution (AI-Team group +
    # hashtag piggybacks), then promote model_stan / general_hype secondaries
    # to primary theme where present. Do this before the _theme validity
    # check so promotions take effect.
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


def engagement_of(r):
    # TikTok rows have play_count; YouTube rows have views.
    return safe_int(r.get("views") or r.get("play_count"))


def aggregate_engagement(rows, key):
    """Returns {topic: total_views}, {topic: n_videos}, total_engagement."""
    eng_by_topic = defaultdict(int)
    n_by_topic = defaultdict(int)
    total = 0
    for r in rows:
        t = r.get(key) or ""
        if not t or t == "unknown":
            continue
        e = engagement_of(r)
        eng_by_topic[t] += e
        n_by_topic[t] += 1
        total += e
    return eng_by_topic, n_by_topic, total


def horiz_engagement_share(ax, eng, n_videos, total, order, labels, colors,
                            title, xlabel, highlight=None, highlight_note=""):
    items = [(t, eng[t] / total * 100, n_videos[t]) for t in order if eng[t] > 0]
    items.sort(key=lambda x: -x[1])
    names = [labels.get(t, t) for t, _, _ in items]
    shares = [s for _, s, _ in items]
    cmap = [colors[t] for t, _, _ in items]
    bars = ax.barh(names, shares, color=cmap, alpha=0.9)
    ax.invert_yaxis()
    for bar, (t, share, n) in zip(bars, items):
        ax.text(share + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{share:.1f}%  (n={n:,} videos)", va="center", fontsize=10)
        if highlight and t == highlight:
            ax.text(share + 9.0, bar.get_y() + bar.get_height() / 2,
                    highlight_note, va="center", fontsize=10,
                    color=BACKLASH, fontweight="bold")
    ax.set_xlim(right=max(shares) * 1.65)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    print("Loading...")
    tt_b, yt_b, tt_p, yt_p = load()

    b26 = [r for r in tt_b + yt_b if is_2026(r)]
    p26 = [r for r in tt_p + yt_p if is_2026(r)]
    print(f"  2026 videos: backlash n={len(b26):,}  positive n={len(p26):,}")

    b_eng, b_n, b_total = aggregate_engagement(b26, "_topic_primary")
    p_eng, p_n, p_total = aggregate_engagement(p26, "_theme")
    print(f"  2026 engagement (views+plays): backlash={b_total:,}  positive={p_total:,}")

    xrisk_share = b_eng["xrisk"] / b_total * 100 if b_total else 0
    science_share = p_eng["breakthrough_science"] / p_total * 100 if p_total else 0

    # ---------- backlash topics ----------
    fig1, ax1 = plt.subplots(1, 1, figsize=(11, 5))
    horiz_engagement_share(
        ax1, b_eng, b_n, b_total,
        list(BACKLASH_TOPIC_COLORS.keys()),
        BACKLASH_LABELS, BACKLASH_TOPIC_COLORS,
        title="AI backlash topics, 2026 (engagement weighted)",
        xlabel="Share of total backlash engagement (%)",
    )
    plt.tight_layout()
    out1 = OUT / "backlash_topics_2026_engagement.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")

    # ---------- positive themes ----------
    fig2, ax2 = plt.subplots(1, 1, figsize=(11, 5))
    horiz_engagement_share(
        ax2, p_eng, p_n, p_total,
        list(POSITIVE_THEME_COLORS.keys()),
        POSITIVE_LABELS, POSITIVE_THEME_COLORS,
        title="AI positive themes, 2026 (engagement weighted)",
        xlabel="Share of total positive engagement (%)",
    )
    plt.tight_layout()
    out2 = OUT / "positive_themes_2026_engagement.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")

    print()
    print("=== 2026 engagement vs video-count shares ===")
    print(f"{'topic':<25} {'eng_share':>10} {'vid_share':>10} {'n_videos':>10} {'engagement':>15}")
    print("-" * 75)
    print("BACKLASH")
    total_vids_b = sum(b_n.values())
    for t in sorted(b_eng, key=lambda k: -b_eng[k]):
        es = b_eng[t] / b_total * 100
        vs = b_n[t] / total_vids_b * 100
        print(f"  {t:<23} {es:>9.1f}% {vs:>9.1f}% {b_n[t]:>10,} {b_eng[t]:>15,}")
    print("POSITIVE")
    total_vids_p = sum(p_n.values())
    for t in sorted(p_eng, key=lambda k: -p_eng[k]):
        es = p_eng[t] / p_total * 100
        vs = p_n[t] / total_vids_p * 100
        print(f"  {t:<23} {es:>9.1f}% {vs:>9.1f}% {p_n[t]:>10,} {p_eng[t]:>15,}")


if __name__ == "__main__":
    main()
