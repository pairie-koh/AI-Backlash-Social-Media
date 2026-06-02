"""Tier 1 sub-category drill-downs (TikTok, 2026) — no new classification.

Two deliverables, built only from fields already in the corpus:

  1. positive_register_split_2026.png
     Each positive theme bar split by _register (genuine_sentiment vs
     creator_promo vs creative_showcase vs news_explainer vs brand_official).
     Answers: "how much of each theme is organic vs promotional?"

  2. drilldown_keywords_backlash_2026.png / drilldown_keywords_positive_2026.png
     Small-multiples: one mini bar chart per theme showing its top seed
     search keywords. A PROXY sub-category view (the seed query that surfaced
     the video, not an independent classification) — labelled as such.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drilldown_common import load_tiktok, REPO

OUT = REPO / "analysis"

POSITIVE_LABELS = {
    "career_productivity": "career / productivity",
    "creative_tool": "creative tools",
    "synthetic_media_play": "synthetic media play",
    "companion_affective": "AI companion",
    "education_learning": "education / learning",
    "general_hype": "general hype",
    "model_stan": "model fandom",
    "breakthrough_science": "breakthrough science",
}
BACKLASH_LABELS = {
    "art": "art / creative theft",
    "deepfake": "deepfakes / misinfo",
    "jobs": "jobs / displacement",
    "general_neg": "general anti-AI",
    "environment": "environment / energy",
    "surveillance": "surveillance / privacy",
    "education": "education",
    "regulation": "regulation calls",
    "xrisk": "existential risk",
}

# Register ordered organic -> promotional, with a fixed colour each.
REGISTER_ORDER = [
    ("genuine_sentiment", "genuine sentiment", "#2ca02c"),
    ("creative_showcase", "creative showcase", "#1f77b4"),
    ("news_explainer", "news / explainer", "#9467bd"),
    ("creator_promo", "creator promo", "#ff7f0e"),
    ("brand_official", "brand / official", "#d62728"),
]


def plot_register_split(p):
    themes = [t for t, _ in Counter(r["_theme"] for r in p).most_common()]
    reg_by_theme = defaultdict(Counter)
    for r in p:
        reg_by_theme[r["_theme"]][r.get("_register") or "NA"] += 1

    fig, ax = plt.subplots(figsize=(11, 6))
    names = [f"{POSITIVE_LABELS.get(t, t)}  (n={sum(reg_by_theme[t].values()):,})"
             for t in themes]
    y = range(len(themes))
    left = [0.0] * len(themes)
    for key, lab, color in REGISTER_ORDER:
        vals = []
        for t in themes:
            tot = sum(reg_by_theme[t].values())
            vals.append(reg_by_theme[t][key] / tot * 100 if tot else 0)
        ax.barh(list(y), vals, left=left, color=color, label=lab, alpha=0.9)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 6:
                ax.text(l + v / 2, i, f"{v:.0f}%", va="center", ha="center",
                        fontsize=8, color="white", fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]

    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of theme's videos (%)", fontsize=11)
    ax.set_title("How organic is each positive theme?  TikTok positive content, 2026\n"
                 "register split per theme — green = genuine, orange/red = promotional",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False,
              fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = OUT / "positive_register_split_2026.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_keyword_smallmultiples(rows, key, labels, title, outname, top_n=8):
    themes = [t for t, _ in Counter(r[key] for r in rows).most_common()]
    kw_by = defaultdict(Counter)
    for r in rows:
        kw_by[r[key]][(r.get("_search_keyword") or "NA")] += 1

    ncol = 3
    nrow = -(-len(themes) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow))
    axes = axes.flatten()
    for i, t in enumerate(themes):
        ax = axes[i]
        kws = kw_by[t]
        tot = sum(kws.values())
        top = kws.most_common(top_n)
        other = tot - sum(c for _, c in top)
        items = list(reversed(top))
        knames = [k for k, _ in items]
        kvals = [c for _, c in items]
        if other > 0:
            knames = [f"other ({len(kws) - len(top)} kw)"] + knames
            kvals = [other] + kvals
        ax.barh(range(len(knames)), kvals, color="#4c72b0", alpha=0.9)
        ax.set_yticks(range(len(knames)))
        ax.set_yticklabels(knames, fontsize=8)
        ax.set_title(f"{labels.get(t, t)}  (n={tot:,})", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for j in range(len(themes), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = OUT / outname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    b, p = load_tiktok(only_2026=True)
    print(f"Loaded TikTok 2026: backlash n={len(b):,}  positive n={len(p):,}")

    o1 = plot_register_split(p)
    print("wrote", o1.name)
    o2 = plot_keyword_smallmultiples(
        b, "_topic_primary", BACKLASH_LABELS,
        "AI backlash — top seed keywords per theme (proxy sub-categories)  ·  TikTok 2026",
        "drilldown_keywords_backlash_2026.png")
    print("wrote", o2.name)
    o3 = plot_keyword_smallmultiples(
        p, "_theme", POSITIVE_LABELS,
        "AI positive — top seed keywords per theme (proxy sub-categories)  ·  TikTok 2026",
        "drilldown_keywords_positive_2026.png")
    print("wrote", o3.name)


if __name__ == "__main__":
    main()
