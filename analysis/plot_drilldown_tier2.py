"""Tier 2 sub-category drill-downs (TikTok, 2026) — uses classify_subtopics output.

Joins the sidecar sub-topic labels back onto the canonical 2026 rows and draws,
for each top-level theme, the distribution of its sub-categories. One figure per
corpus (small-multiples), plus per-theme standalone bars for the narrative-
critical themes so they can drop straight into the Substack piece.

Outputs:
  drilldown_subtopics_backlash_2026.png
  drilldown_subtopics_positive_2026.png
  subtopic_<theme>_2026.png   (standalones for selected themes)
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drilldown_common import load_tiktok, REPO
from classify_subtopics import BACKLASH_SUBTAX, POSITIVE_SUBTAX

OUT = REPO / "analysis"

POSITIVE_LABELS = {
    "career_productivity": "career / productivity", "creative_tool": "creative tools",
    "synthetic_media_play": "synthetic media play", "companion_affective": "AI companion",
    "education_learning": "education / learning", "general_hype": "general hype",
    "model_stan": "model fandom", "breakthrough_science": "breakthrough science",
}
BACKLASH_LABELS = {
    "art": "art / creative theft", "deepfake": "deepfakes / misinfo",
    "jobs": "jobs / displacement", "general_neg": "general anti-AI",
    "environment": "environment / energy", "surveillance": "surveillance / privacy",
    "education": "education", "regulation": "regulation calls", "xrisk": "existential risk",
}

# Pretty labels for sub-codes: humanise underscores; a few hand-tuned.
PRETTY = {
    "noai_activism": "#noAI activism", "all_jobs_panic": "all-jobs panic",
    "ai_slop": "AI slop", "antiai_identity": "#antiAI identity",
    "data_centers": "data centers", "vibe_coding": "vibe coding",
    "chatgpt_workflow": "ChatGPT workflow", "claude_stan": "Claude stan",
    "chatgpt_stan": "ChatGPT stan", "music_suno": "music (Suno)",
}


def pretty(code):
    return PRETTY.get(code, code.replace("_", " "))


def load_subtopics(corpus):
    path = REPO / f"tiktok/data/final/tiktok_{corpus}_subtopics.csv"
    by_id = {}
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            for d in csv.DictReader(f):
                by_id[d["post_id"]] = d["subtopic"]
    return by_id


def small_multiples(rows, theme_key, subtax, labels, title, outname):
    sub_by = load_subtopics("backlash" if theme_key == "_topic_primary" else "positive")
    themes = [t for t, _ in Counter(r[theme_key] for r in rows).most_common()
              if t in subtax]
    dist = defaultdict(Counter)
    for r in rows:
        t = r[theme_key]
        if t in subtax:
            dist[t][sub_by.get(r["post_id"], "other")] += 1

    ncol = 3
    nrow = -(-len(themes) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.2 * nrow))
    axes = axes.flatten()
    for i, t in enumerate(themes):
        ax = axes[i]
        c = dist[t]
        tot = sum(c.values())
        order = [code for code in subtax[t]] + ["other"]
        items = [(code, c[code]) for code in order if c[code] > 0]
        items.sort(key=lambda x: x[1])
        names = [pretty(code) for code, _ in items]
        vals = [n / tot * 100 for _, n in items]
        ax.barh(range(len(names)), vals, color="#4c72b0", alpha=0.9)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        for y, (v, (_, n)) in enumerate(zip(vals, items)):
            ax.text(v + 0.5, y, f"{v:.0f}%", va="center", fontsize=8)
        ax.set_title(f"{labels.get(t, t)}  (n={tot:,})", fontsize=10, fontweight="bold")
        ax.set_xlim(0, max(vals) * 1.25 if vals else 1)
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
    return out, dist


def standalone(rows, theme_key, theme, subtax, label, corpus):
    sub_by = load_subtopics(corpus)
    c = Counter()
    for r in rows:
        if r[theme_key] == theme:
            c[sub_by.get(r["post_id"], "other")] += 1
    tot = sum(c.values())
    order = [code for code in subtax[theme]] + ["other"]
    items = [(code, c[code]) for code in order if c[code] > 0]
    items.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = [pretty(code) for code, _ in items]
    vals = [n / tot * 100 for _, n in items]
    ax.barh(range(len(names)), vals, color="#c0392b" if corpus == "backlash" else "#2ca02c",
            alpha=0.9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    for y, (v, (_, n)) in enumerate(zip(vals, items)):
        ax.text(v + 0.5, y, f"{v:.0f}%  (n={n:,})", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_title(f"Inside '{label}'  ·  TikTok {corpus}, 2026  (n={tot:,})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("share of theme's videos (%)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = OUT / f"subtopic_{theme}_2026.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    b, p = load_tiktok(only_2026=True)
    print(f"2026 TikTok: backlash n={len(b):,} positive n={len(p):,}")

    o1, _ = small_multiples(b, "_topic_primary", BACKLASH_SUBTAX, BACKLASH_LABELS,
                            "AI backlash — sub-categories within each theme  ·  TikTok 2026",
                            "drilldown_subtopics_backlash_2026.png")
    print("wrote", o1.name)
    o2, _ = small_multiples(p, "_theme", POSITIVE_SUBTAX, POSITIVE_LABELS,
                            "AI positive — sub-categories within each theme  ·  TikTok 2026",
                            "drilldown_subtopics_positive_2026.png")
    print("wrote", o2.name)

    # Narrative-critical standalones.
    for theme in ("jobs", "art", "environment"):
        print("wrote", standalone(b, "_topic_primary", theme, BACKLASH_SUBTAX,
                                  BACKLASH_LABELS[theme], "backlash").name)
    for theme in ("career_productivity", "companion_affective"):
        print("wrote", standalone(p, "_theme", theme, POSITIVE_SUBTAX,
                                  POSITIVE_LABELS[theme], "positive").name)


if __name__ == "__main__":
    main()
