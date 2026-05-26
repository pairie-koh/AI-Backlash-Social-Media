"""Streamlit dashboard — AI backlash & positive sentiment on TikTok + YouTube (2026 only).

Run:
    pip install streamlit pandas plotly
    streamlit run analysis/dashboard.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

csv.field_size_limit(2**31 - 1)

REPO = Path(__file__).resolve().parent.parent

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
BACKLASH_COLORS = {
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
POSITIVE_COLORS = {
    "creative_tool":        "#9467bd",
    "career_productivity":  "#d62728",
    "synthetic_media_play": "#ff7f0e",
    "education_learning":   "#bcbd22",
    "companion_affective":  "#e377c2",
    "model_stan":           "#17becf",
    "general_hype":         "#7f7f7f",
    "breakthrough_science": "#000000",
}

DATACENTER_PATTERNS = [
    "data center", "data centre", "datacenter", "datacentre",
    "server farm", "gpu farm", "hyperscale",
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _year_of(date_str: str) -> str | None:
    if not date_str or len(date_str) < 4:
        return None
    head = date_str[:4]
    return head if head.isdigit() else None


def _month_of(date_str: str) -> str | None:
    if not date_str or len(date_str) < 7:
        return None
    return date_str[:7] if date_str[:4].isdigit() else None


@st.cache_data(show_spinner="Loading 2026 data...")
def load_data() -> pd.DataFrame:
    """Load all four CSVs, normalize columns, filter to 2026."""
    frames: list[pd.DataFrame] = []

    # TikTok backlash
    tt_b = _read_csv(REPO / "tiktok/data/tiktok_ai_backlash_final.csv")
    for r in tt_b:
        if (r.get("_backlash_final") or "").upper() != "YES":
            continue
        if _year_of(r.get("create_time", "")) != "2026":
            continue
        frames.append(pd.DataFrame([{
            "platform": "TikTok",
            "sentiment": "Backlash",
            "theme_key": r.get("_topic_primary") or "",
            "date": r.get("create_time", "")[:10],
            "month": _month_of(r.get("create_time", "")),
            "views": int(r.get("play_count") or 0),
            "likes": int(r.get("digg_count") or 0),
            "comments": int(r.get("comment_count") or 0),
            "shares": int(r.get("share_count") or 0),
            "title_desc": (r.get("description") or "")[:300],
            "transcript": r.get("_transcript") or "",
            "url": r.get("url") or "",
            "author": r.get("profile_username") or "",
            "register": "",
        }]))

    # YouTube backlash
    yt_b = _read_csv(REPO / "youtube/data/youtube_ai_backlash_final.csv")
    for r in yt_b:
        if (r.get("_backlash_final") or "").upper() != "YES":
            continue
        if _year_of(r.get("date_posted", "")) != "2026":
            continue
        frames.append(pd.DataFrame([{
            "platform": "YouTube",
            "sentiment": "Backlash",
            "theme_key": r.get("_topic_primary") or "",
            "date": r.get("date_posted", "")[:10],
            "month": _month_of(r.get("date_posted", "")),
            "views": int(r.get("views") or 0),
            "likes": int(r.get("likes") or 0),
            "comments": int(r.get("num_comments") or 0),
            "shares": 0,
            "title_desc": ((r.get("title") or "") + " — " + (r.get("description") or ""))[:300],
            "transcript": r.get("_transcript_text") or "",
            "url": r.get("url") or "",
            "author": r.get("youtuber") or "",
            "register": "",
        }]))

    # TikTok positive
    tt_p = _read_csv(REPO / "tiktok/data/tiktok_positive_classified.csv")
    for r in tt_p:
        theme = r.get("_theme") or ""
        if not theme or theme == "unknown":
            continue
        if _year_of(r.get("create_time", "")) != "2026":
            continue
        frames.append(pd.DataFrame([{
            "platform": "TikTok",
            "sentiment": "Positive",
            "theme_key": theme,
            "date": r.get("create_time", "")[:10],
            "month": _month_of(r.get("create_time", "")),
            "views": int(r.get("play_count") or 0),
            "likes": int(r.get("digg_count") or 0),
            "comments": int(r.get("comment_count") or 0),
            "shares": int(r.get("share_count") or 0),
            "title_desc": (r.get("description") or "")[:300],
            "transcript": r.get("_transcript") or "",
            "url": r.get("url") or "",
            "author": r.get("profile_username") or "",
            "register": r.get("_register") or "",
        }]))

    # YouTube positive
    yt_p = _read_csv(REPO / "youtube/data/youtube_positive_classified.csv")
    for r in yt_p:
        theme = r.get("_theme") or ""
        if not theme or theme == "unknown":
            continue
        if _year_of(r.get("date_posted", "")) != "2026":
            continue
        frames.append(pd.DataFrame([{
            "platform": "YouTube",
            "sentiment": "Positive",
            "theme_key": theme,
            "date": r.get("date_posted", "")[:10],
            "month": _month_of(r.get("date_posted", "")),
            "views": int(r.get("views") or 0),
            "likes": int(r.get("likes") or 0),
            "comments": int(r.get("num_comments") or 0),
            "shares": 0,
            "title_desc": ((r.get("title") or "") + " — " + (r.get("description") or ""))[:300],
            "transcript": r.get("_transcript_text") or "",
            "url": r.get("url") or "",
            "author": r.get("youtuber") or "",
            "register": r.get("_register") or "",
        }]))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Pretty theme label
    def _label(row):
        if row["sentiment"] == "Backlash":
            return BACKLASH_LABELS.get(row["theme_key"], row["theme_key"])
        return POSITIVE_LABELS.get(row["theme_key"], row["theme_key"])

    df["theme"] = df.apply(_label, axis=1)

    # Data-center mention flag (substring search over title/desc/transcript)
    def _is_dc(row):
        blob = (row["title_desc"] + " " + row["transcript"]).lower()
        return any(p in blob for p in DATACENTER_PATTERNS)

    df["mentions_datacenter"] = df.apply(_is_dc, axis=1)
    return df


# ---------- UI ----------

st.set_page_config(
    page_title="AI Sentiment Dashboard — 2026",
    page_icon=None,
    layout="wide",
)

st.title("AI Sentiment on TikTok + YouTube — 2026")
st.caption(
    "Videos classified by Claude Sonnet for sentiment and theme. "
    "Restricted to videos posted in 2026."
)

df = load_data()

if df.empty:
    st.error("No data loaded. Check that the CSV files exist under tiktok/data and youtube/data.")
    st.stop()

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")

    platforms = st.multiselect(
        "Platform",
        options=["TikTok", "YouTube"],
        default=["TikTok", "YouTube"],
    )
    sentiments = st.multiselect(
        "Sentiment",
        options=["Backlash", "Positive"],
        default=["Backlash", "Positive"],
    )

    months_available = sorted([m for m in df["month"].dropna().unique() if m])
    if months_available:
        m_start, m_end = st.select_slider(
            "Month range (2026)",
            options=months_available,
            value=(months_available[0], months_available[-1]),
        )
    else:
        m_start, m_end = None, None

    datacenter_only = st.checkbox("Only videos mentioning data centers", value=False)

    # Register filter (positive only — promotional vs organic)
    registers = sorted([r for r in df["register"].unique() if r])
    if registers:
        sel_registers = st.multiselect(
            "Positive register (promotional / organic / etc.)",
            options=registers,
            default=registers,
            help="Applies to positive-sentiment videos only.",
        )
    else:
        sel_registers = []

    st.markdown("---")
    st.caption(f"Total videos in 2026 corpus: **{len(df):,}**")

# Apply filters
fdf = df[df["platform"].isin(platforms) & df["sentiment"].isin(sentiments)].copy()
if m_start and m_end:
    fdf = fdf[(fdf["month"] >= m_start) & (fdf["month"] <= m_end)]
if datacenter_only:
    fdf = fdf[fdf["mentions_datacenter"]]
if sel_registers:
    # only filter positive rows on register; leave backlash untouched
    mask_pos = fdf["sentiment"] == "Positive"
    fdf = pd.concat([
        fdf[~mask_pos],
        fdf[mask_pos & fdf["register"].isin(sel_registers)],
    ])

# ---------- KPI row ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Videos (filtered)", f"{len(fdf):,}")
n_back = int((fdf["sentiment"] == "Backlash").sum())
n_pos = int((fdf["sentiment"] == "Positive").sum())
c2.metric("Backlash", f"{n_back:,}")
c3.metric("Positive", f"{n_pos:,}")
c4.metric("Mention data centers", f"{int(fdf['mentions_datacenter'].sum()):,}")

if fdf.empty:
    st.warning("No videos match the current filters.")
    st.stop()

# ---------- Tabs ----------
tab_share, tab_time, tab_eng, tab_examples = st.tabs(
    ["Theme share", "Over time", "Engagement", "Example videos"]
)

# ---- Theme share ----
with tab_share:
    st.subheader("Theme share by platform")
    cols = st.columns(len(sentiments) if sentiments else 1)
    for col, sent in zip(cols, sentiments):
        sub = fdf[fdf["sentiment"] == sent]
        if sub.empty:
            col.info(f"No {sent.lower()} videos in current filters.")
            continue
        agg = (
            sub.groupby(["platform", "theme", "theme_key"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
        )
        totals = agg.groupby("platform")["n"].transform("sum")
        agg["share"] = agg["n"] / totals * 100

        color_map = (BACKLASH_COLORS if sent == "Backlash" else POSITIVE_COLORS)
        label_map = (BACKLASH_LABELS if sent == "Backlash" else POSITIVE_LABELS)
        ordered_keys = list(color_map.keys())
        agg["theme_key"] = pd.Categorical(agg["theme_key"], categories=ordered_keys, ordered=True)
        agg = agg.sort_values(["platform", "share"], ascending=[True, False])

        fig = px.bar(
            agg,
            x="share",
            y="theme",
            color="platform",
            orientation="h",
            barmode="group",
            text=agg["share"].map(lambda x: f"{x:.1f}%"),
            title=f"{sent} — share by platform (n={len(sub):,})",
            labels={"share": "% of videos", "theme": ""},
        )
        fig.update_layout(height=480, yaxis={"categoryorder": "total ascending"})
        col.plotly_chart(fig, use_container_width=True)

# ---- Time series ----
with tab_time:
    st.subheader("Monthly volume by theme")
    sent_for_time = st.selectbox(
        "Sentiment for time series",
        options=sentiments or ["Backlash", "Positive"],
        index=0,
    )
    sub = fdf[fdf["sentiment"] == sent_for_time]
    if sub.empty:
        st.info("No videos in current filters.")
    else:
        ts = (
            sub.groupby(["month", "theme", "theme_key"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
            .sort_values("month")
        )
        color_map = BACKLASH_COLORS if sent_for_time == "Backlash" else POSITIVE_COLORS
        label_map = BACKLASH_LABELS if sent_for_time == "Backlash" else POSITIVE_LABELS
        ts["theme"] = ts["theme_key"].map(label_map).fillna(ts["theme_key"])
        color_discrete_map = {label_map.get(k, k): v for k, v in color_map.items()}

        fig = px.area(
            ts,
            x="month",
            y="n",
            color="theme",
            color_discrete_map=color_discrete_map,
            title=f"{sent_for_time} videos per month, 2026 (n={len(sub):,})",
            labels={"month": "", "n": "Videos"},
        )
        fig.update_layout(height=520, legend_title="")
        st.plotly_chart(fig, use_container_width=True)

# ---- Engagement ----
with tab_eng:
    st.subheader("Engagement by theme")
    st.caption(
        "Engagement is reported as the median across videos in each theme "
        "(median is more robust to viral outliers than mean)."
    )
    metric = st.radio(
        "Metric", options=["views", "likes", "comments"], horizontal=True
    )
    sub = fdf.copy()
    if sub.empty:
        st.info("No videos in current filters.")
    else:
        agg = (
            sub.groupby(["sentiment", "platform", "theme", "theme_key"], as_index=False)
            .agg(median_metric=(metric, "median"), n=(metric, "size"))
        )
        # combine sentiment+platform for x-axis grouping
        agg["group"] = agg["platform"] + " — " + agg["sentiment"]
        fig = px.bar(
            agg,
            x="theme",
            y="median_metric",
            color="group",
            barmode="group",
            title=f"Median {metric} per video, by theme",
            labels={"median_metric": f"Median {metric}", "theme": ""},
            hover_data={"n": True},
        )
        fig.update_layout(height=520, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

# ---- Examples ----
with tab_examples:
    st.subheader("Example videos")
    sort_by = st.selectbox(
        "Sort by", options=["views", "likes", "comments", "date"], index=0
    )
    show_n = st.slider("How many to show", min_value=10, max_value=200, value=50, step=10)

    cols_to_show = ["sentiment", "platform", "theme", "date", "views", "likes",
                    "comments", "author", "title_desc", "url"]
    table = fdf[cols_to_show].copy()
    if sort_by == "date":
        table = table.sort_values("date", ascending=False)
    else:
        table = table.sort_values(sort_by, ascending=False)
    table = table.head(show_n)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("url"),
            "title_desc": st.column_config.TextColumn("title / description", width="large"),
            "views": st.column_config.NumberColumn("views", format="%d"),
            "likes": st.column_config.NumberColumn("likes", format="%d"),
            "comments": st.column_config.NumberColumn("comments", format="%d"),
        },
    )

    # Download
    csv_bytes = fdf[cols_to_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered set as CSV",
        data=csv_bytes,
        file_name="ai_sentiment_2026_filtered.csv",
        mime="text/csv",
    )
