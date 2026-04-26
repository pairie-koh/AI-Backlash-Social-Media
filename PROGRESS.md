# Collection Progress

**Last updated:** 2026-04-26 (collection complete)

## Status: Bright Data collection COMPLETE

The full keyword × platform sweep finished at 23:31 on 2026-04-26.
See `scripts/collect_run.log` for the run trace.

## What was collected

- **68 keywords** (from `KEYWORDS.md`) × **2 platforms** = **136 newly-fired snapshots**
- **+ 3 seed snapshots** carried over from earlier aborted runs
- **= 139 snapshots total** in `scripts/brightdata_snapshots.json` (70 YouTube / 69 TikTok)
- **`limit_per_input`: NONE** — every keyword scraped until Bright Data discovery exhausted (no per-keyword cap)
- Dataset IDs: TikTok `gd_lu702nij2f790tmv9h`, YouTube `gd_lk56epmy2i5g7lzu0k`
- API token: read from `BRIGHTDATA_API_TOKEN` env var (script exits if unset)

## Final yield (on disk)

| Platform | Records | File | Avg/keyword |
|----------|---------|------|-------------|
| TikTok   | **8,108**  | `tiktok/data/raw/tiktok_raw.json` (40 MB)   | ~119 |
| YouTube  | **23,865** | `youtube/data/raw/youtube_raw.json` (1.79 GB) | ~351 |
| **Total** | **31,973** | | |

YouTube ran ~3× richer than TikTok per keyword — TikTok keyword discovery is shallower on niche / anti-AI tags, so the no-cap setting still produced fewer records than the 200-cap floor would have suggested. The original README projection of 50–80k assumed `limit_per_input=200`; that cap was dropped.

Estimated Bright Data spend: **~$48** at $0.0015/record.

## Next steps

The full pipeline is the LLM passes on top of the raw JSONs. All filter/classify scripts have resume support (`*_filter_llm_progress.json` etc.), so they're safe to interrupt and re-run.

Default LLM is `anthropic/claude-sonnet-4` via OpenRouter (override with `LLM_MODEL`).

```bash
cd C:/Users/hello/AI-Backlash-Social-Media
export OPENROUTER_API_KEY=...   # or OPENAI_API_KEY / ANTHROPIC_API_KEY (fallback chain)

# 1. Filter for AI-backlash relevance (YES/NO per video)
python youtube/scripts/filter_by_topic.py
python tiktok/scripts/filter_by_topic.py

# 2. TikTok-only: classify backlash theme, extract details, transcribe
python tiktok/scripts/classify_backlash_theme.py
python tiktok/scripts/extract_details.py
python tiktok/scripts/whisper_transcribe.py
```

## If you ever need to re-collect

`python scripts/collect_brightdata.py resume` is idempotent — it loads the manifest and only polls existing snapshot IDs (no re-triggering, no double-charge). Bright Data snapshots stay available for ~30 days after triggering.

To fire a *new* sweep with different keywords, edit `KEYWORDS.md` and run `python scripts/collect_brightdata.py` (no `resume` arg).

## Notes / gotchas

- `_search_keyword` field is appended to every record so you can trace which keyword surfaced each video.
- The 3 seed snapshots are duplicates of `AI taking jobs` / `AI replacing workers` — dedup by `post_id` (TikTok) / `video_id` (YouTube) handles this automatically.
- LLM filter resume files (`youtube_filter_llm_progress.json`, `tiktok_filter_llm_progress.json`) will be created on first filter run.
