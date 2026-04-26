"""Quick check: do YouTube raw records have transcripts populated?"""
import json
from pathlib import Path

PATH = Path("C:/Users/hello/AI-Backlash-Social-Media/youtube/data/raw/youtube_raw.json")

with open(PATH, "rb") as f:
    chunk = f.read(80_000_000)

text = chunk.decode("utf-8", errors="replace")
records = []
i = text.find("{")
while i > 0 and len(records) < 200:
    depth = 0
    in_str = False
    esc = False
    end = i
    for j in range(i, len(text)):
        ch = text[j]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    try:
        rec = json.loads(text[i:end])
        records.append(rec)
    except Exception:
        pass
    nxt = text.find("{", end)
    if nxt < 0:
        break
    i = nxt

print(f"sampled records: {len(records)}")
has_ft = sum(1 for r in records if r.get("formatted_transcript"))
has_t = sum(1 for r in records if r.get("transcript") and len(str(r.get("transcript", ""))) > 10)
has_either = sum(
    1
    for r in records
    if r.get("formatted_transcript")
    or (r.get("transcript") and len(str(r.get("transcript", ""))) > 10)
)
print(f"has formatted_transcript: {has_ft}")
print(f"has transcript field: {has_t}")
print(f"has EITHER: {has_either}  ({has_either / len(records) * 100:.1f}%)")
title_lens = [len(r.get("title", "") or "") for r in records]
desc_lens = [len(r.get("description", "") or "") for r in records]
print(f"avg title len: {sum(title_lens) / len(title_lens):.0f}")
print(f"avg description len: {sum(desc_lens) / len(desc_lens):.0f}")
print("keys on first:", list(records[0].keys())[:35])
