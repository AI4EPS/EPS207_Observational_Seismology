#!/usr/bin/env python
"""Emit the full planning table from topics.yml -> ../notes/schedule.md

The working table (ML method / research problem / anchor -> baseline and check) lived only in a
chat transcript and had to be recovered from a JSONL by grep. It is a VIEW of topics.yml, so
generate it; do not keep a copy.

    python tools/make_table.py
"""
import datetime as dt
import pathlib
import textwrap

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = yaml.safe_load((ROOT / "topics.yml").read_text())
FIRST = dt.date(2026, 9, 1)


def cell(text):
    return " ".join(str(text or "").split()).replace("|", "\\|")


rows = []
for week, t in enumerate(sorted(spec["topics"], key=lambda x: x["n"]), start=1):
    n = t["n"]                                    # stable topic id; gaps are retired topics
    date = (FIRST + dt.timedelta(weeks=week)).strftime("%b %-d")   # Sep 1 is the introduction
    pap = t.get("papers") or {}
    anchor = f"**origin** {cell(pap.get('origin', 'TBD'))}<br>**sota** {cell(pap.get('sota', 'TBD'))}"
    anchor += f"<br>**baseline** {cell(t.get('baseline'))}"
    if t.get("data_hint"):
        anchor += f"<br>**data** {cell(t['data_hint'])}"
    finding = cell(t.get("finding")) or "— no recon yet —"
    rows.append(
        f"| {n} | {date} | **{cell(t['title'])}** | **{cell(t.get('task'))}** — "
        f"{cell(t.get('question'))} | {anchor} |\n"
        f"| | | *concepts* | {'; '.join(t.get('concepts', []))} | *finding* {finding} |"
    )

out = f"""# EPS 207 Fall 2026 — the full planning table

**Generated from `topics.yml` by `tools/make_table.py`. Do not edit; edit the spine and re-run.**

{spec['meeting']}

| # | Date | ML method | Seismology problem | Anchor → baseline and check |
|---|---|---|---|---|
{chr(10).join(rows)}
"""
f = ROOT.parent / "notes" / "schedule.md"
f.write_text(out)
print(f"wrote {f} — {len(spec['topics'])} topics")
