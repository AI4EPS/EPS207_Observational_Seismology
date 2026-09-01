#!/usr/bin/env python
"""Generate docs/README.md and the mkdocs nav from topics.yml.

Both are GENERATED. Edit topics.yml and re-run; never hand-edit the outputs, or the schedule on the
site and the schedule the course actually follows will drift apart -- which is how the site spent
three years advertising a Monday class that met on Tuesday.

    python tools/make_site.py
"""
import datetime as dt
import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = yaml.safe_load((ROOT / "topics.yml").read_text())
topics = {t["n"]: t for t in spec["topics"]}

FIRST = dt.date(2026, 9, 1)                      # Tue 1 Sep 2026
DATES = [FIRST + dt.timedelta(weeks=k) for k in range(14)]
PRESENT = FIRST + dt.timedelta(weeks=14)         # Dec 8, RRR week

nb_dir = ROOT / "docs" / "notebooks"
built = {}
for f in sorted(nb_dir.glob("*.ipynb")):
    if f.stem.endswith("_solution"):
        continue                                  # solutions are never published
    m = re.match(r"(\d+)_", f.name)
    if m:
        built[int(m.group(1))] = f"notebooks/{f.name}"

rows = []
for k, d in enumerate(DATES, start=1):
    t = topics.get(k)
    title = t["title"] if t else "TBA"
    link = f"[{title}]({built[k]})" if k in built else title
    rows.append(f"| {k} | {d:%b %-d} | {link} |")

readme = f"""# EPS 207 · Laboratory in Observational Seismology

**Fall 2026 · University of California, Berkeley · Department of Earth and Planetary Science**

Machine learning now builds earthquake catalogues, locates events, and inverts for structure.
Each week takes one **method** and one **published result**: you work the notebook live in class,
reproduce part of the result, then try to break it.

## Time and place

- **Tuesdays 9:00–10:59 am, 325 McCone Hall**, 1 September – 1 December 2026
- Final project presentations **Tuesday 8 December** (RRR week)
- Instructor: Weiqiang Zhu · 285 McCone Hall · `zhuwq@berkeley.edu`

## Schedule

| # | Date | Topic |
|---|---|---|
{chr(10).join(rows)}
| — | {PRESENT:%b %-d} | Final project presentations |

Notebooks are linked as they are released, normally the week before the session.

## Assessment

| | |
|---|---|
| Participation | 50% — the notebooks are worked in the room, not watched |
| Project | 50% — proposal 10%, presentation 20%, report 20% |

There are no problem sets. The project is yours to choose; a region or dataset of your own works
best, and the proposal is due in week 5.

## What you need

**Python**: `numpy`, `pandas`, `scikit-learn`, `pytorch`, plus `matplotlib`, `obspy` and `scipy`.
Everything runs on **Google Colab** or **Berkeley DataHub** — no local install and no GPU is
required for any session.

**Data** is public throughout: USGS, IRIS, NCEDC and SCEDC web services, HuggingFace, and datasets
released with this repository. You will never need credentials.

**Background**: graduate standing. No prior seismology is assumed, but you should be comfortable
writing Python and reading a paper.

## Previous offerings

Fall 2023 is preserved on the [`fall2023`](https://github.com/AI4EPS/EPS207_Observational_Seismology/tree/fall2023)
branch, including the lecture slides.
"""
(ROOT / "docs" / "README.md").write_text(readme)

# ── mkdocs nav
mk = (ROOT / "mkdocs.yml").read_text()
nav = ["nav:", "  - Overview: README.md"]
if built:
    nav.append("  - Notebooks:")
    for k in sorted(built):
        nav.append(f'    - "{k}. {topics[k]["title"]}": {built[k]}')
mk = re.sub(r"^nav:.*?(?=^theme:)", "\n".join(nav) + "\n\n", mk, flags=re.S | re.M)
(ROOT / "mkdocs.yml").write_text(mk)

print(f"docs/README.md: {len(DATES)} sessions, {len(built)} notebook(s) linked {sorted(built)}")
print("mkdocs.yml nav regenerated")
