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

# Students open notebooks on Berkeley DataHub via nbgitpuller: the link clones/updates the repo
# into their account and opens the file. Change HUB if the course moves to a departmental hub.
HUB = "https://datahub.berkeley.edu"
REPO = "https://github.com/AI4EPS/EPS207_Observational_Seismology"
BRANCH = "main"


def datahub(path):
    from urllib.parse import quote
    urlpath = f"lab/tree/{REPO.rsplit('/', 1)[-1]}/{path}"
    return (f"{HUB}/hub/user-redirect/git-pull?repo={quote(REPO, safe='')}"
            f"&urlpath={quote(urlpath, safe='')}&branch={BRANCH}")


nb_dir = ROOT / "docs" / "notebooks"
built = {}
for f in sorted(nb_dir.glob("*.ipynb")):
    if f.stem.endswith("_solution"):
        continue                                  # solutions are never published
    m = re.match(r"(\d+)_", f.name)
    if m:
        built[int(m.group(1))] = f"docs/notebooks/{f.name}"

rows = []
for k, d in enumerate(DATES, start=1):
    t = topics.get(k)
    title = t["title"] if t else "TBA"
    link = f"[{title}]({datahub(built[k])})" if k in built else title
    task = t.get("task", "") if t else ""
    rows.append(f"| {d:%b %-d} | {task} | {link} |")

readme = f"""# EPS 207 · Laboratory in Observational Seismology

Seismology is the study of earthquakes and the seismic waves that travel through the Earth. This
course covers the main tasks of observational seismology — magnitude estimation, phase picking,
event location, focal mechanism determination, ground-motion prediction, and seismic tomography —
together with the machine learning methods now used for each.

## Time and place

- **Tuesdays 9:00–10:59 am, 325 McCone Hall**, 1 September – 1 December 2026
- Final project presentations **Tuesday 8 December** (RRR week)
- Instructor: Weiqiang Zhu · 285 McCone Hall · `zhuwq@berkeley.edu`

## Schedule

| Date | Seismology | Machine learning |
|---|---|---|
{chr(10).join(rows)}
| {PRESENT:%b %-d} | Final project presentations | |

## Assessment

| | |
|---|---|
| Participation | 50% — the notebooks are worked in the room, not watched |
| Project | 50% — presentation 20%, report 20% |

## What you need

**Python**: `numpy`, `pandas`, `scikit-learn`, `pytorch`, plus `matplotlib`, `obspy` and `scipy`.

## Previous offerings

Fall 2023 is preserved on the [`fall2023`](https://github.com/AI4EPS/EPS207_Observational_Seismology/tree/fall2023)
branch, including the lecture slides.
"""
(ROOT / "docs" / "README.md").write_text(readme)

# ── mkdocs nav
mk = (ROOT / "mkdocs.yml").read_text()
nav = ["nav:", "  - Overview: README.md"]
if (ROOT / "docs" / "project.md").exists():
    nav.append("  - Final project: project.md")
if built:
    nav.append("  - Notebooks:")
    for k in sorted(built):
        nav.append(f'    - "{k}. {topics[k]["title"]}": {built[k].replace("docs/", "", 1)}')
mk = re.sub(r"^nav:.*?(?=^theme:)", "\n".join(nav) + "\n\n", mk, flags=re.S | re.M)
(ROOT / "mkdocs.yml").write_text(mk)

print(f"docs/README.md: {len(DATES)} sessions, {len(built)} notebook(s) linked {sorted(built)}")
print("mkdocs.yml nav regenerated")
