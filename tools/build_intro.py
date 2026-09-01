#!/usr/bin/env python3
"""Assemble docs/lectures/00_introduction.md from the Fall 2023 decks.

The 2023 offering ran nine lecture decks, ~340 slides. This course is organised by
machine-learning method instead, so session 1 is a single pass down the monitoring
pipeline: what an earthquake is, what the waveform carries, then detection, picking,
association, location, catalogue statistics, mechanism. Every later week attaches
to one stop on that pass.

SELECTION rules used below:
  - weight by where THIS course spends its weeks, not by how much 2023 material
    exists. Source physics had 103 slides in 2023 and gets 19 here (it is EPS 130
    background, and only weeks 1 and 12 return to it); location keeps 21, because
    the chi-square / uncertainty material is the spine of weeks 1, 3, 13 and 14.
  - keep the arc "classical method -> its failure -> what learning changed" wherever
    2023 already told it (detection and picking do; that is why they keep most slides).
  - cut derivations the course does not use: Green's functions, radiation-pattern
    algebra, directivity, energy partitioning, apparent stress, filter design.

Run:  python tools/build_intro.py       (rewrites docs/lectures/00_introduction.md)
"""
import datetime as dt
import re
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
REF = "origin/fall2023"
OUT = REPO / "docs/lectures/00_introduction.md"

FRONT = """---
marp: true
paginate: true
theme: gaia
backgroundColor: #fff
style: |
  section {
    font-size: 28px;
  }
  img + br + em {
    font-style: normal;
    display: inherit;
    text-align: right;
    font-size: 70%;
  }
---"""

TITLE = """
# Observational Seismology

### EPS 207 · Fall 2026

Weiqiang Zhu · Tuesdays 9:00-10:59 · McCone 325
"""

# A divider before each part. (heading, question, classical answer, what learning
# changed, which weeks return to it)
def divider(n, name, question, classical, learned, weeks):
    return f"""
<!-- _class: lead -->

# Part {n}
# {name}

**{question}**

Classically: {classical}
With learning: {learned}

*Returns in {weeks}*
"""

# (source deck, [chunk indices]) -- indices from the 2023 files, see notes/.
PARTS = [
    (None, None, TITLE),
    (None, None, divider(
        1, "Why we monitor",
        "What is the cost of not knowing?",
        "instrument the ground and wait.",
        "the same instruments, read faster and more completely.",
        "every week")),
    ("00_introduction", list(range(2, 23)), None),

    (None, None, divider(
        2, "What an earthquake is",
        "What quantity are we actually estimating?",
        "a double couple, six moment-tensor components, one magnitude.",
        "nothing yet - this is the vocabulary the rest of the course uses.",
        "weeks 1 and 12")),
    ("01_source_and_wave",
     [2, 3, 4, 5, 6, 7, 9, 29, 30, 32, 45, 59, 60, 61, 62, 63, 70, 71, 74], None),

    (None, None, divider(
        3, "The waveform",
        "Which part of this record is signal?",
        "a bandpass filter, chosen by eye.",
        "a learned mask - but you must show it helps downstream, not that it looks better.",
        "week 11")),
    ("02_signal_processing", [2, 3, 6, 7, 10, 12, 13, 17, 20], None),

    (None, None, divider(
        4, "Detection",
        "Is there an earthquake in this hour of data?",
        "amplitude threshold, then STA/LTA, then template matching.",
        "a detector trained on catalogues, finding events below every earlier threshold.",
        "weeks 6, 7 and 9")),
    ("03_earthquake_detection", [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15], None),

    (None, None, divider(
        5, "Phase picking",
        "When did P and S arrive, on each station?",
        "an analyst, by hand, at a few tens of thousands of picks a year.",
        "segmentation of the trace - the single change that grew catalogues by an order of magnitude.",
        "week 6")),
    ("04_phase_picking", [2, 3, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20], None),

    (None, None, divider(
        6, "Association",
        "Which picks belong to the same earthquake?",
        "grid search over candidate origins.",
        "clustering with a physical forward model inside it.",
        "week 4")),
    ("05_phase_association", [2, 3, 4, 5, 6, 7, 8], None),

    (None, None, divider(
        7, "Location, and how wrong it is",
        "Where was it, and how far could that be off?",
        "linearised least squares; a covariance matrix; a chi-square ellipse.",
        "the same inverse problem, differentiated automatically, with a posterior instead of an ellipse.",
        "weeks 1, 3, 13 and 14")),
    ("06_location_and_relocation",
     [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25], None),

    (None, None, divider(
        8, "What a catalogue is for",
        "What do a million earthquakes say that one does not?",
        "Gutenberg-Richter, Omori, ETAS - three laws and a few parameters.",
        "the same laws, but now the catalogue is large enough that the parameters move.",
        "weeks 1 and 4")),
    ("07_statistics",
     [2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 24, 25, 26, 29, 33, 39], None),

    (None, None, divider(
        9, "Mechanism",
        "How did the fault move?",
        "first motions on a stereonet, or a waveform inversion.",
        "machine polarities at a scale that makes mechanisms routine.",
        "week 12")),
    ("09_focal_mechanism", [2, 3, 4, 5, 6], None),

    (None, None, """
<!-- _class: lead -->

# Part 10
# This course
"""),
    ("00_introduction", [23, 24, 25, 26, 27], None),
]

def schedule_table():
    """Same source and same dates as docs/README.md: Sep 1 is the introduction,
    topics run from Sep 8, and any topic past the 13th Tuesday is unscheduled."""
    spec = yaml.safe_load((REPO / "topics.yml").read_text())
    first = dt.date(2026, 9, 1)
    rows = ["| Date | Seismology | Machine learning |", "| --- | --- | --- |",
            "| 09/01 | **Introduction** | *today* |"]
    for i, t in enumerate(sorted(spec["topics"], key=lambda x: x["n"])[:13], start=1):
        d = first + dt.timedelta(weeks=i)
        rows.append(f"| {d:%m/%d} | {t.get('task','')} | {t['title']} |")
    return "\n".join(rows)


CLOSING = """
### Schedule

@@SCHEDULE@@

---

### How each week runs

One notebook, worked in the room, not watched.

1. A published claim, and the paper it comes from.
2. The data, and what is wrong with it.
3. **A baseline** - the simplest thing that could work.
4. The method, against that baseline, on the same data and the same metric.
5. What would have to be true for the result to be wrong.

A method that does not beat its baseline has not earned the week.

---

### Final project

The Geysers: the most seismically active field in California, and the shaking is
a side effect of an industrial process.

- One field, one catalogue, your own question.
- Proposal in week 5, presentations in week 15.
- `docs/project.md` lists nine questions - or bring your own.

---

### Grading

- Attendance and participation (50%)
- Final project (50%)
    - Project proposal (10%)
    - Project presentation (20%)
    - Project report (20%)
- Extra credit (up to 10%)

---

### Questions?
"""


# Images that no longer resolve (checked 2026-09-01). The silverchair/OUP links were
# signed CDN URLs carrying "Expires=1698..." - they lapsed in November 2023 and cannot
# be recovered. Matching images are stripped; a slide left with nothing is dropped.
DEAD = [
    "earthquake.ca.gov/wp-content/uploads/sites/8/2020/09/android_alerts.gif",
    "cdn-dfdfc.nitrocdn.com",
    "gsw.silverchair-cdn.com",
    "oup.silverchair-cdn.com",
    "www.jmp.com/en_no/statistics-knowledge-portal",
    "www.jreast.co.jp/e/development/theme/safety",
    "www.mdpi.com/sensors/sensors-19-00597",
    "www.mlpack.org/gsocblog/images/5_clusters_QGMM.gif",
    "www.open.edu/openlearn/pluginfile.php",
    "www.researchgate.net/profile/Eloi-Batlle",
    "www.researchgate.net/publication/320856220",
]

FIXME = "<!-- FIXME: figure lost to an expired CDN link; paste a screenshot here -->"

# The silverchair/OUP URLs embed the paper's DOI in the path (10.1785_0220190052 ->
# 10.1785/0220190052). Titles below verified against Crossref on 2026-09-01, so a slide
# that lost its only figure still cites what the figure was.
TITLES = {
    "10.1785/0220190052": "Zhang, Ellsworth & Beroza (2019), Rapid Earthquake Association and Location, SRL",
    "10.1785/0120220182": "McBrearty & Beroza (2023), Earthquake Phase Association with Graph Neural Networks, BSSA",
    "10.1785/0120010200": "Hardebeck & Shearer (2002), A New Method for Determining First-Motion Focal Mechanisms, BSSA",
    "10.1093/gji/ggy423": "Zhu & Beroza (2018), PhaseNet, GJI",
}


def cite_from_url(line):
    """Recover a citation from a signed-CDN image URL that embeds the DOI."""
    m = re.search(r"(10\.\d{4}_[0-9A-Za-z_.]+?)/", line)
    if not m:
        return None
    doi = m.group(1).replace("_", "/")
    title = TITLES.get(doi)
    return f"*Figure: [{title or doi}](https://doi.org/{doi})*" if title else None


def strip_dead(slide):
    """Drop image lines pointing at dead URLs. Return (slide, n_removed)."""
    out, removed = [], 0
    for line in slide.split("\n"):
        if any(d in line for d in DEAD) and ("![" in line or line.strip().startswith("<!--")):
            if line.strip().startswith("<!--"):
                continue          # already commented out; just delete it
            removed += 1
            cite = cite_from_url(line)
            if cite and cite not in out:
                out.append(cite)
            continue
        out.append(line)
    return "\n".join(out), removed


def is_empty(slide):
    """A slide with no image and no real prose left."""
    import re as _re
    if "![" in slide:
        return False
    txt = _re.sub(r"<style scoped>.*?</style>", "", slide, flags=_re.S)
    txt = _re.sub(r"<!--.*?-->", "", txt, flags=_re.S)
    txt = "\n".join(l for l in txt.split("\n") if not l.strip().startswith("#"))
    return len(txt.strip()) < 40


def load(name):
    txt = subprocess.run(
        ["git", "show", f"{REF}:docs/lectures/{name}.md"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    return re.split(r"\n---\n", txt)


def main():
    cache, slides = {}, []
    dropped, thinned = [], []
    for src, idxs, literal in PARTS:
        if literal is not None:
            slides.append(literal.strip())
            continue
        if src not in cache:
            cache[src] = load(src)
        chunks = cache[src]
        for i in idxs:
            s = chunks[i].strip()
            if not s:
                raise SystemExit(f"{src}[{i}] is empty - the deck changed, re-index it")
            s, removed = strip_dead(s)
            if removed and is_empty(s):
                dropped.append(f"{src}[{i}]")
                continue
            if removed:
                s = s.rstrip() + "\n\n" + FIXME
                thinned.append(f"{src}[{i}]")
            slides.append(s.strip())
    slides.append(CLOSING.strip().replace("@@SCHEDULE@@", schedule_table()))

    body = FRONT + "\n\n" + "\n\n---\n\n".join(slides) + "\n"
    OUT.write_text(body)

    # restore every local asset the selected slides reference
    refs = sorted(set(re.findall(r"\(((?:assets|codes)/[^)\s]+)\)", body)) |
                  set(re.findall(r"!\[[^\]]*\]\((assets/[^)\s]+)\)", body)))
    missing = []
    for r in refs:
        dest = OUT.parent / r
        if dest.exists():
            continue
        got = subprocess.run(["git", "show", f"{REF}:docs/lectures/{r}"],
                             cwd=REPO, capture_output=True)
        if got.returncode:
            missing.append(r)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(got.stdout)
        print(f"  restored {r}")

    n = body.count("\n---\n") + len(re.findall(r"\n-{4,}\n", body))
    print(f"{OUT.relative_to(REPO)}: ~{n} slides, {len(body)} chars")
    ext = len(re.findall(r"https?://[^)\s]+\.(?:png|jpe?g|gif|webp|svg|mp4)", body))
    print(f"  {len(refs)} local refs, {ext} remote images")
    if dropped:
        print(f"  dropped {len(dropped)} slide(s) left empty by dead images: {', '.join(dropped)}")
    if thinned:
        print(f"  {len(thinned)} slide(s) lost a figure: {', '.join(thinned)}")
    if missing:
        print("  MISSING (not in fall2023):", *missing, sep="\n    ")


if __name__ == "__main__":
    main()
