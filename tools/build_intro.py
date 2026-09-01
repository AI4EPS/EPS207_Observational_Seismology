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
size: 16:12
style: |
  section {
    font-size: 28px;
  }
  /* The 2023 decks mix slides authored for 16:9 and 16:12 and several set an explicit
     image height that overflows the frame. Cap images so a slide cannot run off. */
  section img {
    max-height: 62vh;
    object-fit: contain;
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

# ── The argument ──────────────────────────────────────────────────────────────
# Day one has to make ONE claim and support it, not tour nine decks. The claim:
#
#   Seismology's catalogues grew by orders of magnitude because the pipeline that
#   builds them was replaced with learned models. That same scale makes it much
#   easier to fool yourself, so every week of this course tests a published claim
#   against a baseline and an honest error bar.
#
# Six movements: why it matters -> what we have -> how a catalogue is built ->
# what learning changed -> why that is dangerous -> how the course runs.
#
# Everything else from 2023 moves to an APPENDIX behind a divider: present for
# questions and reference, out of the way of the argument. Roughly 50 slides in
# front, ~100 behind.

def divider(name, sub=""):
    # Title only. Any strapline here reads as filler on a projector.
    return f"""
<!-- _class: lead -->

# {name}
"""


# The three slides that carry the course's own argument. Every number here was
# measured in preparing this course, on the DeVries aftershock data used in week 3,
# except where a paper is named. Do not round them or restate them from memory.
CATCH_1 = """
### A deep net, and a number with no parameters

DeVries et al. (2018, *Nature*) predicted where aftershocks occur from static stress
change, with a 13,451-parameter neural network.

| model | fitted parameters | AUC |
| --- | --- | --- |
| the published network | 13,451 | 0.8486 |
| **max shear stress change** | **0** | **0.8474** |
| logistic regression | 13 | 0.8310 |

Mignan & Broccardo (2019, *Nature*) showed a two-parameter model matches the network.
Measured here: a single stress quantity, **nothing fitted at all**, comes within 0.0012.
"""

CATCH_2 = """
### How to get +0.067 AUC out of nothing

Same data. Add one extra feature: a **random number, constant within each mainshock**,
carrying no information whatsoever.

| split | what the noise feature is worth |
| --- | --- |
| random split over cells | **+0.067 AUC** |
| split by mainshock | −0.032 AUC |

That is roughly **four times** the entire advantage the deep network was reported to have.

Cells around one mainshock share a rupture and a stress field. Split them at random and
the model reads the answer off the group it is in.
"""

CATCH_3 = """
### The error bar decides the result

The same comparison, bootstrapped two ways:

| resampled over | 95% interval | reads as |
| --- | --- | --- |
| 1,378,120 cells | [−0.0194, −0.0153] | a clear result |
| **33 independent earthquakes** | **[−0.0280, +0.0059]** | **no result** |

**8.3x wider**, and it now contains zero. The data did not change; the assumption about
what counts as an independent observation did.

So: every week, a **baseline** and an **honest interval**.
"""


# 16:12 at 28px leaves roughly 960px of usable height once the heading is drawn.
# Merging two 2023 slides routinely blows past that, so every slide is measured and
# shrunk to fit: image heights first (they cost the most and lose the least), then a
# scoped font size. Background images are free vertically and are not counted.
BUDGET = 820


def _cost(slide):
    # A 2023 chunk can hold several rendered slides (marp breaks on any thematic
    # rule), so cost the worst one rather than the sum.
    parts = re.split(r"^-{3,}\s*$", slide, flags=re.M)
    if len(parts) > 1:
        return max((_cost(x) for x in parts), key=lambda t: t[0])
    imgs = re.findall(r"!\[([^\]]*)\]\(", slide)
    inline = [a for a in imgs if "bg" not in a.split()]
    heights = [int(m) for m in re.findall(r"height:(\d+)px", slide)]
    text = [l for l in slide.split("\n")
            if l.strip() and not l.strip().startswith(("!", "#", "<", "|"))]
    tbl = [l for l in slide.split("\n") if l.strip().startswith("|")]
    px = sum(heights) + 34 * len(text) + 30 * len(tbl)
    if inline and not heights:
        px += 300 * len(inline)          # an unsized inline image renders large
    return px, len(inline)


def fit_slide(slide):
    px, n_inline = _cost(slide)
    if px <= BUDGET:
        return slide
    if n_inline >= 1:
        cap = 300 if n_inline >= 2 else 420
        slide = re.sub(r"height:(\d+)px",
                       lambda m: f"height:{min(int(m.group(1)), cap)}px", slide)
        # an inline image with no height at all gets one, or it fills the slide
        # only size an image that carries NO sizing directive of its own
        def _size(m):
            alt = m.group(1)
            if re.search(r"\b(bg|w:|h:|width:|height:)|\d+%", alt):
                return m.group(0)
            return f"![{(alt + ' ').lstrip()}height:{cap}px]("
        slide = re.sub(r"!\[([^\]]*)\]\(", _size, slide)
        px, _ = _cost(slide)
    if px > BUDGET:
        size = 24 if px < BUDGET * 1.35 else 21
        slide = f"<style scoped>section {{ font-size: {size}px; }}</style>\n\n" + slide
    return slide


def merge(src, idxs, heading):
    """Combine several 2023 chunks into ONE slide under a single heading.

    The 2023 decks spread one idea over two or three slides -- a question, then its
    figure, then a variant -- which is fine at 340 slides and wrong in a 100-minute
    introduction. Each source heading is dropped and the bodies are concatenated in
    order, so the images and links survive but the idea arrives once.
    """
    return ("MERGE", src, idxs, heading)


MAIN = [
    (None, None, TITLE),

    # 1. Concrete and local first: what an earthquake does, and the fault under campus.
    (None, None, divider("1 · Earthquakes")),
    ("00_introduction", [2, 3, 4, 5], None),

    # 2. The response, on its natural axis - time relative to the event. 00[16] is the
    #    frame slide that names the four stages, so it must lead them.
    (None, None, divider("2 · Before, during, after")),
    ("00_introduction", [16], None),
    merge("00_introduction", [17, 18], "Before an earthquake"),
    merge("00_introduction", [19, 20], "A few seconds after"),
    ("00_introduction", [21, 22], None),

    # 3. What we actually record. 00[10] and 00[12] carry the same heading; [12] keeps
    #    the worked M5.1 example, so [10] is dropped rather than shown twice.
    (None, None, divider("3 · The data")),
    merge("00_introduction", [8, 9], "More sensors, recording for longer"),
    ("00_introduction", [12], None),

    # 4. The pipeline. 00[13] lists what gets extracted, so it opens the section and the
    #    six stages then answer it one at a time.
    (None, None, divider("4 · From waveforms to a catalogue")),
    ("00_introduction", [13], None),
    merge("03_earthquake_detection", [2, 9], "Detect: is there an earthquake?"),
    ("04_phase_picking", [7, 8], None),   # 04[7] is a full slide; merging overflows
    ("05_phase_association", [2], None),
    merge("06_location_and_relocation", [2, 4], "Locate: an inverse problem"),
    merge("01_source_and_wave", [59, 60], "Size it: magnitude"),
    ("09_focal_mechanism", [2, 3], None),  # 09[3] alone carries five images

    # 5. Why the catalogue is the product. These two were previously stacked in front of
    #    the pipeline, where they were a wall of bullets about work not yet described.
    (None, None, divider("5 · What a catalogue is for")),
    ("00_introduction", [14, 15], None),

    # 6. The same pipeline, rebuilt.
    (None, None, divider("6 · What learning changed")),
    merge("03_earthquake_detection", [13, 14], "Detection, learned"),
    merge("04_phase_picking", [9, 16], "Picking is segmentation"),
    ("04_phase_picking", [17, 20], None),
    ("05_phase_association", [5], None),
    ("02_signal_processing", [17], None),
    ("07_statistics", [39], None),
    merge("00_introduction", [25, 26], "Why machine learning"),

    # 7. The course's own argument.
    (None, None, divider("7 · The catch")),
    (None, None, CATCH_1),
    (None, None, CATCH_2),
    (None, None, CATCH_3),

    (None, None, divider("8 · This course")),
    ("00_introduction", [24], None),
]

APPENDIX = [
    (None, None, divider("Appendix")),

    (None, None, divider("A · Earthquake source")),
    ("01_source_and_wave",
     [2, 3, 4, 5, 6, 7, 9, 29, 30, 32, 45, 61, 62, 63, 70, 71, 74], None),

    (None, None, divider("B · Signal processing")),
    ("02_signal_processing", [2, 3, 6, 7, 10, 12, 13, 20], None),

    (None, None, divider("C · Detection")),
    ("03_earthquake_detection", [3, 6, 7, 8, 10, 11, 12, 15], None),

    (None, None, divider("D · Phase picking")),
    ("04_phase_picking", [2, 3, 14, 15, 18, 19], None),

    (None, None, divider("E · Association")),
    ("05_phase_association", [3, 4, 6, 7, 8], None),

    (None, None, divider("F · Location and its uncertainty")),
    ("06_location_and_relocation",
     [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25], None),

    (None, None, divider("G · Catalogue statistics")),
    ("07_statistics",
     [2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 24, 25, 26, 29, 33], None),

    (None, None, divider("H · Focal mechanism")),
    ("09_focal_mechanism", [4, 5], None),
]

PARTS = MAIN + [(None, None, "@@CLOSING@@")] + APPENDIX


def schedule_table():
    """Same source and dates as docs/README.md: Sep 1 is the introduction, topics run
    from Sep 8, and topics.yml order is the schedule order."""
    spec = yaml.safe_load((REPO / "topics.yml").read_text())
    first = dt.date(2026, 9, 1)
    rows = ["| Date | Seismology | Machine learning |", "| --- | --- | --- |",
            "| 09/01 | **Introduction** | *today* |"]
    for i, t in enumerate(spec["topics"][:13], start=1):
        d = first + dt.timedelta(weeks=i)
        rows.append(f"| {d:%m/%d} | {t.get('task','')} | {t['title']} |")
    return "\n".join(rows)


CLOSING = """
### Schedule

@@SCHEDULE@@

---

### Final project

**The Geysers** - the most seismically active field in California, where the shaking is
a side effect of an industrial process.

---

### Grading

- Attendance and participation (50%)
- Final project (50%)

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


# Typos carried over from the 2023 decks. Fixed here rather than on the fall2023
# branch, which is an archive and stays as it was delivered.
TYPOS = {
    "Earthquake monitoring and earthquake rick?": "Earthquake monitoring and earthquake risk?",
    "How are information extracted/determined?": "How is information extracted?",
    "How to use these information?": "How is this information used?",
    "What additional information can we get from millions of earthquakes?":
        "What can we learn from millions of earthquakes?",
    "Faimilar with seismic data": "Familiar with seismic data",
    "Siminlarity search": "Similarity search",
    "trade-off between event dpeth and origin time": "trade-off between event depth and origin time",
    "The far-\ufb01eld radiation pattern": "The far-field radiation pattern",
    "Self-Similar Earthquake Scaling": "Self-similar earthquake scaling",
    "Stess and Radiated Energy": "Stress and radiated energy",
    "fractial scaling": "fractal scaling",
    "What controls the slop $b$?": "What controls the slope $b$?",
    "Obpsy": "ObsPy",
}


def fix_typos(slide):
    for wrong, right in TYPOS.items():
        slide = slide.replace(wrong, right)
    return slide


def defragment(slide):
    """Marp renders `* item` lists as fragments that appear one click at a time.
    Rewrite the marker to `-` so every slide arrives whole. Only list markers are
    touched: `*emphasis*` and `**bold**` need the asterisk and are left alone."""
    return "\n".join(
        re.sub(r"^(\s*)\*(\s)", r"\1-\2", ln) if re.match(r"^\s*\*\s", ln) else ln
        for ln in slide.split("\n"))


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
    for entry in PARTS:
        if entry[0] == "MERGE":
            _, src, idxs, heading = entry
            if src not in cache:
                cache[src] = load(src)
            body_parts = []
            for i in idxs:
                c = fix_typos(defragment(cache[src][i].strip()))
                c, _ = strip_dead(c)
                # drop the chunk's own heading; the merged slide supplies one
                c = "\n".join(l for l in c.split("\n")
                               if not l.strip().startswith(("#", "<!-- footer")))
                if c.strip():
                    body_parts.append(c.strip())
            slides.append(fit_slide(f"### {heading}\n\n" + "\n\n".join(body_parts)))
            continue
        src, idxs, literal = entry
        if literal is not None:
            if literal == "@@CLOSING@@":
                literal = CLOSING.replace("@@SCHEDULE@@", schedule_table())
            slides.append(literal.strip())
            continue
        if src not in cache:
            cache[src] = load(src)
        chunks = cache[src]
        for i in idxs:
            s = chunks[i].strip()
            if not s:
                raise SystemExit(f"{src}[{i}] is empty - the deck changed, re-index it")
            if "09/18" in s or "Full-waveform Inversion" in s:
                raise SystemExit(
                    f"{src}[{i}] is the FALL 2023 schedule - never select it; the live "
                    f"schedule is generated by schedule_table() from topics.yml")
            s = fix_typos(defragment(s))
            s, removed = strip_dead(s)
            if removed and is_empty(s):
                dropped.append(f"{src}[{i}]")
                continue
            if removed:
                s = s.rstrip() + "\n\n" + FIXME
                thinned.append(f"{src}[{i}]")
            slides.append(fit_slide(s.strip()))


    body = FRONT + "\n\n" + "\n\n---\n\n".join(slides) + "\n"
    OUT.write_text(body)

    # Restore every local asset the selected slides reference. Paths appear as
    # `assets/x.png` and `./assets/x.png`, and names with spaces arrive URL-encoded
    # (%20), so decode before looking the file up in git and before writing it.
    from urllib.parse import unquote
    refs = sorted({unquote(m) for m in
                   re.findall(r"\((?:\./)?((?:assets|codes)/[^)\s]+)\)", body)})
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
