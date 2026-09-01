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
    /* gaia stacks content from the top, which leaves the bottom third of almost every
       slide empty. Centre it vertically and let the heading stay put. */
    justify-content: center;
  }
  /* Last-resort cap only. fit_slide() sizes figures per slide, so this must sit ABOVE
     the heights it assigns (520px = 72vh) or it silently shrinks every figure. */
  section img {
    max-height: 78vh;
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


# The gaia theme renders 1280x720 -- it defines only 16:9 and 4:3, so the 2023 decks'
# "size: 16:12" was always silently ignored. After the heading and the page padding
# roughly 560px of body height is left. Merging two 2023 slides blows past that, so
# every slide is measured and shrunk: image heights first (they cost the most and lose
# the least), then a scoped font size. Background images are free vertically.
BUDGET = 560


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
    px = sum(heights) + 34 * len(text) + 50 * len(tbl)
    if inline and not heights:
        px += 300 * len(inline)          # an unsized inline image renders large
    return px, len(inline)


def drop_stale_columns(slide):
    """A 2023 slide that set `column-count: 2` for two figures should not keep it when
    one of them died with its CDN link -- the survivor gets squeezed into a column and
    half the slide stays blank."""
    imgs = re.findall(r"!\[[^\]]*\]\(", slide)
    if "column-count" in slide and len(imgs) < 2:
        slide = re.sub(r"<style scoped>.*?</style>\n*", "", slide, flags=re.S)
    return slide


def two_pane(slide):
    """One figure under a few bullets fills the left half and wastes the right.

    A slide with text and exactly one inline figure becomes a two-pane slide: text
    left, figure right, using marp's own `bg right:` idiom. Slides that already place
    a background image, already set a column layout, or are figure-only are untouched
    -- there the 2023 author had chosen a layout and it should stand.
    """
    if "column-count" in slide or re.search(r"!\[[^\]]*\bbg\b", slide):
        return slide
    imgs = re.findall(r"!\[([^\]]*)\]\([^)]*\)", slide)
    if len(imgs) != 1:
        return slide
    text = [l for l in slide.split("\n")
            if l.strip() and not l.strip().startswith(("!", "#", "<", "$"))]
    if len(text) < 2:
        # The figure IS the slide: give it whatever height is actually left after the
        # heading and any stray line, rather than a fixed 520 that overflows.
        if re.search(r"\b(width:|w:)\d+", slide):
            return slide                       # the author already sized it
        # 720px canvas, ~40px padding top and bottom, ~95px heading, ~40px per text line
        avail = 720 - 80 - 95 - 40 * len(text) - 20
        avail = max(220, min(avail, 470))
        if re.search(r"height:\d+px", slide):
            return re.sub(r"height:\d+px", f"height:{avail}px", slide)
        return re.sub(r"!\[([^\]]*)\]\(",
                      lambda m: f"![{(m.group(1) + ' ').lstrip()}height:{avail}px](",
                      slide, count=1)
    return re.sub(r"!\[([^\]]*)\]\(", "![bg right:47% contain](", slide, count=1)


def fit_slide(slide):
    px, n_inline = _cost(slide)
    if px <= BUDGET:
        return slide
    if n_inline >= 1:
        cap = 230 if n_inline >= 2 else 380
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
        size = 23 if px < BUDGET * 1.35 else 19
        if "<style scoped>" in slide:
            # fold into the existing block; two scoped blocks on one slide is asking
            # for trouble and the 2023 decks already ship several
            slide = slide.replace("<style scoped>",
                                  f"<style scoped>\nsection {{ font-size: {size}px; }}", 1)
        else:
            slide = f"<style scoped>section {{ font-size: {size}px; }}</style>\n\n" + slide
    # A figure wider than one column is pushed out of view by column-count.
    if "column-count" in slide:
        slide = re.sub(r"\b(width:|w:)(\d+)",
                       lambda m: f"{m.group(1)}{min(int(m.group(2)), 460)}", slide)
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
    ("00_introduction", [2, 3, 4, 5], None),

    # 2. The response, on its natural axis - time relative to the event. 00[16] is the
    #    frame slide that names the four stages, so it must lead them.
    ("00_introduction", [16], None),
    merge("00_introduction", [17, 18], "Before an earthquake"),
    merge("00_introduction", [19, 20], "A few seconds after"),
    ("00_introduction", [21, 22], None),

    # 3. What we actually record. 00[10] and 00[12] carry the same heading; [12] keeps
    #    the worked M5.1 example, so [10] is dropped rather than shown twice.
    merge("00_introduction", [8, 9], "More sensors, recording for longer"),
    ("00_introduction", [12], None),

    # 4. The pipeline. 00[13] lists what gets extracted, so it opens the section and the
    #    six stages then answer it one at a time.
    ("00_introduction", [13], None),
    merge("03_earthquake_detection", [2, 9], "Detect: is there an earthquake?"),
    ("04_phase_picking", [7, 8], None),   # 04[7] is a full slide; merging overflows
    ("05_phase_association", [2], None),
    merge("06_location_and_relocation", [2, 4], "Locate: an inverse problem"),
    ("01_source_and_wave", [59, 60], None),  # both are full slides; merging overflows
    ("09_focal_mechanism", [2, 3], None),  # 09[3] alone carries five images

    # 5. Why the catalogue is the product. These two were previously stacked in front of
    #    the pipeline, where they were a wall of bullets about work not yet described.
    ("00_introduction", [14, 15], None),

    # 6. The same pipeline, rebuilt.
    # Same six stages as section 4, in the same order, so the replacement is visible
    # one-to-one. Then the progression PhaseNet -> EQTransformer -> PhaseNO, and a close
    # on what it cost rather than on what it promised.
    ("03_earthquake_detection", [13], None),   # [14]'s science.org figure is paywalled: blank slide
    merge("04_phase_picking", [9, 16], "Pick: it is a segmentation problem"),
    ("04_phase_picking", [17, 18, 19], None),          # PhaseNet -> EQTransformer -> PhaseNO
    ("05_phase_association", [5], None),               # associate: GaMMA
    ("02_signal_processing", [17], None),              # denoise
    ("07_statistics", [39], None),                     # and downstream, forecasting
    ("04_phase_picking", [20], None),                  # what made it work
    ("03_earthquake_detection", [15], None),           # and what it costs

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
    ("03_earthquake_detection", [3, 6, 7, 8, 10, 11, 12], None),

    (None, None, divider("D · Phase picking")),
    ("04_phase_picking", [2, 3, 14, 15], None),

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

![bg right:52% contain](assets/geysers_ml_catalog.png)

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
    return None   # a dangling citation reads as the caption of whatever survived


def strip_dead(slide):
    """Drop image lines pointing at dead URLs. Return (slide, n_removed)."""
    out, cites, removed = [], [], 0
    for line in slide.split("\n"):
        if any(d in line for d in DEAD) and ("![" in line or line.strip().startswith("<!--")):
            if line.strip().startswith("<!--"):
                continue          # already commented out; just delete it
            removed += 1
            cite = cite_from_url(line)
            if cite:
                cites.append(cite)
            continue
        out.append(line)
    body = "\n".join(out).rstrip()
    for c in cites:
        if c not in body:
            body += f"\n\n{c}"
    return body, removed


# Typos carried over from the 2023 decks. Fixed here rather than on the fall2023
# branch, which is an archive and stays as it was delivered.
RENAME = {
    # two 2023 slides share the title "Deep learning"; say what each one is about
    "### Deep learning\n\n- Generalized similarity search":
        "### Detection, learned\n\n- Generalized similarity search",
    "### Deep learning\n\n- Pros:":
        "### What deep learning costs\n\n- Pros:",
    # 01[60] renders as two slides that carried the same title in 2023
    "### Richter Magnitude (Local magnitude $M_L$)\n\nAn approximate empirical formula":
        "### Richter magnitude: the empirical formula\n\nAn approximate empirical formula",
}

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
    for wrong, right in RENAME.items():
        slide = slide.replace(wrong, right)
    for wrong, right in TYPOS.items():
        slide = slide.replace(wrong, right)
    return slide


def localise_directives(slide):
    """`<!-- footer: x -->` is a GLOBAL marp directive: once one 2023 slide sets it,
    every later slide carries that citation. The underscore form scopes it to the
    slide it is written on, which is what the 2023 decks meant."""
    return (slide.replace("<!-- footer:", "<!-- _footer:")
                 .replace("<!-- header:", "<!-- _header:"))


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
                raw = cache[src][i].strip()
                if re.search(r"^-{3,}\s*$", raw, flags=re.M):
                    raise SystemExit(
                        f"cannot merge {src}[{i}]: it already contains a slide break, so "
                        f"merging produces a broken pair (a headless slide with a stray "
                        f"figure). Select it as a plain slide instead.")
                c = localise_directives(fix_typos(defragment(raw)))
                c, _ = strip_dead(c)
                # drop the chunk's own heading; the merged slide supplies one
                c = "\n".join(l for l in c.split("\n")
                               if not l.strip().startswith(("#", "<!-- footer")))
                if c.strip():
                    body_parts.append(c.strip())
            merged = f"### {heading}\n\n" + "\n\n".join(body_parts)
            # Two figures stacked in one column waste the right half of a 16:9 slide.
            # The 2023 decks already solve this with a scoped two-column section, so
            # reuse that rather than inventing a layout.
            inline = [a for a in re.findall(r"!\[([^\]]*)\]\(", merged)
                      if "bg" not in a.split()]
            if len(inline) >= 2:
                merged = re.sub(r"height:\d+px", "w:470", merged)
                merged = ("<style scoped>\n"
                          "section { column-count: 2; column-gap: 2.5rem; }\n"
                          "h3 { column-span: all; }\n"
                          "p { margin: 0.3em 0; }\n"
                          "</style>\n\n" + merged)
            slides.append(fit_slide(two_pane(merged)))
            continue
        src, idxs, literal = entry
        if literal is not None:
            if literal == "@@CLOSING@@":
                literal = CLOSING.replace("@@SCHEDULE@@", schedule_table())
            slides.append(fit_slide(literal.strip()))
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
            s = localise_directives(fix_typos(defragment(s)))
            s, removed = strip_dead(s)
            if removed and is_empty(s):
                dropped.append(f"{src}[{i}]")
                continue
            if removed:
                s = s.rstrip() + "\n\n" + FIXME
                thinned.append(f"{src}[{i}]")
            slides.append(fit_slide(two_pane(drop_stale_columns(s.strip()))))


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
