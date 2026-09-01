#!/usr/bin/env python
"""Deterministic checks on a topic's notebooks. No LLM, no judgement.

Every check here exists because something went wrong on 2026-08-31 while building topic 1. The
number-provenance check is the important one: all three errors caught that day were claims asserted
in prose that no cell had computed, and every one of them survived a clean execution.

    python tools/check_notebook.py 1
"""
import json, pathlib, re, sys, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "docs" / "notebooks"

# Numerals that are legitimately in prose without being computed.
ALLOW = re.compile(r"""
    ^(19|20)\d\d$            # years
  | ^10\.\d{4}              # DOI prefixes
  | ^[0-9]$                 # single digits: section numbers, small counts
  | ^(0|1)\.0+$             # trivial constants
""", re.X)

def numerals(text):
    """Numbers in prose, minus maths, code spans, links and citations."""
    t = re.sub(r"\$[^$]*\$", " ", text)            # inline maths
    t = re.sub(r"`[^`]*`", " ", t)                 # code spans and DOIs
    t = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", t)     # links
    t = re.sub(r"\|", " ", t)                      # table pipes
    return {m for m in re.findall(r"-?\d+(?:[.,]\d+)*", t) if not ALLOW.match(m)}

def check(n):
    spec = yaml.safe_load((ROOT / "topics.yml").read_text())["topics"] \
        if (ROOT / "topics.yml").exists() else {}
    stem = next((p.stem for p in NB.glob(f"{n:02d}_*.ipynb")
                 if not p.stem.endswith("_solution")), None)
    if not stem:
        sys.exit(f"no notebook for topic {n} in {NB}")
    sol, stu = NB / f"{stem}_solution.ipynb", NB / f"{stem}.ipynb"
    errs, warns, notes = [], [], []

    s = json.loads(sol.read_text())
    s_cells = s["cells"]
    code = [c for c in s["cells"] if c["cell_type"] == "code"]
    mds  = [c for c in s["cells"] if c["cell_type"] == "markdown"]
    src  = lambda c: "".join(c["source"])

    # 1 ── it ran
    bad = [i for i, c in enumerate(code)
           if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    if bad:  errs.append(f"solution has execution errors in code cells {bad}")
    unrun = [i for i, c in enumerate(code) if not c.get("outputs")]
    if unrun: warns.append(f"code cells with no output (never executed?): {unrun}")

    # 2 ── every number in prose came from a cell   <-- the check that matters
    out_text = "\n".join(
        "".join(o.get("text", "")) + json.dumps(o.get("data", {}).get("text/plain", ""))
        for c in code for o in c.get("outputs", []))
    out_nums = set(re.findall(r"-?\d[\d,]*(?:\.\d+)?", out_text))
    def seen(x):
        if x in out_nums: return True
        try:  v = float(x.replace(",", ""))
        except ValueError: return False
        # a rounded quotation of a printed value counts as computed
        for o in out_nums:
            try: ov = float(o.replace(",", ""))
            except ValueError: continue
            # Precision claimed should match precision available: a 1-2 significant-figure
            # number in prose is a deliberate rounding for memorability, not an invented value.
            sig = len(x.replace("-", "").replace(".", "").replace(",", "").lstrip("0"))
            tol = max(abs(v) * (0.05 if sig <= 2 else 0.005), 5e-4)
            # sign-insensitive: prose says "under-reads by 0.38" for a printed -0.38
            if abs(ov - v) < tol or abs(abs(ov) - abs(v)) < tol: return True
        return False
    # Strip the numerals that are LABELS, not claims, or the real ones drown. Eleven of twelve
    # flags on topic 1 were "EPS 207", "week 12", "88 years apart" and instrument constants.
    NOT_A_CLAIM = [
        r"EPS\s*\d+", r"\b[Tt]opic\s*\d+", r"\b[Ww]eek\s*\d+", r"\b[Ss]ection\s*\d+",
        r"\b(?:18|19|20)\d{2}\b",                      # years
        r"10\.\d{4,9}/\S+",                             # DOIs
        r"\d+\s*years apart",
        r"\$[^$]*\$",                                    # inline maths: symbols, not results
    ]
    def prose(c):
        t = src(c)
        for pat in NOT_A_CLAIM:
            t = re.sub(pat, " ", t)
        return t
    unsourced = sorted({x for c in mds for x in numerals(prose(c)) if not seen(x)},
                       key=lambda z: -len(z))
    if unsourced:
        notes.append("numbers in prose with no matching cell output "
                     "(each is a claim to justify or delete):\n      " + ", ".join(unsourced[:25]))

    # 3 ── portability
    for i, c in enumerate(code):
        t = src(c)
        for pat, msg in [(r"['\"](?:/Users|/home|~/|\.\./)", "local filesystem path"),
                         (r"gs://", "gs:// URI — students have no credentials")]:
            if re.search(pat, t): errs.append(f"code cell {i}: {msg}")
    if not any("default_rng" in src(c) or "seed" in src(c) for c in code):
        errs.append("no random seed set anywhere")
    # Every LIVE cell needs its own guard, not one `except` somewhere else in the notebook.
    # Week 1 shipped nine FDSN calls with a retry on only the first, and this check passed it.
    LIVE = r"get_waveforms|urlopen|requests\.get|\.get_events|read_csv\s*\(\s*[A-Z_]*URL|https?://"
    guarded = {m for c in code if re.search(r"except|for attempt|range\(tries", src(c))
                 for m in re.findall(r"def\s+(\w+)", src(c))}
    n_live = 0
    for i, c in enumerate(code):
        t = src(c)
        if not re.search(LIVE, t):
            continue
        n_live += 1
        own = re.search(r"except|for attempt|range\(tries", t)
        helper = any(re.search(rf"\b{g}\s*\(", t) for g in guarded)
        if not (own or helper):
            errs.append(f"code cell {i}: live network call with no retry or fallback — eight people "
                        f"on one wifi will not all reach a live service")
    if n_live > 6:
        warns.append(f"{n_live} live network cells; x8 students in the same two minutes. SCEDC returns "
                     f"'500 STP clients exceeded' under one agent's query rate — mirror the data")

    # 4 ── it is a session, not a script
    t_pre = json.loads(stu.read_text())
    n_yours = sum(1 for c in t_pre["cells"]
                  if c["cell_type"] == "code" and "".join(c["source"]).strip() == "# your code here")
    if n_yours < 2:
        errs.append(f"only {n_yours} exercise(s) — blanked cells in the student notebook; "
                    f"participation is 50% of the grade")
    body = "\n".join(src(c) for c in mds).lower()
    # Neither a "threats" nor a "what remains open" section is required any more. Both were tried
    # and both filled up with speculation and housekeeping rather than science; a limitation worth
    # stating belongs beside the cell that exposes it, where it can be checked.
    if "takeaway" not in body.lower():
        errs.append("no takeaways section - a session must say what a student carries out of it")

    # 5 ── the student notebook is derived, not duplicated
    t = json.loads(stu.read_text())
    scode = [ "".join(c["source"]) for c in t["cells"] if c["cell_type"] == "code" ]
    ccode = [ src(c) for c in code ]
    ident = sum(1 for a in scode if a in ccode)
    if scode and ident / len(scode) > 0.9:
        warns.append(f"{100*ident/len(scode):.0f}% of student code cells are identical to the "
                     f"solution — nothing was deleted")

    # 6 ── is it a hundred minutes of teaching, or a skeleton?
    n_fig = sum(1 for c in code for o in c.get("outputs", [])
                if "image/png" in o.get("data", {}))
    if len(code) < 20:
        errs.append(f"only {len(code)} code cells for a 100-minute live session — a session runs "
                    f"~20-30. Build the model up in steps rather than in one cell.")
    if n_fig < 4:
        errs.append(f"only {n_fig} figure(s). A live session needs one roughly every 15 minutes.")
    if n_yours < 3:
        warns.append(f"{n_yours} exercises for 100 minutes; aim for 3-4")
    long_cells = [i for i, c in enumerate(code) if len(src(c).split(chr(10))) > 30]
    if long_cells:
        warns.append(f"code cells over 30 lines (hard to follow live): {long_cells}")


    # 7 ── dead code and fragile idioms (all found by hand on topic 1, 2026-08-31)
    for i, c in enumerate(code):
        t = src(c)
        if re.search(r"\bif\s+False\b|\bif\s+0\s*:", t):
            errs.append(f"code cell {i}: dead `if False` branch left in a teaching notebook")
        if "searchsorted" in t and ".index" in t:
            warns.append(f"code cell {i}: indexes via .index after a possible student filter — "
                         f"use positional indexing so a filtered frame cannot silently corrupt it")

    # 8 ── checkpoints. An exercise that defines a variable the rest of the notebook needs will
    # strand any student who fumbles it. Topic 1 had exactly this and no way back in.
    all_src = body + "\n".join(src(c) for c in code)
    if "checkpoint" not in all_src.lower():
        errs.append("no checkpoint cells. A student who breaks at minute 40 has no way to rejoin; "
                    "add a `# ── Checkpoint N ── run this if you are behind ──` cell per section")
    # An exercise is a cell the STUDENT notebook blanks. Keying this on marker text meant the
    # check silently matched nothing once the markers changed.
    stu_cells = json.loads(stu.read_text())["cells"]
    ex_idx = [i for i, c in enumerate(stu_cells)
              if c["cell_type"] == "code"
              and "".join(c["source"]).strip() == "# your code here"]
    for j in ex_idx:
        sol = s_cells[j] if j < len(s_cells) and s_cells[j]["cell_type"] == "code" else None
        if not sol: continue
        assigned = set(re.findall(r"^\s*(\w+)\s*=", "".join(sol["source"]), re.M))
        later = "\n".join(src(c) for c in code[code.index(sol)+1:] if sol in code)
        used = {v for v in assigned if re.search(rf"\b{v}\b", later)}
        if used:
            warns.append(f"exercise near cell {j} defines {sorted(used)} which later cells use — "
                         f"put those in a checkpoint so a wrong answer does not cascade")

    # 8b ── takeaways a student can carry. Four rounds of review were spent removing numbers
    # nobody should memorise (-1.3149, 0.0122, 640,926) from sentences meant to be remembered.
    for cell in mds:
        txt = src(cell)
        if "takeaways" not in txt.lower():
            continue
        for line in txt.split("\n"):
            if not line.lstrip().startswith("- "):
                continue
            clean = re.sub(r"\b(?:18|19|20)\d{2}\b", " ", line)      # years are not quantities
            nums = re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", clean)
            def sigfigs(n):
                t = n.replace("-", "").replace(",", "")
                if "." in t:
                    return len(t.replace(".", "").lstrip("0"))
                return len(t.lstrip("0").rstrip("0")) or 1   # 200 -> 1 sig fig, 205 -> 3
            precise = [n for n in nums if sigfigs(n) >= 3]
            if precise:
                warns.append(f"takeaway quotes {precise} - {len(precise[0])}-figure values are "
                             f"evidence, not something a student carries. Round it or cut it: "
                             f"'{line.strip()[:60]}...'")
            elif len(nums) > 2:
                warns.append(f"takeaway carries {len(nums)} numbers; it is probably two takeaways "
                             f"or none: '{line.strip()[:60]}...'")

    # 8c ── seismology takeaways must be attributable. We cannot verify a source mechanically,
    # but a section with no attribution anywhere is certainly synthesised.
    for cell in mds:
        txt = src(cell)
        if "seismology takeaway" not in txt.lower():
            continue
        cited = (re.search(r"10\.\d{4,9}/", txt)                      # a DOI
                 or re.search(r"et al\.", txt)                          # "et al."
                 or re.search(r"\b[A-Z][a-z]+\s*&\s*[A-Z][a-z]+", txt)  # "Hutton & Boore"
                 or re.search(r"\b(?:18|19|20)\d{2}\b", txt))          # a year
        if not cited:
            errs.append("seismology takeaways cite nobody — every one must come from a textbook or "
                        "a paper opened while building, not from synthesis")

    # 9 ── every paper named in the header must do work later, not decorate the intro
    header = "".join(s_cells[0]["source"]) if s_cells else ""
    dois = set(re.findall(r"10\.\d{4,9}/[^\s`\)]+", header))
    rest = body + out_text
    for doi in dois:
        tail = doi.split("/")[-1][:12]
        author = None
        m = re.search(rf"\*\*([A-Z][A-Za-z&,.\s]+?)\s*\(\d{{4}}\)\*\*[^`]*{re.escape(doi)}", header)
        if m: author = m.group(1).split()[0].strip(",")
        if author and author.lower() not in rest.lower():
            errs.append(f"'{author}' is cited in the header but never appears again — a paper that "
                        f"is only mentioned is decoration. Use it or drop it.")

    # 10 ── does each section actually show something?
    secs = [i for i, c in enumerate(s_cells)
            if c["cell_type"] == "markdown" and re.search(r"^## ", "".join(c["source"]), re.M)]
    for a, b_ in zip(secs, secs[1:] + [len(s_cells)]):
        span = s_cells[a:b_]
        title = re.search(r"^## (.+)$", "".join(span[0]["source"]), re.M)
        name = (title.group(1) if title else "").lower()
        if any(w in name for w in ("threats", "could be wrong", "remains open",
                                   "take away", "takeaway", "setup")):
            continue
        if not any(c["cell_type"] == "code" and c.get("outputs") for c in span):
            warns.append(f"section '{(title.group(1)[:48] if title else '?')}' has no computed "
                         f"output — prose asserting what a cell should show")

    print(f"── topic {n}: {stem}")
    print(f"   {len(code)} code cells, {len(mds)} markdown, {n_yours} exercises, "
          f"{sum(1 for c in code for o in c.get('outputs',[]) if 'image/png' in o.get('data',{}))} figures")
    for e in errs:  print(f"   ERROR  {e}")
    for w in warns: print(f"   warn   {w}")
    for t_ in notes: print(f"   note   {t_}")
    print(f"   -> {len(errs)} errors, {len(warns)} warnings")
    return 1 if errs else 0

if __name__ == "__main__":
    sys.exit(check(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
