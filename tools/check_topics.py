#!/usr/bin/env python3
"""Validate topics.yml, the course spine. Run after every edit to it.

Nothing else checks this file, so a topic could silently lose a paper, grow a fourth concept, or
claim a finding it never measured. Usage: python3 tools/check_topics.py
"""
import pathlib, re, sys, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTES = ROOT.parent / "notes" / "topics"
DOI = re.compile(r"10\.\d{4,9}/\S+")

spec = yaml.safe_load((ROOT / "topics.yml").read_text())
errs, warns = [], []
seen_n, seen_slug = set(), set()

# which topics have a persisted recon report, keyed by leading number
recon = {}
for f in sorted(NOTES.glob("*.md")):
    m = re.match(r"(\d+)_", f.name)
    if m:
        recon[int(m.group(1))] = f

for t in spec["topics"]:
    n = t.get("n")
    tag = f"topic {n}"
    for field in ("n", "slug", "title", "question", "baseline"):
        if not t.get(field):
            errs.append(f"{tag}: missing `{field}`")
    if n in seen_n:
        errs.append(f"{tag}: duplicate n")
    seen_n.add(n)
    if t.get("slug") in seen_slug:
        errs.append(f"{tag}: duplicate slug {t.get('slug')!r}")
    seen_slug.add(t.get("slug"))

    papers = t.get("papers") or {}
    for role in ("origin", "sota"):
        v = str(papers.get(role, "") or "")
        if not v or v.strip() == "TBD":
            warns.append(f"{tag}: `{role}` paper not pinned")
        elif not DOI.search(v) and "arXiv" not in v:
            errs.append(f"{tag}: `{role}` has no DOI or arXiv id — an anchor must be resolvable")

    con = t.get("concepts") or []
    if len(con) != 3:
        errs.append(f"{tag}: {len(con)} concepts; the cap is exactly 3")

    if n in recon:
        for field in ("finding", "scale"):
            if not t.get(field):
                errs.append(f"{tag}: recon report exists ({recon[n].name}) but no `{field}` in topics.yml")
    else:
        if t.get("finding"):
            errs.append(f"{tag}: claims a `finding` with no recon report in notes/topics/")
        warns.append(f"{tag}: no recon report — cannot be authored yet")

missing = sorted(set(recon) - seen_n)
if missing:
    errs.append(f"recon reports with no topics.yml entry: {missing}")

# ── docs/project.md names the sessions its questions rely on. Those were once week
# numbers and broke silently when a topic was retired, so they are dates now and this
# checks every one of them is a real session date.
proj = ROOT / "docs" / "project.md"
if proj.exists():
    import datetime as _dt
    first = _dt.date(2026, 9, 1)
    valid = {f"{first + _dt.timedelta(weeks=i):%b %-d}"
             for i in range(1, len(spec["topics"]) + 1)}
    named = set(re.findall(r"\*Sessions?: ([^*]+)\*", proj.read_text()))
    for group in named:
        for d in [x.strip().rstrip(".") for x in group.split(",")]:
            if d and d not in valid:
                errs.append(f"docs/project.md names session '{d}', which is not a "
                            f"session date in topics.yml")
    if re.search(r"\*Weeks? \d", proj.read_text()):
        errs.append("docs/project.md still uses week NUMBERS; they break when a topic "
                    "is retired. Use session dates.")

print(f"── topics.yml: {len(spec['topics'])} topics, {len(recon)} with recon reports")
for e in errs:  print(f"   ERROR  {e}")
for w in warns: print(f"   warn   {w}")
print(f"   -> {len(errs)} errors, {len(warns)} warnings")
sys.exit(1 if errs else 0)
