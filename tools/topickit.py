#!/usr/bin/env python
"""Shared machinery for building EPS 207 topic notebooks.

ONE copy of the cell helpers, so fourteen build scripts cannot drift apart. The sibling course
learned this the hard way: two hand-written copies of the same job diverged immediately.

A topic script does:

    from topickit import md, run, ckpt, yrs, write
    md("## 1 · ...")
    run("...")
    write("01_regression_uncertainty")

`write` emits BOTH notebooks from the one cell list: the solution, and the student version derived
from it by blanking exercise answers. The student notebook is never authored by hand.
"""
import pathlib
import re

import nbformat as nbf

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "notebooks"
C = []


def md(s):
    """A markdown cell."""
    C.append(("md", s, None))


def run(s):
    """A code cell that is simply run."""
    C.append(("code", s, "run"))


def ckpt(n, body):
    """A cell that rebuilds the state a section needs, for anyone who fell behind.

    Mandatory at section boundaries: a student who breaks at minute 40 otherwise has no way to
    rejoin, and an exercise that defines a variable later cells need will strand them.
    """
    C.append(("code",
              f"# ── Checkpoint {n} ── run this if you are behind or something "
              f"broke ──\n" + body, "run"))


def yrs(solution, prompt):
    """An exercise: `prompt` is shown to everyone, `solution` only in the solution notebook.

    The student notebook gets a blank cell here. check_notebook.py counts exercises by these blanked
    cells rather than by marker text, because a label can drift from reality and a deleted cell cannot.
    """
    C.append(("code", solution, ("y", prompt)))


def _make(cells, student):
    nb = nbf.v4.new_notebook()
    nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python",
                                       "name": "python3"},
                        "language_info": {"name": "python"}})
    for kind, src, tag in cells:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        elif isinstance(tag, tuple):
            nb.cells.append(nbf.v4.new_markdown_cell(tag[1]))
            nb.cells.append(nbf.v4.new_code_cell("# your code here\n" if student else src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))
    return nb


def write(slug, cells=None):
    """Emit both notebooks and report what was built."""
    cells = C if cells is None else cells
    if not cells:
        raise RuntimeError("no cells to write - did the topic script append anything?")
    n_ex = sum(1 for _, _, t in cells if isinstance(t, tuple))
    if n_ex < 2:
        raise RuntimeError(f"only {n_ex} exercise(s); participation is 50% of the grade")
    # Compile every code cell before writing. A stray backslash-n inside a non-raw
    # triple-quoted string becomes a REAL newline and silently breaks an f-string; without
    # this the build looks fine and nbconvert only fails minutes later, three cells deep.
    for i, (kind, body, tag) in enumerate(cells):
        if kind != "code":
            continue
        try:
            compile(body, f"<cell {i}>", "exec")
        except SyntaxError as e:
            raise SyntaxError(f"cell {i} does not compile: {e.msg} (line {e.lineno})") from None

    # Markdown escape damage. Every one of these has shipped: a real tab from an unescaped
    # \t in "\\top", a literal \u2014 written into a RAW block, and doubled backslashes from a
    # non-raw block edited as if it were raw. Catch them here, not in the rendered page.
    for i, (kind, body, tag) in enumerate(cells):
        texts = [body] if kind == "md" else ([tag[1]] if isinstance(tag, tuple) else [])
        for txt in texts:
            if "\t" in txt:
                raise ValueError(f"cell {i}: literal TAB in markdown - almost certainly a LaTeX "
                                 f"command like \\top or \\tau in a NON-raw string. Use md(r\"\"\"...).")
            m = re.search(r"\\u[0-9a-fA-F]{4}", txt)
            if m:
                raise ValueError(f"cell {i}: unrendered escape {m.group(0)!r} in markdown - a "
                                 f"\\uXXXX written into a RAW block. Use the character itself.")
            m = re.search(r"\\\\[a-zA-Z]{2,}", txt)
            if m:
                raise ValueError(f"cell {i}: doubled backslash before {m.group(0)!r} - this block "
                                 f"is raw, so write single backslashes.")

    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(_make(cells, False), OUT / f"{slug}_solution.ipynb")
    nbf.write(_make(cells, True), OUT / f"{slug}.ipynb")
    print(f"wrote {slug} — {len(cells)} cells, {n_ex} exercises")
