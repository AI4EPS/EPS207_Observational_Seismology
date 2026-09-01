# Builder brief — topic {N}

Author the solution notebook for topic {N}, derive the student notebook from it, and pass the gates.
**You will not review your own work.** An independent reviewer reads it afterwards from
`tools/brief_review.md`; your job is to make that review boring.

## Read first, in this order

1. **`notes/topics/{NN}_*.md` — the recon report.** This is where the session comes from. It contains
   the measured finding, the smallest data scale that still shows it, the baseline number, the threats,
   and what broke. **Every number you put in the notebook must be computed by the notebook or come from
   this report.** Nothing from memory, ever.
2. `topics.yml`, entry `n: {N}` — question, two papers, three concepts, stated baseline.
3. `TEMPLATE.md` in full. Do not summarise it into your own prompt; the rules are the point.
4. `tools/build_topic01.py` as the worked example of the build-script form, and
   `docs/notebooks/01_*_solution.ipynb` as the worked example of the output.

## What you produce

- `tools/build_topic{NN}.py` — a script that emits **both** notebooks from one source.
- `docs/notebooks/{NN}_{slug}_solution.ipynb` — executed, every cell with output.
- `docs/notebooks/{NN}_{slug}.ipynb` — derived by blanking exercise answers, never written by hand.

## Hard rules

- **Every number computed, never recalled.** A number in prose with no cell output behind it is a claim
  you must justify or delete.
- **Write the idiom the field uses**, not the shortest correct code — see `TEMPLATE.md`,
  "Write code the way the field writes it".
- **Name the dataset near the top** - region, span, selection - and list what differs between your
  fit and any published value you compare against. See `TEMPLATE.md`, "Name your data".
- **Grep your own forward references before shipping.** Every "section N will show" is a promise.
- **Public data only.** Both `gs://quakeflow_*` buckets are 403 anonymous. Students have no credentials.
- **Every live cell needs a retry and a stated fallback.** Count your network calls and multiply by
  eight students in the same two minutes. One FDSN endpoint will rate-limit; SCEDC returns
  `500 STP clients exceeded` under a single agent's query rate.
- **A cell must never print `nan` under a confident heading.** Assert your preconditions.
- **Check row uniqueness of every dataset you ship.** `len(df)`, `df.duplicated().sum()`,
  `df.duplicated(subset=natural_key).sum()`. Week 1 shipped 6.7% exact duplicates inherited from
  the source catalogue; they inflate `n` and shrink every interval by sqrt(n).
- **No markers.** A cell the student writes is a cell that is blank (`# your code here`).
  A marker is a label that can drift from the thing it labels; the blank cannot.
- **Both papers must do work after the header.** A paper named once in the intro is decoration.
- **Run the stated baseline**, on the same data and metric, with the comparison visible.
- **Checkpoint cells at section boundaries**, so a student who breaks at minute 40 can rejoin.
- No local paths. Set a seed. ~20–30 code cells, a figure every ~15 minutes, 3–4 exercises.

## Claims — where every notebook so far has failed

The checker verifies that a *number* in prose appears in an output. It cannot verify that the
*sentence around the number* is true, and that is where every defect in topic 1 lived.

- **A mechanism you name must be produced by a cell that could have come out otherwise.** Topic 1
  claimed a bootstrap/analytic gap was caused by clustering. It was not: the bootstrap resampled
  40,000 rows and was compared against the analytic value at 641,374, and sqrt(641374/40000) = 4.00,
  exactly the "effect". A plausible story was attached to a number that only reflected sample size.
  If you cannot design the cell that would refute your explanation, delete the explanation.
- **Two numbers you compare must be computed under the same conditions, and the cell must say so.**
  Print the n, the sample, the seed for both sides. Most of the false conclusions in this course so
  far were comparisons between things that were never comparable.
- **Every domain claim needs a source you opened this session.** Not a number — a claim: what a
  catalogue does, what a convention is, what a term means in the literature. Topic 1 asserted
  "routine catalogues do not correct for site" when Uhrhammer et al. (2011) publish station
  adjustments for 1185 California channels. Nobody checked, because it sounded right.
- **Borrowed vocabulary must be checked against its own definition.** Topic 1 mapped a
  between-station variance onto the ground-motion term phi_S2S while never removing a between-event
  term, so the remaining scatter was not phi_ss and the mapping was half wrong.
- **Every seismology takeaway must cite a source you opened this session** — a textbook page or a
  paper's abstract/conclusions — and you deliver each one with that source named. Synthesised
  takeaways are the single most-rejected thing in this course. If you cannot source it, cut it.
- **Takeaways: follow `TEMPLATE.md` -> "Writing the takeaways" exactly.** The governing test is
  whether the sentence would still be true if the notebook had produced different numbers. If it
  would, it is background and belongs in the body. Deliver each item with the cell and the
  printed line it rests on.

## When you correct a claim, correct all of it

Grep the source for the *concept*, not for the wording you remember writing. A claim about aleatoric
and epistemic uncertainty was corrected in three places and survived in a fourth, because the fix was
applied from memory of what had been written rather than from a search of the file. Patch, then
search, then re-read the section.

## Gates — all three, in order

1. `python3 tools/build_topic{NN}.py`
2. `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/{NN}_*_solution.ipynb`
   — **run it unpiped.** A pipeline returns the *last* command's status, so piping into `tail` reports
   success while the notebook died halfway through. This has already cost two rebuild rounds.
3. `python3 tools/check_notebook.py {N}` → **0 errors.**

Passing the checker is the floor, not the bar. It is fitted to defects already found in topic 1, so a
clean report means "none of the known bugs," not "this is good."

## Before you hand it over

**Read the whole notebook end to end, as a student, in order.** Every defect that mattered so far was
invisible cell-by-cell and obvious in sequence. Then state, in your report:

- the finding the session turns on, and the cell that establishes it;
- every number in prose that is *not* computed, and why it is there;
- what a student sees if each network call fails;
- what you were unable to verify.

**You do not edit shared files.** If `topics.yml`, `TEMPLATE.md` or `notes/` need changing, say what and
why in your report; the orchestrator applies it.
