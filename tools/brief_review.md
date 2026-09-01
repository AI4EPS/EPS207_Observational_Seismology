# Reviewer brief — topic {N}

> **Trash/ — DO NOT READ.** Abandoned work lives there. Do not open it for reference, for style, or
> to see how something was done before. Every lesson worth keeping was extracted into this file,
> `notes/defects.yml` and the recon reports before anything was moved; the artefacts themselves carry
> the mistakes as well as the lessons.


**You did not build this notebook and you must not fix it.** Your job is to find what is wrong and
say so precisely enough that the builder can act. A reviewer who edits becomes the builder, and then
nobody is reviewing.

Three self-review rounds on the sibling course produced a notebook that graded itself PASS on every
standard, which an independent reviewer then returned with two blocking defects. That is why you exist.

## What to read, in this order

1. `notes/topics/{NN}_*.md` — **the recon report. This is the source of truth for every number.**
   If the notebook states a number the report does not support, that is a finding.
2. `topics.yml`, entry `n: {N}` — the question, the two papers, the three concepts, the baseline.
3. `TEMPLATE.md` — the arc and the rules.
4. `docs/notebooks/{NN}_*_solution.ipynb` — **read it end to end, as a student would, before judging any
   part of it.** Not cell by cell against a checklist. Most of the defects found so far were only
   visible in sequence: a variable defined in a cell that a student will have rewritten, a section that
   asserts what the previous section was supposed to have shown.

Run `python3 tools/check_notebook.py {N}` first and **assume it passes** — it is fitted to defects
already found, so a clean report tells you only that the known bugs are absent. Everything it catches
is beneath your attention; everything it misses is your job.

## First: read it as a reader

Before any verification, **read the notebook front to back once, at reading speed, as a student who
has not seen it.** Note every place you stopped — a sentence you had to re-read, a figure you could
not decode, a marker that told you nothing, a paragraph that was throat-clearing, a claim that
sounded like an aphorism.

This is the highest-yield thing in this document. Every defect the instructor caught in topic 1 was
findable this way and by no check: a framing paragraph with a non-sequitur bolted on, an unrendered
`\u2014` in the text, sixteen `BUILD` markers on three exercises, four amplitudes drawn as lines on
an axis those lines set, and an invented aphorism as the lead takeaway. The automated checks passed
all of them.

**State the version you read** — the cell count and the date — because a review is only valid for the
draft it saw, and drafts here change fast.

## What to hunt for

**Claims the output does not support.** The single most common defect. For every numeric or comparative
claim in prose, find the cell output that establishes it. "Within about one percent" survived in topic 1
for hours as evidence for a measurement convention that three different conventions would all have
satisfied. An agreement every candidate explanation predicts is not evidence for one of them.

**Silent failure paths.** What does each cell print if its network call returns nothing, its filter
matches nothing, its fit does not converge? A cell that prints `nan` under a confident heading is worse
than one that raises. Trace at least every `try/except` and every cell that indexes or aggregates.

**Papers that decorate.** Both anchors must do work *after* the header — supply a number to compare
against, a definition, a method being reproduced. A paper named once in the introduction is a citation,
not an anchor.

**Sections that assert instead of showing.** Each `##` section should compute something. Prose-only
sections are legitimate only for threats, what-remains-open, and takeaways.

**The baseline actually run.** `topics.yml` names a baseline. It must be run on the same data and the
same metric, with the comparison visible. A stated-but-unrun baseline is the defect that makes a method
look necessary when it is not.

**Overclaim in the uncertainty.** Is the quoted interval the one that matters? Topic 1 quoted a
bootstrap σ of 0.0028 while sitting 11.5 published standard errors from the accepted value. Topic 3's
random-split R² has a seed sd of 0.0008 while the quantity that actually moves has sd 0.018. A tight
interval around the wrong thing is the failure mode, and it recurs in every session.

**Circularity.** Is any predictor derived from the target? Is any "validation" source downstream of the
thing being validated? Both have already been found in this course.

**Does the notebook say what its data IS?** Region, time span, selection. If you have to load the
file to find out, that is a blocking finding - and check whether any comparison with a published
result survives knowing it.

**Forward references that are never honoured.** Grep for "section N", "later", "at the end".

**Comparisons between incomparable things.** For every pair of numbers the notebook sets against
each other, check they were computed on the same sample, at the same n, under the same conditions.
This is the single highest-yield check in this document: topic 1 shipped a bootstrap at n=40,000
compared against an analytic standard error at n=641,374, and the resulting factor of 4 was written
up as evidence of clustering when it was sqrt(16) and nothing else.

**Named mechanisms with no test.** Every "because", "so", "which is why" that attaches a cause to a
number: find the cell that could have refuted it. A mechanism that no cell tested is a finding even
when the mechanism is real.

**Seismology takeaways with no source.** Every one must trace to a textbook or a paper the builder
opened. Ask for the source and the page. A takeaway that reads as a memorable aphorism is the most
likely to be invented — the sentence that sounds most quotable is the one least likely to have a
citation behind it.

**Concepts introduced more than once.** Grep the notebook for each core idea. If it appears before
the argument needs it, the structure is wrong and the earlier appearance should move.

**Editorialising about the method.** Paragraphs that step back to comment on the reasoning — "do
not conclude", "note what this could not do", "conclude something narrower". This is the most
reliable machine-written tell in the material and it is always cuttable.

**Conclusions the sources do not support.** Where the notebook states what a catalogue, network or
community does, check the source says it. An inference from the notebook's own measurement, written
in the register of settled fact, is the defect to hunt.

**Domain claims with no source.** Statements about what a catalogue does, what a convention is, what
a term means in the literature. Ask for the source. "It sounded right" is how topic 1 shipped a claim
that operational practice does not apply site corrections, when the operative paper publishes them.

**Borrowed vocabulary.** Where the notebook maps its own quantity onto a term of art from another
field, check the term's actual definition and whether the mapping survives it.

**Room feasibility.** Count the live network calls and multiply by eight students in the same two
minutes. Count the minutes of compute. A session is 100 minutes and the notebook is worked.

## Verdict format

Report findings in three tiers, most severe first. For each: file, cell index, what is wrong, and the
concrete failure — inputs or state that produce the wrong output.

- **BLOCKING** — a wrong number, an unsupported claim, a silent failure, a missing baseline, an
  unrunnable cell. The notebook does not ship.
- **SHOULD FIX** — a defect that will cost class time or teach a bad habit.
- **NOTE** — a judgement call you would have made differently. The builder may decline these.

**Say "no blocking defects" only if you looked for each of the above and found none.** A review that
finds nothing is itself a finding, and should say which of these you checked.
