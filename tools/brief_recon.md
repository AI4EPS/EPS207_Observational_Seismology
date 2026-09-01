# RECON — one EPS 207 topic

> **Trash/ — DO NOT READ.** Abandoned work lives there. Do not open it for reference, for style, or
> to see how something was done before. Every lesson worth keeping was extracted into this file,
> `notes/defects.yml` and the recon reports before anything was moved; the artefacts themselves carry
> the mistakes as well as the lessons.


You are answering one question, and only this question:

> **Is there a session here, and what are the real numbers?**

This is throwaway code. **No notebook, no teaching material, no prose beyond the report.** Nobody
will read your script. What gets read is your Finding Report.

## Why this stage exists

Published research does not fit in two hours. The paper behind your topic used more data, more
compute and more time than a class session has. Your job is to **shrink it until it fits and then
measure what the shrinking cost** — and to find out early if it cannot be done at all, before
anyone authors a session around it.

Evidence for why: on 2026-08-31 topic 1 was authored three times. The first version claimed a
"68σ disagreement with the published attenuation relation" that turned out to be an artifact of the
magnitude range; the second asserted "site effects are most of the error" from an invalid variance
comparison; the third had to be rebuilt again when the predictor turned out to be derived from the
target. **Every one of those was a claim asserted without being computed.** Your report exists so
that does not happen thirteen more times.

## Read first

- `topics.yml`, your topic's entry — question, the two papers, the three ML concepts, the baseline,
  the data hint, and any stated risk.
- `notes/ml-concepts.md` — the concept budget and the four threads running through the course.
- `notes/topics/01_magnitude.md` — the one worked example of what a good topic looks like.

## Hard constraints

- **Students have NO credentials.** `gs://quakeflow_catalog` and `gs://quakeflow_dataset` return
  403 anonymously. You may read them to build a subset, but the student-facing route must be
  public: USGS/IRIS/NCEDC FDSN, HuggingFace, or a GitHub release asset.
- **Colab and Berkeley DataHub only.** DataHub is CPU. Anything needing a GPU must say so.
- **The session is 100 teaching minutes, live in the room.** Anything that takes longer than about
  20 minutes of compute has to be pre-computed, and you should say what to pre-compute.

## What to do

1. **Get the data and look at it.** Print shapes, spans, null counts, class balance. Every claim in
   `topics.yml` about the data is a hypothesis until you have run it — including mine. Report
   anything that contradicts it.
2. **Find the origin paper** if `papers.origin` is TBD. One paper that introduced the problem or the
   method, verified against OpenAlex or Crossref with a real DOI. Two papers per topic, no more.
3. **Run the baseline named in the topic.** Get its number.
4. **Run the method at a deliberately small scale, then increase** until the finding appears.
   Powers of ten are fine. Stop at the smallest scale where it holds.
5. **Put an interval on it.** Bootstrap or repeat with different seeds. A gap that does not clear
   its own uncertainty is not a finding yet — say so, and say at what scale it would be.
6. **Time everything**, per step, on CPU and on GPU if relevant.
7. **Find the data budget.** What is the SMALLEST dataset that still shows the finding? Test it —
   subsample and re-measure at several sizes, reporting the numbers at each. Topic 1 went from
   734,277 rows to 20,000 with every lesson intact and better figures; topic 4 went from 289 MB of
   public downloads to 0.43 MB. **Target under ~50 MB for the student-facing asset**, and say
   plainly if that is impossible. A dataset a student cannot plot in full is usually a design
   failure, not a necessity.

## Report — the deliverable

    # Topic N — Finding Report
    ## VERDICT
    FINDING | NO FINDING | BLOCKED     + one sentence
    ## Papers
    origin + state of the art, with DOIs you verified
    ## Data
    what loaded, from where, how big, how long, what is wrong with it,
    and THE PUBLIC ROUTE a student would use
    ## Baseline
    the number, and how you computed it
    ## The finding
    the comparison, with an interval. The smallest scale at which it holds.
    ## preserved / lost
    what survives the shrink to class scale; what does NOT (be specific and honest —
    this is the most valuable line in the report)
    ## Student data budget
    smallest dataset that still shows the finding, with numbers at each size ·
    exactly what a student downloads, from what public URL, in MB ·
    what WE must pre-compute and mirror as a release asset rather than fetch live
    ## Threats
    circularity (is a predictor derived from the target?) · confounding (are two
    explanations inseparable in this sample?) · overclaiming (with large n, is sigma
    meaningless?) · sample sensitivity (does it survive a defensible subset change?) ·
    IS THE NULL THE RIGHT NULL? (topic 4 nearly reported a result that was an artifact
    of comparing against uniform-random when every mapped fault in the region shares an
    orientation; the correct spatial-permutation null killed it)
    ## What broke
    every failure with its error; version pins that mattered; anything needing a GPU
    ## Runtimes
    per step, so a session can be budgeted

## Rules

- **Every number you report is one you computed.** Never a number from a paper, never one you
  remember. If you quote a paper's figure, label it as the paper's.
- **BLOCKED and NO FINDING are good outcomes.** They save a week of authoring. Do not manufacture a
  finding, and do not soften a blocker. If the data does not exist, say the data does not exist.
- Seed everything. Work in a scratch directory. Do not touch the repo, do not commit, do not push.
- **Two papers, no more** — one origin, one state of the art, both verified against OpenAlex or
  Crossref. A third may be cited as a fact or a number to compare against, never as an anchor. And
  check the paper you propose can actually DO WORK in a notebook: one that would only be mentioned
  in an introduction is decoration, and will be rejected.
- **Verify every DOI you report.** A recon on 2026-08-31 found three candidate papers in its brief
  that do not exist, and one author name that was wrong.
