# How to build an EPS 207 session

Every rule here exists because something went wrong. Topic 1 was rebuilt four times on
2026-08-31; the reasons are recorded so the next thirteen cost less.

---

## The shape

**One notebook per topic.** You author the **solution**; the student version is derived from it by
deleting code, never written separately. You author a **generator script**, `tools/build_topicNN.py`
— the notebook is the artifact, the script is the source. A hand-edited notebook cannot be re-run
against corrected data, and every number here must be re-runnable.

    tools/build_topicNN.py                 the source
    docs/notebooks/NN_slug_solution.ipynb  executed; every number computed
    docs/notebooks/NN_slug.ipynb           derived by deleting the exercise answers

A session is **100 teaching minutes, live, worked not watched**: roughly 20–30 code cells, a figure
every ~15 minutes, 3–4 exercises. Fewer than 20 cells or fewer than 4 figures is a skeleton, and
`check_notebook.py` fails it.

## The arc

1. **The two papers, and what they are for.** One origin, one state of the art. **Both must do work
   later in the notebook** — a paper mentioned once in the header is decoration, and the checker
   now fails on it. Other papers may appear as *a number to compare against* or *a fact*; they are
   not anchors.
2. **Look at one raw record.** Before regressing 700,000 rows somebody else measured, measure one
   yourself and check it against the column. Topic 1 fetches a single waveform, simulates a
   Wood–Anderson, and recovers the catalogue amplitude to 1 %. This is the cell that lets the rest
   of the session rest on something checked rather than trusted.
3. **The data's defects.** Gaps, impossible values, class imbalance, duplicated coordinates. Real
   catalogues contain things that cannot be true; find them before fitting. Then let the students
   choose the cut — and test at the end how much the answer depended on it.
4. **The science, derived — not summarised.** This is the failure that took longest to see. The ML
   gets built up carefully while the seismology gets one hand-waving sentence. *"A big earthquake
   radiates at longer periods"* names a mechanism without explaining it; the corner-frequency
   argument, with f_c ∝ M₀^(−1/3) and a figure showing the instrument band crossing it, explains
   it — and predicts a slope ratio of 3 that the notebook then measures as 2.9.
5. **Build the model one term at a time.** Do not write the final model down. Fit, look at the
   residuals, see what structure is left, add the term that removes it. That is what fitting looks
   like, and it is not visible in a single `polyfit` call.
6. **The baseline, run.** `topics.yml` names it. Its number sits beside the method's.
7. **Uncertainty that belongs to this model.** See the spine in `notes/ml-concepts.md`.
8. **Validation against something independent** — a published value, another catalogue, held-out
   data. Never the training set, never a plot that merely looks right.
9. **Sensitivity to the choices made.** Threshold, window, cut, k, sample. Change one; show how far
   the answer moves. Topic 1's closing exercise moves c₂ by 100 standard errors.
10. **Threats to this conclusion.** The notebook argues against itself, in writing. Circularity ·
    confounding · overclaiming from significance · sample sensitivity · what was taken on trust.
11. **What remains open**, and which later topic picks it up.

## Working in the room

**No markers at all.** A cell the student writes is a cell that is blank, holding only
`# your code here`. Markers were tried at three levels, then one, and removed: a label drifts from
what it labels (16 `BUILD` markers ended up on 13 cells whose code was already written), while a
blank cell cannot lie about being blank.
An unmarked cell is simply run. Whether a cell is typed together or alone is a pacing decision you
make in the room; the notebook should not encode it, and three markers were noise.

The checker counts exercises by **blanked cells in the student notebook**, not by the marker text —
a label can drift from reality, a deleted cell cannot.

**Checkpoints are mandatory.** If an exercise defines a variable later cells need, a student who
fumbles it loses the session. Put a checkpoint at each section boundary:

```python
# ── Checkpoint 1 ── run this if you are behind or something broke ──
```

## Data

- **Students have no credentials.** `gs://quakeflow_*` returns 403 anonymously. Public routes only:
  FDSN (USGS/IRIS/NCEDC/SCEDC), HuggingFace, or a GitHub release asset on this repo.
- **Budget: under ~50 MB, and as far under as the finding allows.** Ask what the *smallest* dataset
  is that still shows the result, and test it — topic 1 went from 734,277 readings to 20,000 with
  every lesson intact and better plots, and topic 4 went from 289 MB of public downloads to 0.43 MB.
  A dataset a student cannot plot in full is usually a design failure, not a necessity.
- **Every live cell needs a fallback or a retry.** Eight people on one wifi at 9 am.
- **Pre-compute anything over ~20 minutes** and mirror it as a release asset; show the code that
  built it as an appendix cell rather than running it live.

## The notebook IS the lecture — derive the mathematics on the page

There are no slides. Whatever a student needs to follow the session has to be in the notebook, which
means **every mathematical result the session uses is derived, in numbered steps, with no step the
reader has to supply.** A graduate student who has not seen the result should be able to follow it
line by line without stopping.

The worked example is topic 1's ridge/MAP equivalence, which had been one sentence — "it is exactly
the MAP estimate under a Gaussian prior" — and is now seven steps:

1. **Write the forward model probabilistically.** $y = X\beta + \varepsilon$, $\varepsilon \sim
   N(0, \sigma^2 I)$, and from it the likelihood. Note that maximising it alone *is* least squares.
2. **Write down the prior**, explicitly, before seeing the data, and say what its width means in the
   units of the parameter.
3. **Bayes' theorem**, and why the denominator can be dropped.
4. **Take the negative log**, which turns the product into a sum of two quadratics.
5. **Rescale** by a positive constant to recover the damped objective, and read off
   $\lambda = \sigma^2/s^2$. This is the step that makes the equivalence *visible* rather than
   asserted.
6. **Minimise**: expand, differentiate, set to zero, get the estimator; then say why the Hessian is
   positive definite even when $X^\top X$ is singular.
7. **Explain the mechanism**, via the SVD and the filter factors $f_i = d_i^2/(d_i^2+\lambda)$.

Then, and this is the half that makes it a notebook rather than a textbook: **compute the quantity on
the session's own data immediately afterwards.** Topic 1 prints the four singular values of its
actual design matrix and the filter factors at five values of $\lambda$, so the reader sees the
best-resolved direction sitting at 1.0000 while the weakest falls to 0.2594. Abstract derivation,
then the same numbers from the data in front of them.

**The failure mode this replaces** is a sentence that names a result and moves on. "Ridge is the MAP
estimate under a Gaussian prior" is a claim a student can only take on trust; it teaches the name of
a fact rather than the fact.

## Writing the takeaways

Two lists, **Seismology** and **Machine learning**, four or five items each. They are the last thing
you write and the hardest, and every defect this course has shipped has been in them.

**Every seismology takeaway is grounded in a textbook or a paper you opened this session. Never
synthesised.**

This is the hardest rule here and the one most often broken, because a synthesised takeaway feels
like understanding. Six consecutive drafts of topic 1's list were rejected, and each time the failure
was the same: a sentence that sounded like the right lesson, with no source behind it. One of them —
"a magnitude is a property of a network, not of an earthquake" — was not only unsourced but false,
since M_w is derived from seismic moment and is a property of the source.

So: **open the source, and take the claim from it.** The anchor papers' abstracts and conclusions
first, then the standard references:

- **Bormann (ed.), *New Manual of Seismological Observatory Practice* (NMSOP-2)** — the IASPEI
  reference, and the authority on magnitude scales and their calibration.
- **Shearer, *Introduction to Seismology*** · **Aki & Richards, *Quantitative Seismology*** ·
  **Lay & Wallace, *Modern Global Seismology*** · **Stein & Wysession**.
- Weiqiang's Zotero library (2431 items). The MCP's search and fulltext need **Zotero desktop
  running** — without it only `list_libraries` works, from the SQLite file.

**If you cannot source it, you do not ship it.** Three sourced takeaways beat five with two invented.
Deliver each item with its source and the page or section it came from; a reviewer will check them.

**A takeaway is the durable science of the topic, evidenced by that source and by the notebook.**

Six drafts were rejected before that was clear, and the two failure modes are worth naming because
both feel productive while you are writing them:

- **Not your observations about this dataset.** "Our refit gives -1.3149" is a fact about one
  catalogue in one desert. A student who never touches Ridgecrest gets nothing from it.
- **Not a recitation of the anchor paper's internal comparisons.** "Their curve attenuates less
  rapidly than Richter's 1958 table" is that paper's business with a table nobody uses now. Read the
  abstract and conclusions — but extract the science, not the bookkeeping.

What survives is what a student would still need five years on: *magnitude is a property of a network
rather than of an earthquake; attenuation is regional so a curve does not transfer; site response is
first-order and cannot be averaged away; a fixed-period instrument saturates because the source
spectrum moves and it does not; a catalogue cannot be recalibrated without breaking the statistics
built on it.* Each of those is supported by the paper or by a cell, and none of them is about
Ridgecrest.

**The machine-learning list is different in kind: it is textbook, stated plainly.** Least squares is
maximum likelihood; a prediction interval carries an irreducible term; collinearity pins a
combination and not the parts; ridge is damped least squares and a Gaussian prior; standard errors
assume independent rows. The notebook's job is to make a student believe these, not to discover them.
Do not dress a textbook fact as a finding.

**Each item is one sentence and carries the evidence for itself.**

The test that settles it: *would this sentence still be true if the notebook had produced different
numbers?* If yes, it is background, not a takeaway — move it into the body.

> "A magnitude is fitted, not measured." — survives any data. Not a takeaway.
> "Refitting Richter's form on 640,926 readings gives a log R coefficient of -1.3149 where Hutton &
> Boore published -1.110." — a takeaway.

Then, each rule below because something shipped without it:

- **One sentence.** If it needs two, it is two takeaways or none.
- **It must be CHECKABLE against a printed number — not necessarily quote one.** The number sits in
  the cell output a few inches above; repeating it in the sentence a student is meant to remember
  usually makes the sentence worse. Quote a number only when the number *is* the memorable thing
  ("half the variance", "an eighth of the rows"). Never quote a fitted coefficient's value: "our
  refit does not reproduce the published curve" is the takeaway, and `-1.3149` is the evidence.
- **Say what was done and what came out**, not what it means in general.
- **Never name a mechanism no cell tested.** A bootstrap/formula gap was written up as clustering
  when it was sqrt(16); the method used could not have detected clustering at all.
- **Never use a term of art without opening its definition.** A between-station variance was
  labelled phi_ss; the actual definition requires removing the event term first, which no cell did.
- **Two numbers compared must come from the same conditions**, and the cell must print both.
- **Label the coverage of every interval** — 1 sigma or 95%. Two takeaways quoted both without
  saying which, so they read as contradicting each other.
- **Round to the precision the claim needs.** A takeaway is what a student carries out of the room,
  and `-1.3149` is not carryable while `about -1.31` makes the same point. Four significant figures
  belong in the cell output, not in a sentence someone is meant to remember. Ask of every number:
  *should a student remember this?* If the claim survives rounding, round it. If it does not survive
  rounding, it is a claim about the fourth digit and is probably not a takeaway at all. Prefer a word
  to a decimal where a word is true — "half the variance", "thirty times larger", "an eighth of the
  rows".
- **Attribute a number to what produced it.** A polyfit through your own synthetic curve is a check
  on the code, not a measurement of the Earth. Do not write "measured" for it.

**No "what remains open" section, and no "threats" section.** Both were tried and both silted up
with speculation and with the author's own unfinished business. A limitation worth stating belongs
next to the cell that exposes it, where a reader can check it against an output. The notebook ends
on the two takeaway lists.

**Then verify, item by item:** name the cell and quote the printed line it rests on. An item you
cannot trace that way is an opinion, and it goes.

## Write code the way the field writes it

A geophysics student should recognise the idiom, not just follow the logic. Fitting per-station
terms by subtracting each station's mean is correct, compact, and econometrics — nobody in this
field would write it. Building one sparse column per station and calling `lsqr` is the same fit, is
longer, and is how station terms are built in every tomography and relocation code the students will
read. Choose the second.

The same rule kills clever indexing, chained one-liners, and any trick whose correctness is not
obvious on the page. A notebook is read once, live, by someone typing along.

Libraries: **numpy, pandas, scikit-learn, pytorch**, plus **matplotlib** for figures, **obspy** for
waveforms and **scipy** where sparse linear algebra is the honest tool. Anything else needs a reason.

## Figures

- **Never compare quantities on an axis those quantities set.** Three amplitude conventions differing
  by 15% were drawn as horizontal lines on a trace whose y-range they defined: all four crowded
  against the frame and the difference — the entire point — was invisible. Put the *where* in one
  panel and the *how much* in another.
- **Look at the rendered image before shipping it.** Two defects survived a read of the code and died
  instantly on looking: a legend entry pointing at nothing, and a label clipped outside the axes.
- **Every legend entry must mark something visible**, and every annotation must sit on the feature it
  annotates.

## A section that exists because your data was bad is not a lesson

Topic 1 grew a section teaching students to check for duplicate rows. The duplicates were real — the
source catalogue files some readings twice — but the section existed because the shipped release
asset carried them. Clean the data where it is prepared, keep one sentence saying what was removed
and why, and give the section back to the science.

## Name your data, and say what it is not

The notebook must state, near the top, **what the dataset actually is** — region, time span, how it
was selected. Topic 1 shipped with the word "Ridgecrest" appearing exactly once in fifty-six cells,
inside a URL. A reviewer had to load the file to discover that every event lay in one 0.9-degree box.

This is not housekeeping. It decides whether the session's headline comparison is valid at all:
topic 1 reported its refit as **12.1 of Hutton & Boore's published standard errors** away from their
value, while having fitted one aftershock sequence against their whole-province result, without
station terms, over a different distance range and a different amplitude floor. H&B say on p. 2077
that they had the data to fit sub-regions and chose not to. The number was arithmetically right and
the comparison was meaningless.

**Before any comparison with a published value, list what differs between the two fits.** If the
list is not short, the comparison is not a measurement.

## Promises the notebook makes to itself

Every forward reference is a promise: "section 8 measures how much this matters", "you will test
this at the end". Grep for them before shipping and check each is kept. Topic 1 told students twice
that a later section would test the sensitivity of *their* chosen cut; that section varied the
magnitude range instead and never touched the cut.

The same applies backwards: a paragraph that says "you added a station term and a third of the
scatter became structure" must come *after* the cell that adds it, and must quote the number that
cell printed.

## Report the numbers that did not cooperate

A cell in topic 1 introduced itself as "the first honest comparison in this notebook" and then
listed the two coefficients that improved, omitting the one that moved further from the published
value. If a table has three rows and two of them help your argument, the paragraph names all three.

## Any table worth discussing is a view, not a message

The working schedule — ML method, seismology problem, anchor papers, baseline, data — was built and
revised across a dozen chat messages and existed nowhere else. When the conversation was compacted
it was gone, and had to be recovered by grepping a JSONL transcript. Every field in it was already
in `topics.yml`; only the view had been lost.

So: if a table is worth iterating on, it is worth generating. `tools/make_table.py` emits the full
planning view, `tools/make_site.py` emits the student-facing one. Both read `topics.yml`. Edit the
spine and re-run; never paste a table into a reply and treat that as the record.

## Rules that cost something to learn

- **Every number in prose is computed by a cell.** Never recalled, never lifted from a paper without
  recomputation. All three errors caught on 2026-08-31 were prose claims no cell produced, and every
  one survived a clean execution.
- **A claim that needs a paywalled paper gets a libproxy link, not a hedge.** `~/claude/tools/libproxy.py`.
- **Check the null is the right null.** Topic 4 nearly reported a result that was an artifact of
  comparing against a uniform-random baseline when every mapped fault in the region shares an
  orientation. The correct spatial-permutation null killed it.
- **Anchor patches on literal characters, never on `\uXXXX` escapes.** The build script holds
  real en- and em-dashes; a patch that writes `\u2014` in its search string silently matches
  nothing. Assert the anchor was found and let it fail loudly -- two rounds of "fixed" edits here
  never applied, and the notebook rebuilt cleanly each time with the old text still in it.
- **Never pipe the execute step into `tail`/`head`.** A shell pipeline returns the *last*
  command's status, so `nbconvert --execute ... | tail -3` reports success while the notebook
  died on a `NameError` halfway through. Run it unpiped, or `set -o pipefail`. The only reason
  this was caught is that `check_notebook.py` flags cells with no outputs -- the exit code lied.
- **Watch the escaping in the build script.** `\n` inside a non-raw triple-quoted string becomes a
  real newline in the generated cell and produces a syntax error. Use `\\n`, or avoid it. This bit
  three times in one evening.
- **Read the notebook end to end as a student before shipping.** The checker cannot see dead code
  that runs, prose that asserts what no cell shows, or a section that is thin. Reading it can.

## Before it ships

    python tools/check_notebook.py NN     # must be 0 errors
    # then execute on a fresh kernel and READ EVERY PRINTED NUMBER

Then the attack pass, then Weiqiang. One topic at a time.
