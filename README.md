# EPS 207 · Laboratory in Observational Seismology

Seismology is the study of earthquakes and the seismic waves that travel through the Earth. This
course covers the main tasks of observational seismology — magnitude estimation, phase picking,
event location, focal mechanism determination, ground-motion prediction, and seismic tomography —
together with the machine learning methods now used for each.

## Time and place

- **Tuesdays 9:00–10:59 am, 325 McCone Hall**
- Instructor: Weiqiang Zhu · 285 McCone Hall · `zhuwq@berkeley.edu`

## The final project

The course is built on **one field**: The Geysers, a geothermal field in the Coast Ranges about
120 km north of San Francisco. It is the largest geothermal electricity-generating complex in
operation, and it produces tens of thousands of earthquakes a year because the water injected to
sustain the reservoir makes existing fractures easier to slip on. Every session draws its data from
that field, and the final project answers a question about it.

The design is deliberate. A tour of famous earthquakes would spend each week introducing a new
dataset and never get past the introduction. Here the datasets are introduced once, in the **project
dataset** notebook, and every week afterwards returns to them. By November they are familiar enough
that a new method can be judged on what it adds rather than on whether it ran at all.

**The project is 60 per cent of the grade, and you choose the question.** Nine are offered: how long
the reservoir takes to respond to injection and whether the lag depends on depth; why a 1984 study of
this same field found no correlation where one is now taken for granted; whether the *b*-value tracks
injection at a resolution the data actually support; what a machine-learning catalogue adds and what
it gets wrong; and others. Each names the sessions whose methods apply and the baseline you must beat
or show to be sufficient. Bring your own question if you have one and it can be answered with these
data in the time available. The brief is in [`docs/project.md`](https://github.com/AI4EPS/EPS207_Observational_Seismology/blob/main/docs/project.md).

**How the schedule relates to it.** Each row below is one session. The *Seismology* column names a
measurement task from this field and the *Machine learning* column the method applied to it; the
pairing is the point, since every method is taught on data you already hold. When a project question
says *Sessions: Sep 8, Sep 15*, those are the rows to return to. Presentations are on 8 December, in
RRR week.

## Schedule

| Date | Seismology | Machine learning | Open |
|---|---|---|---|
| Sep 1 | [Introduction](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/00_introduction.html) | | |
| Sep 8 | Magnitude calibration | Regression & uncertainty |  |
| Sep 15 | Where aftershocks occur | Bias–variance, boosting, CV |  |
| Sep 22 | Fault structure from seismicity | Clustering, mixture models, EM |  |
| Sep 29 | Earthquake / quarry-blast discrimination | NN: classification |  |
| Oct 6 | Phase picking | NN: segmentation |  |
| Oct 13 | Event detection on DAS | NN: object detection |  |
| Oct 20 | Denoising | NN: Denoising |  |
| Oct 27 | Ground-motion prediction | Transformers |  |
| Nov 3 | Template matching | Similarity & embeddings |  |
| Nov 10 | Waveform generation | VAE and Diffusion |  |
| Nov 17 | Focal mechanism & moment tensor | Inversion I — linear |  |
| Nov 24 | Location & relocation | Inversion II — non-linear |  |
| Dec 1 | Tomography | Inversion III — fields |  |
| Dec 8 | Final project presentations | | |

## Assessment

| | |
|---|---|
| Homework | 40% |
| Final project | 60% |

## What you need

**Python**: `numpy`, `pandas`, `scikit-learn`, `pytorch`, plus `matplotlib`, `obspy` and `scipy`.

## Previous offerings

Fall 2023 is preserved on the [`fall2023`](https://github.com/AI4EPS/EPS207_Observational_Seismology/tree/fall2023)
branch, including the lecture slides.
