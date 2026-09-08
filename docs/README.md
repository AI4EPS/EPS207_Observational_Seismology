# EPS 207 · Laboratory in Observational Seismology

Seismology is the study of earthquakes and the seismic waves that travel through the Earth. This
course covers the main tasks of observational seismology — magnitude estimation, phase picking,
event location, focal mechanism determination, ground-motion prediction, and seismic tomography —
together with the machine learning methods now used for each.

## Time and place

- **Tuesdays 9:00–10:59 am, 325 McCone Hall**
- Instructor: Weiqiang Zhu · 285 McCone Hall · `zhuwq@berkeley.edu`

## What this course is, and how it relates to EPS 130

EPS 130 develops the principal quantities of seismology from the physics: how a seismic wave
propagates, how an earthquake is located, what a magnitude measures, how the geometry of slip on a
fault is recovered, and how the statistics of many earthquakes behave. **EPS 207 takes those same
quantities and applies them to real data**, where nothing is clean and every measurement carries a
convention and an uncertainty.

This semester we practise on **The Geysers**, a geothermal field in the Coast Ranges of northern
California. It has been producing electricity since 1960, water has been injected into it for
decades to sustain the reservoir, and it is among the most seismically active places in the state as
a result — tens of thousands of catalogued earthquakes a year, recorded by a dense local network and
archived, along with the rest of northern California, at the Northern California Earthquake Data
Center here at Berkeley. A long and well documented history of production, a great deal of
seismicity, and a network close enough to see it: that combination is what makes the field a
practical target for a semester of work.

Two threads run through the schedule below.

**The seismology.** The *Seismology* column names a measurement task — magnitude, picking, location,
focal mechanisms, tomography — carried out on real records from this field rather than on a textbook
example.

**The machine learning.** The *Machine learning* column follows a deliberate progression, and it
begins where EPS 130 leaves off: linear regression and uncertainty, then classical statistical
learning and inverse theory, and only then the neural methods, from classification and segmentation
through to transformers and generative models. Each is introduced in the context of the
seismological problem it is being asked to solve.

The first session is an overview of the seismic data available for The Geysers — the catalogue, the
waveforms, the arrival times, the focal mechanisms, and the injection and production record — so
that from the outset you can begin thinking about which problem you would like to look into. That
material is the **project dataset** notebook, and every week afterwards returns to it. Open it in
[Colab](https://colab.research.google.com/github/AI4EPS/EPS207_Observational_Seismology/blob/main/docs/notebooks/project_geysers_data.ipynb), which is where it belongs: a complete run needs about two gigabytes
of memory, and a DataHub session is given one. It is also on
[DataHub](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS207_Observational_Seismology&urlpath=lab%2Ftree%2FEPS207_Observational_Seismology%2Fdocs%2Fnotebooks%2Fproject_geysers_data.ipynb&branch=main), which is the easier home for the weekly notebooks below.

## The final project

The final project is **60 per cent of the grade**, and you choose the question. Nine are offered:
how long the reservoir takes to respond to injection and whether the lag depends on depth; why a
1984 study of this same field found no correlation where one is now taken for granted; whether the
*b*-value tracks injection at a resolution the data actually support; what a machine-learning
catalogue adds and what it gets wrong; and others. Each names the sessions whose methods apply and
the baseline you must beat or show to be sufficient, so a question that says *Sessions: Sep 8,
Sep 15* points at rows in the schedule below. Bring your own question if you have one and it can be
answered with these data in the time available.

The brief is in [`docs/project.md`](https://github.com/AI4EPS/EPS207_Observational_Seismology/blob/main/docs/project.md). Presentations are on
8 December, in RRR week.

## Schedule

| Date | Seismology | Machine learning | Open |
|---|---|---|---|
| Sep 1 | [Introduction](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/00_introduction.html) | | |
| Sep 8 | Magnitude calibration | Regression & uncertainty | [DataHub](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS207_Observational_Seismology&urlpath=lab%2Ftree%2FEPS207_Observational_Seismology%2Fdocs%2Fnotebooks%2F01_regression_uncertainty.ipynb&branch=main) · [Colab](https://colab.research.google.com/github/AI4EPS/EPS207_Observational_Seismology/blob/main/docs/notebooks/01_regression_uncertainty.ipynb) |
| Sep 15 | Where aftershocks occur | Classification & bias-variance |  |
| Sep 22 | Fault structure from seismicity | Clustering |  |
| Sep 29 | Focal mechanism & moment tensor | Inversion I — linear |  |
| Oct 6 | Location & relocation | Inversion II — non-linear |  |
| Oct 13 | Tomography | Inversion III — adjoint |  |
| Oct 20 | Template matching | Similarity search |  |
| Oct 27 | Source discrimination | NN: classification |  |
| Nov 3 | Phase picking | NN: segmentation |  |
| Nov 10 | Denoising | NN: denoising |  |
| Nov 17 | Event detection on DAS | NN: object detection |  |
| Nov 24 | Ground-motion prediction | NN: transformer |  |
| Dec 1 | Waveform generation | NN: VAE and Diffusion |  |
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
