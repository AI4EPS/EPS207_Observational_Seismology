# EPS 207 · Laboratory in Observational Seismology

This course covers the main tasks of observational seismology — magnitude estimation, phase
picking, event location, focal mechanism determination, ground-motion prediction, and seismic
tomography — together with the machine learning methods now used for each.

## Time and place

- **Tuesdays 9:00–10:59 am, 325 McCone Hall**
- Instructor: Weiqiang Zhu · 285 McCone Hall · `zhuwq@berkeley.edu`

## Schedule

| Date | Seismology | Machine learning |
|---|---|---|
| Sep 1 | [Introduction](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/00_introduction.html) | |
| Sep 8 | [Injection-induced seismicity and the b-value](https://colab.research.google.com/github/AI4EPS/EPS207_Observational_Seismology/blob/main/docs/notebooks/01_regression_uncertainty.ipynb) | Regression & uncertainty |
| Sep 15 | Where aftershocks occur | Classification & bias-variance |
| Sep 22 | Fault structure from seismicity | Clustering |
| Sep 29 | Focal mechanism & moment tensor | Inversion I — linear |
| Oct 6 | Location & relocation | Inversion II — non-linear |
| Oct 13 | Tomography | Inversion III — adjoint |
| Oct 20 | Template matching | Similarity search |
| Oct 27 | Source discrimination | NN: classification |
| Nov 3 | Phase picking | NN: segmentation |
| Nov 10 | Denoising | NN: denoising |
| Nov 17 | Event detection on DAS | NN: object detection |
| Nov 24 | Ground-motion prediction | NN: transformer |
| Dec 1 | Waveform generation | NN: VAE and Diffusion |
| Dec 8 | Final project presentations | |

Nothing needs installing. Each week's notebook opens in Colab from the link in its title, and
installs whatever it needs when it runs.

## What this course is, and how it relates to EPS 130

EPS 130 develops the principal quantities of seismology from the physics: how a wave propagates,
how an earthquake is located, what a magnitude measures. **EPS 207 takes those same quantities and
applies them to real data**, where nothing is clean and every measurement carries a convention and
an uncertainty.

This semester we practise on **The Geysers**, a geothermal field in the Coast Ranges of northern
California. It has been producing electricity since 1960, water has been injected into it for
decades to sustain the reservoir, and it is among the most seismically active places in the state as
a result — tens of thousands of catalogued earthquakes a year, recorded by a dense local network and
archived, along with the rest of northern California, at the Northern California Earthquake Data
Center here at Berkeley.

Two threads run through the schedule.

**The seismology.** The *Seismology* column names a measurement task — magnitude, picking, location,
focal mechanisms, tomography — carried out on real records from this field rather than on a textbook
example.

**The machine learning.** The *Machine learning* column begins where EPS 130 leaves off: linear
regression and uncertainty, then classical statistical learning and inverse theory, and only then
the neural methods. Each is introduced in the context of the seismological problem it is being
asked to solve.

The first session is an overview of the seismic data available for The Geysers — the catalogue, the
waveforms, the arrival times, the injection and production record — so that from the outset you can
begin thinking about which problem you would like to look into. That material is the **project
dataset** notebook, and every week afterwards returns to it. Open it in
[Colab](https://colab.research.google.com/github/AI4EPS/EPS207_Observational_Seismology/blob/main/docs/notebooks/project_geysers_data.ipynb): a complete run needs about two gigabytes of memory, and a DataHub
session is given one. It is also on [DataHub](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS207_Observational_Seismology&urlpath=lab%2Ftree%2FEPS207_Observational_Seismology%2Fdocs%2Fnotebooks%2Fproject_geysers_data.ipynb&branch=main), which keeps your edits
between visits.

## The final project

The project is **60 per cent of the grade**; homework is the other 40. You pick the question. The
brief sets out nine, none of them with a settled answer: how long the reservoir takes to respond to
injection, whether the *b*-value tracks it, what a machine-learning catalogue adds. Each lists the
sessions whose methods apply and a baseline to compare against. Propose your own if you would
rather.

## Previous offerings

Fall 2023 is preserved on the [`fall2023`](https://github.com/AI4EPS/EPS207_Observational_Seismology/tree/fall2023)
branch, including the lecture slides.
