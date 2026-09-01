# EPS 207 · Laboratory in Observational Seismology

Seismology is the study of earthquakes and the seismic waves that travel through the Earth. This
course covers the main tasks of observational seismology — magnitude estimation, phase picking,
event location, focal mechanism determination, ground-motion prediction, and seismic tomography —
together with the machine learning methods now used for each.

## Time and place

- **Tuesdays 9:00–10:59 am, 325 McCone Hall**, 1 September – 1 December 2026
- Final project presentations **Tuesday 8 December** (RRR week)
- Instructor: Weiqiang Zhu · 285 McCone Hall · `zhuwq@berkeley.edu`

## Schedule

| Date | Seismology | Machine learning |
|---|---|---|
| Sep 1 | [Introduction](https://ai4eps.github.io/EPS207_Observational_Seismology/lectures/00_introduction.html) | |
| Sep 8 | Magnitude calibration | Regression & uncertainty |
| Sep 15 | Where aftershocks occur | Bias–variance, boosting, CV |
| Sep 22 | Fault structure from seismicity | Clustering, mixture models, EM |
| Sep 29 | Earthquake / quarry-blast discrimination | NN: classification |
| Oct 6 | Phase picking | NN: segmentation |
| Oct 13 | Event detection on DAS | NN: object detection |
| Oct 20 | Denoising | NN: Denoising |
| Oct 27 | Ground-motion prediction | Transformers |
| Nov 3 | Template matching | Similarity & embeddings |
| Nov 10 | Waveform generation | VAE and Diffusion |
| Nov 17 | Focal mechanism & moment tensor | Inversion I — linear |
| Nov 24 | Location & relocation | Inversion II — non-linear |
| Dec 1 | Tomography | Inversion III — fields |
| Dec 8 | Final project presentations | |

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
