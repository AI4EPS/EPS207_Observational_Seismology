# EPS 207 · Laboratory in Observational Seismology

Observational seismology turns recorded ground motion into statements about earthquakes and the
Earth's interior. This laboratory course covers how that is done today — from magnitudes and
catalogues through phase picking, location, source mechanisms and ground motion to tomography — and
the machine-learning methods the field now uses for each.

## Time and place

- **Tuesdays 9:00–10:59 am, 325 McCone Hall**, 1 September – 1 December 2026
- Final project presentations **Tuesday 8 December** (RRR week)
- Instructor: Weiqiang Zhu · 285 McCone Hall · `zhuwq@berkeley.edu`

## Schedule

| Date | Seismology task | Machine learning |
|---|---|---|
| Sep 1 | Magnitude calibration | [Regression & uncertainty](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS207_Observational_Seismology&urlpath=lab%2Ftree%2FEPS207_Observational_Seismology%2Fdocs%2Fnotebooks%2F01_regression_uncertainty.ipynb&branch=main) |
| Sep 8 | Aftershock location forecasting | Classification — logistic, SVM, XGBoost |
| Sep 15 | Relocation uncertainty | Bias & variance, boosting, cross-validation |
| Sep 22 | Fault structure from seismicity | Clustering, mixture models, EM |
| Sep 29 | Earthquake / quarry-blast discrimination | Neural networks: classification |
| Oct 6 | Phase picking | Neural networks: segmentation |
| Oct 13 | Event detection on DAS | Neural networks: object detection |
| Oct 20 | Ground-motion prediction | Transformers, attention & masked modelling |
| Oct 27 | Template matching | Embeddings & similarity search |
| Nov 3 | Waveform generation | Generative models: the VAE |
| Nov 10 | Denoising | Denoising autoencoders; self-supervised learning |
| Nov 17 | Focal mechanism & moment tensor | Linear inversion: d = Gm, the SVD and the null space |
| Nov 24 | Location & relocation | Non-linear inversion: from a point estimate to a posterior |
| Dec 1 | Tomography | Estimating a field: regularisation, and physics as the prior |
| Dec 8 | Final project presentations | |

**Notebook links open on [Berkeley DataHub](https://datahub.berkeley.edu)** — they pull the repo
into your account and open the file, so there is nothing to install. Read-only renderings are in
the sidebar. Notebooks appear the week before their session.

## Assessment

| | |
|---|---|
| Participation | 50% — the notebooks are worked in the room, not watched |
| Project | 50% — proposal 10%, presentation 20%, report 20% |

There are no problem sets. The project is yours to choose; a region or dataset of your own works
best, and the proposal is due in week 5.

## What you need

**Python**: `numpy`, `pandas`, `scikit-learn`, `pytorch`, plus `matplotlib`, `obspy` and `scipy`.

## Previous offerings

Fall 2023 is preserved on the [`fall2023`](https://github.com/AI4EPS/EPS207_Observational_Seismology/tree/fall2023)
branch, including the lecture slides.
