# EPS 207 · Laboratory in Observational Seismology

Machine learning now builds earthquake catalogues, locates events, and inverts for structure.
Each week takes one **method** and one **published result**: you work the notebook live in class,
reproduce part of the result, then try to break it.

## Time and place

- **Tuesdays 9:00–10:59 am, 325 McCone Hall**, 1 September – 1 December 2026
- Final project presentations **Tuesday 8 December** (RRR week)
- Instructor: Weiqiang Zhu · 285 McCone Hall · `zhuwq@berkeley.edu`

## Schedule

| # | Date | Topic |
|---|---|---|
| 1 | Sep 1 | [Regression & uncertainty](notebooks/01_regression_uncertainty.ipynb) |
| 2 | Sep 8 | Classification — logistic, SVM, XGBoost |
| 3 | Sep 15 | Bias & variance, boosting, cross-validation |
| 4 | Sep 22 | Clustering, mixture models, EM |
| 5 | Sep 29 | Neural networks: classification |
| 6 | Oct 6 | Neural networks: segmentation |
| 7 | Oct 13 | Neural networks: object detection (DAS) |
| 8 | Oct 20 | Transformers, attention & masked modelling |
| 9 | Oct 27 | Embeddings & similarity search |
| 10 | Nov 3 | Generative models: train a VAE |
| 11 | Nov 10 | Denoising |
| 12 | Nov 17 | Inversion I — the linear case: moment tensor & focal mechanism |
| 13 | Nov 24 | Inversion II — the non-linear case: location, relocation and the posterior |
| 14 | Dec 1 | Inversion III — estimating a field: regularisation, and physics instead of it |
| — | Dec 8 | Final project presentations |

Notebooks are linked as they are released, normally the week before the session.

## Assessment

| | |
|---|---|
| Participation | 50% — the notebooks are worked in the room, not watched |
| Project | 50% — proposal 10%, presentation 20%, report 20% |

There are no problem sets. The project is yours to choose; a region or dataset of your own works
best, and the proposal is due in week 5.

## What you need

**Python**: `numpy`, `pandas`, `scikit-learn`, `pytorch`, plus `matplotlib`, `obspy` and `scipy`.
Everything runs on **Google Colab** or **Berkeley DataHub** — no local install and no GPU is
required for any session.

**Data** is public throughout: USGS, IRIS, NCEDC and SCEDC web services, HuggingFace, and datasets
released with this repository. You will never need credentials.

**Background**: graduate standing. No prior seismology is assumed, but you should be comfortable
writing Python and reading a paper.

## Previous offerings

Fall 2023 is preserved on the [`fall2023`](https://github.com/AI4EPS/EPS207_Observational_Seismology/tree/fall2023)
branch, including the lecture slides.
