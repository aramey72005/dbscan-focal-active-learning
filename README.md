# DBSCAN vs. K-Means for Focal Sample Selection

**A controlled comparison for diabetes prediction**

Team: Amber Ramey, Nairuti Vora, Jonus Ronquillo

## Overview

Diabetes prediction is a binary classification problem where datasets usually contain far more non-diabetic than diabetic cases. Zhang et al. (2025) handle this imbalance with **Focal Active Learning**: they cluster the training data with **K-Means**, choose a representative point ("focus") from each cluster, and build a smaller labeled sub-pool around those foci.

This project **replaces K-Means with DBSCAN** inside the same Focal Active Learning pipeline and checks whether that improves prediction, especially for the minority (diabetic) class.

### Research question

> Does replacing K-Means with DBSCAN for focal sample selection improve recall and F1-score in diabetes prediction?

- **Hypothesis:** Foci selected by DBSCAN will improve diabetic-class recall and F1, because density-based clusters can handle noise and non-spherical structure.
- **Null hypothesis:** Replacing K-Means with DBSCAN gives no meaningful improvement in downstream classification.

See [docs/project_overview.md](docs/project_overview.md) for the full experiment design and a summary of the paper.

## Repository layout

```
.
├── data/
│   ├── raw/            # Original dataset (not committed; see data/README.md)
│   └── processed/      # Cleaned / standardized data (not committed)
├── docs/
│   ├── paper/          # Original paper (Zhang et al., 2025, PLOS One)
│   ├── proposal/       # Our project proposal deck
│   └── project_overview.md
├── notebooks/          # Exploratory analysis and experiments
├── src/                # Reusable pipeline code (preprocessing, focal selection, training, evaluation)
├── results/
│   └── figures/        # Plots and tables for the final report
├── requirements.txt
└── CONTRIBUTING.md     # How we work together in this repo
```

## Getting started

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd dbscan-focal-active-learning
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download the dataset into `data/raw/` (instructions in [data/README.md](data/README.md)).

## Reference

Zhang W, Xia Z, Cai G, Wang J, Dong X (2025). *Enhancing diabetes risk prediction through focal active learning and machine learning models.* PLoS One 20(7): e0327120. https://doi.org/10.1371/journal.pone.0327120

The paper is open access under CC BY 4.0, and a copy is in [docs/paper/](docs/paper/).
