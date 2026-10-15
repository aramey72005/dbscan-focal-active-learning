# Data

Data files are **not committed** to this repo (see `.gitignore`). Every teammate downloads them locally.

## Diabetes Prediction Dataset

- **Source:** https://github.com/AbiDanalyst/Diabetes-Prediction-Analysis
- **File:** download the dataset CSV from that repo and save it in `data/raw/`
- **Features:** age, gender, BMI, hypertension, heart disease, smoking history, HbA1c level, blood glucose level
- **Target:** `diabetes` (1 = positive, 0 = negative)

## Folder conventions

- `raw/`: original downloaded files. Never edit these by hand.
- `processed/`: cleaned / encoded / standardized outputs produced by our code.

## Optional: PIMA Indians Diabetes Database

The paper also validated on https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database. Not part of our core experiment, but useful if we want to test generalization.
