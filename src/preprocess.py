"""Preprocessing for the Focal Active Learning replication (replication spec §1).

Follows the paper's pipeline (Zhang et al. 2025, pp. 4, 8-10): drop rows with
unknown smoking history, remove duplicates, remove IQR outliers, label-encode,
z-score, then split 70/15/15. Writes everything later steps need to
data/processed/.

    python src/preprocess.py

Outputs (data/processed/):
    pool.csv              70% training pool U (labels present but "hidden" by the AL loop)
    val.csv               15% validation set
    test.csv              15% test set; do not touch until final evaluation
    dev_subset_ids.csv    row_ids of the 10,000-row stratified dev subset (drawn from the pool)
    scaler.json           z-score mean/std per feature, fit on the training split only
    preprocess_log.json   row counts removed at each step, and split sizes

Each split file has `row_id` (0-based row in the raw CSV), the 8 encoded but
unscaled features, the 8 z-scored features (prefix `z_`), and `diabetes`.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Diabetes_prediction.csv"
OUT_DIR = ROOT / "data" / "processed"

SPLIT_SEED = 42
VAL_SIZE = TEST_SIZE = 0.15  # paper p. 8, 10: 70/15/15
DEV_ROWS = 10_000

TARGET = "diabetes"
FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
CONTINUOUS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
# Columns in the raw file that are not features (names and IDs). Must never reach a model.
DROP_COLUMNS = ["EmployeeName", "Patient_id"]

GENDER_CODES = {"Female": 0, "Male": 1}  # paper p. 4, Table 1
# Paper p. 4: five smoking levels, 0 = never up to 4. Table 1 shows current = 4, and
# this ordering reproduces the paper's reported mean of 1.157 (p. 6). "No Info" has
# no code because those rows are dropped (see drop_unknown_smoking).
SMOKING_CODES = {"never": 0, "former": 1, "ever": 2, "not current": 3, "current": 4}


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    expected = set(FEATURES + [TARGET] + DROP_COLUMNS)
    if set(df.columns) != expected:
        raise ValueError(f"Unexpected columns in {path.name}: {sorted(df.columns)}")
    df = df.drop(columns=DROP_COLUMNS)
    df.insert(0, "row_id", range(len(df)))
    return df


def drop_unknown_smoking(df, log):
    """Drop smoking_history == "No Info".

    Not stated in the paper, but inferred: its Table 2 means and stds (p. 5) are
    reproduced exactly only after these rows are removed, and its smoking codes
    have five levels (p. 4), which leaves no code for "No Info".
    """
    unknown = df["smoking_history"] == "No Info"
    log["smoking_no_info_dropped"] = int(unknown.sum())
    return df.loc[~unknown]


def remove_duplicates(df, log):
    """Paper p. 9: "eliminate all duplicate samples"."""
    dup = df.duplicated(subset=FEATURES + [TARGET])
    log["duplicates_dropped"] = int(dup.sum())
    return df.loc[~dup]


def remove_iqr_outliers(df, log):
    """Paper p. 9-10: drop rows outside 1.5 x IQR on the continuous features."""
    keep = pd.Series(True, index=df.index)
    log["iqr_outliers"] = {}
    for col in CONTINUOUS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        inside = df[col].between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
        log["iqr_outliers"][col] = int((~inside).sum())
        keep &= inside
    log["iqr_rows_dropped"] = int((~keep).sum())
    log["iqr_dropped_diabetic_share"] = round(float(df.loc[~keep, TARGET].mean()), 4)
    return df.loc[keep]


def clean(df, log):
    n_missing = int(df[FEATURES + [TARGET]].isna().sum().sum())
    if n_missing:
        raise ValueError(f"Found {n_missing} missing values; the paper reports none.")
    log["raw_rows"] = len(df)

    other = df["gender"] == "Other"  # paper encodes gender as 0/1 only
    log["gender_other_dropped"] = int(other.sum())
    df = df.loc[~other]

    df = drop_unknown_smoking(df, log)
    df = remove_duplicates(df, log)
    df = remove_iqr_outliers(df, log)
    return df.reset_index(drop=True)


def encode(df):
    df = df.copy()
    df["gender"] = df["gender"].map(GENDER_CODES)
    df["smoking_history"] = df["smoking_history"].map(SMOKING_CODES)
    if df[FEATURES].isna().any().any():
        raise ValueError("Unknown category found while encoding.")
    df[FEATURES] = df[FEATURES].astype(float)
    df[TARGET] = df[TARGET].astype(int)
    return df


def split(df):
    pool, rest = train_test_split(
        df, test_size=VAL_SIZE + TEST_SIZE, stratify=df[TARGET], random_state=SPLIT_SEED
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest[TARGET], random_state=SPLIT_SEED
    )
    return pool, val, test


def fit_scaler(train):
    mean = train[FEATURES].mean()
    std = train[FEATURES].std(ddof=0)
    return {f: {"mean": float(mean[f]), "std": float(std[f])} for f in FEATURES}


def add_scaled(df, scaler):
    df = df.copy()
    for f in FEATURES:
        df[f"z_{f}"] = (df[f] - scaler[f]["mean"]) / scaler[f]["std"]
    return df


def main():
    log = {}
    df = encode(clean(load_raw(), log))
    log["rows_after_cleaning"] = len(df)

    pool, val, test = split(df)
    # Deviation: the paper standardizes before splitting (p. 10); we fit on the pool only
    # so no test statistics leak in. The effect on the numbers is negligible.
    scaler = fit_scaler(pool)
    pool, val, test = (add_scaled(d, scaler) for d in (pool, val, test))

    dev, _ = train_test_split(
        pool, train_size=DEV_ROWS, stratify=pool[TARGET], random_state=SPLIT_SEED
    )

    log.update(
        {
            "pool_rows": len(pool),
            "val_rows": len(val),
            "test_rows": len(test),
            "dev_rows": len(dev),
            "al_label_budget_20pct_of_pool": int(0.2 * len(pool)),
            "diabetic_rate": {
                name: round(float(d[TARGET].mean()), 4)
                for name, d in [("pool", pool), ("val", val), ("test", test), ("dev", dev)]
            },
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in [("pool", pool), ("val", val), ("test", test)]:
        d.sort_values("row_id").to_csv(OUT_DIR / f"{name}.csv", index=False)
    dev[["row_id"]].sort_values("row_id").to_csv(OUT_DIR / "dev_subset_ids.csv", index=False)
    (OUT_DIR / "scaler.json").write_text(json.dumps(scaler, indent=2))
    (OUT_DIR / "preprocess_log.json").write_text(json.dumps(log, indent=2))

    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
