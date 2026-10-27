"""Run the replication experiments (replication spec §5-7).

    python src/run_experiments.py                    # R1-R4, seeds 0-4, full pool
    python src/run_experiments.py --dev              # same, on the 10,000-row dev subset
    python src/run_experiments.py --runs R3 --seeds 0

Runs:
    R1  LightGBM on every pool label (no active learning)
    R2  random 20% of the pool
    R3  K-Means focal AL, `paper` variant (replication check vs. Table 3)
    R4  K-Means focal AL, `realistic` variant (baseline for the DBSCAN comparison)

Each finished run is appended to runs.csv straight away, so a long job keeps
its progress; summary.csv holds mean and std across seeds.
"""

import argparse
import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from active_learning import BUDGET_FRACTION, run_focal_al
from focal import select_foci
from preprocess import FEATURES, OUT_DIR as DATA_DIR, ROOT, TARGET

Z_COLS = [f"z_{f}" for f in FEATURES]
SEEDS = [0, 1, 2, 3, 4]
RUNS = {
    "R1": {"sampler": "full"},
    "R2": {"sampler": "random"},
    "R3": {"sampler": "focal", "method": "kmeans", "variant": "paper"},
    "R4": {"sampler": "focal", "method": "kmeans", "variant": "realistic"},
}
# Spec §5: the same small grid for every run (the paper does not give its grid).
GRID = {"n_estimators": [200, 500], "max_depth": [-1, 6], "learning_rate": [0.05, 0.1]}
PAPER_TABLE3 = {"accuracy": 0.9741, "precision": 0.9806, "recall": 0.9470, "f1": 0.9632}


def load(dev):
    pool = pd.read_csv(DATA_DIR / "pool.csv")
    if dev:
        ids = pd.read_csv(DATA_DIR / "dev_subset_ids.csv")["row_id"]
        pool = pool[pool["row_id"].isin(ids)].reset_index(drop=True)
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return [(d[Z_COLS].to_numpy(), d[TARGET].to_numpy()) for d in (pool, val, test)]


def training_rows(spec, Z, y, seed, al_log):
    if spec["sampler"] == "full":
        return np.arange(len(y))
    if spec["sampler"] == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(len(y), size=int(BUDGET_FRACTION * len(y)), replace=False)
    foci = select_foci(spec["method"], Z, seed=seed)
    return run_focal_al(Z, y, foci, spec["variant"], seed=seed, log=al_log)


def fit_final(Z_train, y_train, Z_val, y_val, seed):
    """Paper p. 9, 16: grid search with 5-fold cross-validation on F1, checked on validation."""
    search = GridSearchCV(
        lgb.LGBMClassifier(random_state=seed, verbose=-1),
        GRID,
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
    ).fit(Z_train, y_train)
    val_f1 = f1_score(y_val, search.best_estimator_.predict(Z_val))
    return val_f1, search.best_estimator_, search.best_params_


def evaluate(model, Z, y):
    pred = model.predict(Z)
    prob = model.predict_proba(Z)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "pr_auc": average_precision_score(y, prob),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dev", action="store_true", help="use the dev subset as the pool")
    args = parser.parse_args()

    out_dir = ROOT / "results" / ("replication_dev" if args.dev else "replication")
    for sub in ("al_logs", "test_probs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.csv"

    (Z, y), (Z_val, y_val), (Z_test, y_test) = load(args.dev)
    print(f"pool={len(y)} val={len(y_val)} test={len(y_test)}")

    for run, seed in itertools.product(args.runs, args.seeds):
        spec = RUNS[run]
        start = time.perf_counter()
        al_log = []
        rows = training_rows(spec, Z, y, seed, al_log)
        sample_s = time.perf_counter() - start
        val_f1, model, params = fit_final(Z[rows], y[rows], Z_val, y_val, seed)
        result = {
            "run": run,
            "seed": seed,
            "n_labels": len(rows),
            "diabetic_share_train": round(float(y[rows].mean()), 4),
            **evaluate(model, Z_test, y_test),
            "val_f1": val_f1,
            "params": json.dumps(params),
            "sampling_s": round(sample_s, 1),
            "total_s": round(time.perf_counter() - start, 1),
        }
        if al_log:
            (out_dir / "al_logs" / f"{run}_seed{seed}.json").write_text(json.dumps(al_log))
        # Test-set P(diabetic), row-aligned with test.csv, for confusion matrices and PR/ROC curves.
        np.save(out_dir / "test_probs" / f"{run}_seed{seed}.npy", model.predict_proba(Z_test)[:, 1])
        pd.DataFrame([result]).to_csv(runs_path, mode="a", header=not runs_path.exists(), index=False)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()})

    runs = pd.read_csv(runs_path)
    metrics = ["accuracy", "precision", "recall", "f1", "pr_auc", "diabetic_share_train", "total_s"]
    summary = runs.groupby("run")[metrics].agg(["mean", "std"]).round(4)
    summary.to_csv(out_dir / "summary.csv")
    print(summary.to_string())
    print("Paper Table 3 (LightGBM):", PAPER_TABLE3)


if __name__ == "__main__":
    main()
