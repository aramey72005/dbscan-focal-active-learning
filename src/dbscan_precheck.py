"""DBSCAN pre-check on the dev subset (replication spec §4, "Pre-check").

Question: on this mostly-discrete data, does DBSCAN find useful structure, or
does it just split the data along category combinations?

    python src/preprocess.py        # once, to create data/processed/
    python src/dbscan_precheck.py

Writes results/precheck/: k-distance plot, grid results CSV, and summary JSON.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from preprocess import FEATURES, OUT_DIR as DATA_DIR, ROOT, TARGET

OUT_DIR = ROOT / "results" / "precheck"
SEED = 0
BASE_MIN_SAMPLES = 2 * len(FEATURES)  # spec §4 starting value: 16
EPS_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
MIN_SAMPLES_GRID = [8, 16, 32]
# Spec §4 selection rule.
MAX_NOISE = 0.10
MIN_CLUSTERS = 20
# Features that take only a few values; their combinations are the "category strata".
DISCRETE = ["gender", "hypertension", "heart_disease", "smoking_history"]


def load_dev():
    pool = pd.read_csv(DATA_DIR / "pool.csv")
    ids = pd.read_csv(DATA_DIR / "dev_subset_ids.csv")["row_id"]
    dev = pool[pool["row_id"].isin(ids)].reset_index(drop=True)
    X = normalize(dev[[f"z_{f}" for f in FEATURES]].to_numpy())  # L2 rows, as in spec §3-4
    strata = dev[DISCRETE].astype(int).astype(str).agg("-".join, axis=1)
    return dev, X, pd.factorize(strata)[0]


def k_distances(X, k):
    # n_neighbors=k includes the point itself, matching how sklearn's DBSCAN counts min_samples.
    dist, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    return np.sort(dist[:, -1])


def knee(sorted_d):
    """Point farthest from the straight line joining the curve's ends."""
    x = np.linspace(0, 1, len(sorted_d))
    y = (sorted_d - sorted_d[0]) / (sorted_d[-1] - sorted_d[0])
    return float(sorted_d[np.argmax(x - y)])


def stratum_purity(labels, strata):
    """Share of clustered points that sit in their cluster's most common category stratum."""
    mask = labels >= 0
    if not mask.any():
        return float("nan")
    df = pd.DataFrame({"c": labels[mask], "s": strata[mask]})
    return float(df.groupby("c")["s"].agg(lambda s: s.value_counts().iloc[0]).sum() / mask.sum())


def describe(labels, X, strata, y):
    mask = labels >= 0
    sizes = np.bincount(labels[mask]) if mask.any() else np.array([])
    n_clusters = len(sizes)
    sil = (
        float(silhouette_score(X[mask], labels[mask], random_state=SEED))
        if n_clusters >= 2
        else float("nan")
    )
    return {
        "n_clusters": n_clusters,
        "noise_rate": round(float(1 - mask.mean()), 4),
        "largest_cluster_share": round(float(sizes.max() / len(labels)), 4) if n_clusters else 0.0,
        "median_cluster_size": float(np.median(sizes)) if n_clusters else 0.0,
        "clusters_ge_16": int((sizes >= 16).sum()),
        "silhouette": round(sil, 4),
        "stratum_purity": round(stratum_purity(labels, strata), 4),
        "ari_vs_strata": round(float(adjusted_rand_score(strata[mask], labels[mask])), 4)
        if mask.any()
        else float("nan"),
        "diabetic_rate_noise": round(float(y[~mask].mean()), 4) if (~mask).any() else float("nan"),
        "diabetic_rate_clustered": round(float(y[mask].mean()), 4) if mask.any() else float("nan"),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, X, strata = load_dev()
    y = dev[TARGET].to_numpy()

    feat_cols = [f"z_{f}" for f in FEATURES]
    summary = {
        "dev_rows": len(dev),
        "distinct_points": int(len(np.unique(X.round(12), axis=0))),
        "duplicate_points": int(dev.duplicated(subset=feat_cols).sum()),
        "category_strata": int(strata.max() + 1),
        "distinct_values_per_feature": {f: int(dev[f].nunique()) for f in FEATURES},
    }

    # k-distance plots and knees.
    fig, ax = plt.subplots(figsize=(7, 4))
    knees = {}
    for k in MIN_SAMPLES_GRID:
        d = k_distances(X, k)
        knees[k] = knee(d)
        summary[f"zero_k_distance_share_k{k}"] = round(float((d < 1e-9).mean()), 4)
        ax.plot(d, label=f"k={k} (knee ≈ {knees[k]:.3f})")
    ax.set(
        xlabel="Points sorted by k-distance",
        ylabel="Distance to k-th neighbor (L2-normalized space)",
        title="k-distance plot, dev subset",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k_distance.png", dpi=150)
    summary["knee_eps"] = {str(k): round(v, 4) for k, v in knees.items()}

    # DBSCAN grid around each knee.
    rows = []
    for k in MIN_SAMPLES_GRID:
        for m in EPS_MULTIPLIERS:
            eps = knees[k] * m
            labels = DBSCAN(eps=eps, min_samples=k).fit_predict(X)
            rows.append({"min_samples": k, "eps_mult": m, "eps": round(eps, 4), **describe(labels, X, strata, y)})
            print(rows[-1])
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "dbscan_grid.csv", index=False)

    ok = grid[(grid.noise_rate <= MAX_NOISE) & (grid.n_clusters >= MIN_CLUSTERS)]
    summary["settings_passing_spec_rule"] = int(len(ok))
    summary["spec_rule_pick"] = (
        ok.sort_values("silhouette", ascending=False).iloc[0].to_dict() if len(ok) else None
    )

    # K-Means reference at k=200 (scaled down from the full pool is not needed for this check).
    km = KMeans(n_clusters=200, n_init=10, random_state=SEED).fit_predict(X)
    summary["kmeans_200_reference"] = describe(km, X, strata, y)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
