"""Focal Active Learning loop (replication spec §2).

Simulated pool-based active learning: every pool row has a true label, but the
loop only "reveals" a label once that row moves into the labeled sub-pool L.
The single exception is the `paper` variant's minority term, which uses the true
label of unlabeled rows exactly as the paper's Eq. 7-8 is written (spec §2a).
"""

import lightgbm as lgb
import numpy as np
import shap
from sklearn.preprocessing import normalize

BATCH_SIZE = 400  # 2 x foci, paper p. 15, 17
BUDGET_FRACTION = 0.20  # stop when |L| reaches 20% of the pool, paper p. 15, 17
ALPHA = BETA = GAMMA = 1 / 3  # not given in the paper; spec §2a
VARIANTS = ("paper", "realistic")


def selection_model(seed):
    """Fixed-setting LightGBM used inside the loop (spec §2a)."""
    return lgb.LGBMClassifier(random_state=seed, verbose=-1)


def shap_weights(model, Z_labeled):
    """Spec §2b: mean |SHAP| per feature over L, rescaled to average 1.0."""
    values = shap.TreeExplainer(model).shap_values(Z_labeled)
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    importance = np.abs(values).mean(axis=0)
    if importance.sum() == 0:  # model found no splits (e.g. one class in L)
        return np.ones_like(importance)
    return importance / importance.mean()


def minmax(v):
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 0 else np.zeros_like(v)


def score_unlabeled(Z_u, Z_foci, weights, p_diabetic, y_u, variant):
    """Eq. 8: alpha * similarity + beta * uncertainty + gamma * minority, each min-max scaled."""
    # Similarity: max cosine similarity to any focus, in the SHAP-weighted space.
    similarity = (normalize(Z_u * weights) @ normalize(Z_foci * weights).T).max(axis=1)
    # Uncertainty: 1 - max_c P(y=c|x) (Eq. 6, read as 1 - max).
    uncertainty = 1 - np.maximum(p_diabetic, 1 - p_diabetic)
    if variant == "paper":
        minority = (y_u == 1).astype(float)  # true label, as written in the paper
    else:
        minority = p_diabetic  # predicted probability; spec's main variant
    return ALPHA * minmax(similarity) + BETA * minmax(uncertainty) + GAMMA * minmax(minority)


def run_focal_al(Z, y, foci, variant, seed=0, log=None):
    """Return the pool row positions of the final labeled sub-pool L.

    Z: standardized pool features (n x 8). y: pool labels. foci: row positions from focal.py.
    The last batch is trimmed so |L| equals the budget exactly, which keeps the
    label count identical across methods and against the random-sample control.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    n = len(y)
    budget = int(BUDGET_FRACTION * n)
    labeled = np.zeros(n, dtype=bool)
    labeled[foci] = True
    Z_foci = Z[foci]

    rnd = 0
    while labeled.sum() < budget:
        L, U = np.flatnonzero(labeled), np.flatnonzero(~labeled)
        model = selection_model(seed).fit(Z[L], y[L])
        weights = shap_weights(model, Z[L])
        p = model.predict_proba(Z[U])[:, 1]
        scores = score_unlabeled(Z[U], Z_foci, weights, p, y[U], variant)

        take = min(BATCH_SIZE, budget - labeled.sum())
        labeled[U[np.argsort(-scores, kind="stable")[:take]]] = True
        rnd += 1
        if log is not None:
            log.append(
                {
                    "round": rnd,
                    "labeled": int(labeled.sum()),
                    "diabetic_share_L": round(float(y[labeled].mean()), 4),
                    "weights": np.round(weights, 3).tolist(),
                }
            )
    return np.flatnonzero(labeled)
