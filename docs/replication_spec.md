# Replication Spec

> **Status: DRAFT. Every decision marked _Proposed_ needs team sign-off before coding starts.**

## What this document is

This spec is the single agreed description of exactly what we build. Zhang et al. (2025) did not publish code, so we have to rebuild their pipeline from the paper. The paper gives many settings but leaves some out. This document records:

1. **What the paper specifies**, with page references, so we copy it faithfully.
2. **What the paper leaves out**, with the choice we made for each gap and why.

Why it matters:
- **All three of us build the same pipeline.** Without it, each person fills the gaps differently and the pieces don't fit.
- **The comparison stays fair.** The K-Means and DBSCAN versions must differ in only one way: the clustering step.
- **The final report can explain every choice.** Reviewers will ask why our baseline numbers differ from the paper's, and this is the answer.

If you change a decision, edit this file in a pull request so the change is visible to everyone.

---

## 1. Data

| Item | Paper says | Our decision |
|---|---|---|
| Dataset | Diabetes-Prediction-Analysis, 100,000 rows, 8 features, 8.5% diabetic (p. 4) | Same. **Note:** the proposal deck says 10,000; that needs correcting. |
| Missing values | None found (p. 4) | Verify on load; fail loudly if any are found. |
| Duplicate rows | Not mentioned | **Proposed:** keep them, to match the paper. Log the count. |
| `gender = "Other"` | Not mentioned (the paper encodes gender as 0/1) | **Proposed:** drop these rows (a very small number). Log the count. |
| Categorical encoding | 8-dimensional feature space (p. 8), no one-hot encoding | **Proposed:** label-encode `gender` and `smoking_history`, keeping 8 features. |
| Scaling | Z-score on continuous features, for scale-sensitive models (p. 4) | **Proposed:** z-score **all** features, fit on the training set only. Cosine similarity and clustering are scale-sensitive, and tree models are unaffected. |

### Train / validation / test split

| Item | Paper says | Our decision |
|---|---|---|
| Train/test | 70/30 (p. 23) | 70/30, **stratified**, split seed `42`, the same for every run. |
| Validation | "tuned using cross-validation" (p. 17), no details | **Proposed:** hold out 20% of the training set (stratified) as a validation set for all tuning. |
| Test set | — | **Untouched until final evaluation.** No tuning on it, ever. |

### Development subset

- **Proposed:** a stratified 10,000-row sample (seed `42`) for development and debugging. Save its row IDs to `data/processed/dev_subset_ids.csv` so everyone uses the same rows.
- All **reported results** use the full dataset.

---

## 2. The Focal Active Learning pipeline

This is a simulated active learning setup. Every training row has a true label, but we treat the training pool as unlabeled and "reveal" a label only when a row is selected.

```
U = training pool (labels hidden)        L = labeled sub-pool (starts empty)

1. Focal selection: cluster U and pick 200 foci; move the foci into L.
2. Repeat until |L| >= 20% of the original |U|:
     a. Train the selection model on L.
     b. Compute SHAP importances and reweight the features.
     c. Score every row in U (similarity + uncertainty + minority).
     d. Move the top 400 rows from U into L.
3. Train the final classifier on L and evaluate it on the test set.
```

| Item | Paper says | Our decision |
|---|---|---|
| Number of foci | k = 200 (p. 11, 17) | 200, for **both** methods (see §4 for DBSCAN). |
| Batch size | 400 per round, i.e. 2 × foci (p. 15, 17) | 400. |
| Stopping rule | Sub-pool reaches 20% of the unlabeled data (p. 15, 17) | Stop when \|L\| ≥ 20% of the original \|U\|. |
| Are the foci labeled? | "stabilized foci (labeled samples)" (p. 13) | Yes, the foci form the initial L. |
| What the final model trains on | Ambiguous ("fed back into the overall training dataset", p. 15) | **Proposed:** the final classifier trains on **L only**. Using fewer labels is the point of active learning. |

### 2a. Scoring each unlabeled sample (paper Eq. 8)

`score = α · similarity + β · uncertainty + γ · minority`

| Term | Paper says | Our decision |
|---|---|---|
| Similarity | Cosine similarity to focus µⱼ (Eq. 2) | Max cosine similarity to **any** focus, computed in the SHAP-weighted feature space. |
| Uncertainty | "– maxᶜ P(y=c\|x)" (Eq. 6), probably a typo for 1 – max | `1 − max_c P(y=c|x)` from the selection model. |
| Minority term | δ(yᵢ = minority class), which uses the **true label** (Eq. 7–8) | See below. |
| α, β, γ | **Not given** | **Proposed:** min-max scale each term to [0, 1] and use α = β = γ = 1/3. Optionally tune on validation with a small grid, using the same weights for both methods. |
| Selection model | **Not given** | **Proposed:** LightGBM with fixed default settings during the loop, to keep it fast. |

**The minority-term problem.** The paper's formula uses each sample's true label *before* that sample is labeled. A real deployment couldn't do this, and it probably inflates results. **Proposed:** run two variants.
- **`paper` variant:** uses the true label as written. Used only to check our rebuild against the paper's numbers.
- **`realistic` variant:** uses the predicted probability `P(y = diabetic | x)` from the selection model. This is the **main variant** for the K-Means vs. DBSCAN comparison.

### 2b. SHAP feature weighting (paper Eq. 9–10)

| Item | Paper says | Our decision |
|---|---|---|
| How SHAP is computed | Each iteration, on the current sub-pool (p. 13) | `shap.TreeExplainer` on the selection model, run over L. |
| Importance | Mean SHAP contribution | `importance_j = mean(|SHAP_j|)` over L. |
| Normalization | **Not given** | **Proposed:** rescale so the importances average 1.0. If every feature were equally important, all weights would be 1 and nothing would change. |
| What gets weighted | `weighted_feature = feature × importance` (Eq. 10) | Applied to the feature space used for **similarity scoring**. Tree classifiers are unaffected by per-feature scaling, so the classifier uses unweighted features. |

---

## 3. Baseline: K-Means focal selection (the paper's method)

| Item | Paper says | Our decision |
|---|---|---|
| Algorithm | K-Means with cosine similarity (Eq. 2–4) | L2-normalize the rows, then run `sklearn.cluster.KMeans`. On unit-length vectors, Euclidean distance ranks points the same way cosine similarity does. |
| k | 200 | 200, `n_init=10`, seeded. |
| Focus | Real sample most similar to the centroid (Eq. 5) | Same. |

---

## 4. Proposed method: DBSCAN focal selection (our change)

**This is the only part that differs from the baseline.**

| Item | Decision |
|---|---|
| Input space | Same L2-normalized, standardized features as K-Means. Use the Euclidean metric, which has fast indexing and gives the same neighborhoods as cosine. |
| `min_samples` | **Proposed:** start at 2 × number of features = 16, then tune on validation. |
| `eps` | **Proposed:** choose a starting value from the bend in a k-distance plot, then search a small grid around it on validation. |
| Focus of each cluster | **Medoid:** the real point with the smallest total distance to the rest of its cluster. For large clusters, compute it on a random sample of at most 2,000 points. |
| Noise points | **Proposed:** never chosen as foci, but they stay in U and can still be picked by the scoring step. Report the noise rate. |
| Keeping the label budget equal | DBSCAN picks its own number of clusters C, while K-Means uses 200. **Proposed:** always use exactly 200 foci. If C ≥ 200, take the medoids of the 200 largest clusters. If C < 200, give each cluster foci in proportion to its size (at least 1 each), starting with its medoid and then adding the cluster's points farthest from the foci already chosen. **Team should discuss this one.** |
| Memory backup plan | If DBSCAN on the full training set is too slow or memory-heavy, fit it on a stratified sample (for example 20,000 rows). If we do this, **do the same for K-Means** so the two methods stay comparable. |

---

## 5. Final classifier and tuning

| Item | Paper says | Our decision |
|---|---|---|
| Classifiers | 10 models, best was LightGBM (Table 3) | **Proposed:** primary = **LightGBM**. Optionally add XGBoost and Random Forest. Skip AutoSklearn, which doesn't run on Windows. |
| Hyperparameter tuning | AutoSklearn (p. 17) | **Proposed:** the same small grid search on validation for both methods: `n_estimators ∈ {200, 500}`, `max_depth ∈ {-1, 6}`, `learning_rate ∈ {0.05, 0.1}`. |
| Class weighting | Not mentioned | None. The sampling step is what handles the imbalance. |

---

## 6. Experiments to run

| Run | Purpose |
|---|---|
| **R1: Full-data LightGBM** (no active learning) | Reference point: what's possible with every label. |
| **R2: Random 20% sample** | Control: does focal selection beat random sampling? |
| **R3: K-Means, `paper` variant** | Replication check against Table 3. |
| **R4: K-Means, `realistic` variant** | Baseline for the main comparison. |
| **R5: DBSCAN, `realistic` variant** | **Our proposed method.** |
| R6: DBSCAN, `paper` variant | Optional, for completeness. |

- **Seeds:** `0, 1, 2, 3, 4` for clustering and active learning. The test split stays fixed (seed `42`).
- **Report:** the mean ± standard deviation across seeds.

## 7. Metrics

The positive class is **diabetic (1)**.

- **Primary:** recall, F1
- **Secondary:** precision, accuracy, PR-AUC, runtime, number of clusters, DBSCAN noise rate, and the percentage of diabetic rows in the final sub-pool

Accuracy alone isn't meaningful here: always predicting "not diabetic" already scores about 91.5%.

## 8. Replication target

The paper's best result (Table 3, LightGBM): **accuracy 0.9741, precision 0.9806, recall 0.9470, F1 0.9632.**

**Proposed success criterion:** R3 lands within about 3 points of these on recall and F1. If it doesn't, we document the likely reasons (the missing α/β/γ, AutoSklearn, and so on) and still go ahead with the comparison. The comparison only needs both methods built on the same baseline, not an exact match with the paper.

---

## Sign-off

| Teammate | Reviewed | Comments |
|---|---|---|
| Amber Ramey | ☐ | |
| Nairuti Vora | ☐ | |
| Jonus Ronquillo | ☐ | |
