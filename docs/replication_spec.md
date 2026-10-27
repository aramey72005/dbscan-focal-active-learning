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
| Dataset | Diabetes-Prediction-Analysis, 100,000 rows, 8 features, 8.5% diabetic (p. 4) | Same file (`Diabetes_prediction.csv`). It also has `EmployeeName` and `Patient_id` columns; drop both. **Note:** the proposal deck says 10,000; that needs correcting. |
| Missing values | None found (p. 4) | Verify on load; fail loudly if any are found. |
| `gender = "Other"` | Not mentioned (the paper encodes gender as 0/1, p. 4) | Drop these rows (18). |
| `smoking_history = "No Info"` | **Not stated, but inferred.** The paper's smoking codes have 5 levels, 0–4 (p. 4), so there is no code left for "No Info". Its Table 2 means and standard deviations (p. 5) are reproduced **exactly** only after these rows are dropped (age 46.54 ± 19.54, BMI 28.42 ± 6.52, HbA1c 5.56 ± 1.10, glucose 139.63 ± 42.17). | Drop these rows (35,810). |
| Duplicate rows | **Removed:** "eliminate all duplicate samples" (p. 9) | Remove them (925 after the steps above). This also removes duplicate leakage between train and test. |
| Outliers | **Removed** with 1.5 × IQR on the continuous features (p. 9–10) | Remove them, with quartiles computed on the cleaned data before splitting, as the paper orders it. This drops 5,357 rows, **57% of them diabetic**: every row with HbA1c > 8.3 or glucose > 247.5 is removed, and all of those are diabetic. That makes the task harder, so final diabetic rate is 6.9%. |
| Categorical encoding | Label encoding (p. 10). Gender: female 0, male 1. Smoking: 0 = never … 4, with `current` = 4 in Table 1 (p. 4–5) | Gender as stated. Smoking: `never` 0, `former` 1, `ever` 2, `not current` 3, `current` 4. This order matches Table 1 and reproduces the paper's reported smoking mean of 1.157 (p. 6). |
| Scaling | Z-score (Eq. 1); the paper standardizes before splitting (p. 10) | Z-score **all** features, **fit on the training pool only** (a small deliberate deviation, so no test statistics leak in; its effect on the numbers is negligible). Cosine similarity and clustering are scale-sensitive, and tree models are unaffected. |

After cleaning: **57,890 rows, 6.86% diabetic.**

### Train / validation / test split

| Item | Paper says | Our decision |
|---|---|---|
| Train/validation/test | **70 / 15 / 15** (p. 8, 10; the "70/30" on p. 23 is train vs. validation + test) | 70/15/15, **stratified**, split seed `42`, the same for every run: pool 40,523, validation 8,683, test 8,684. |
| Validation | Used to check tuned models; tuning itself is grid search with 5-fold cross-validation (p. 9, 16) | Follow the paper (see §5). The validation set is only used to report a check score, so its size does not affect the label budget. |
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
| Number of foci | k = 200 (p. 11, 17), "in the vicinity of √(n/2)" (p. 11) | 200, for **both** methods (see §4 for DBSCAN). √(40,523/2) ≈ 142, so 200 is still "in the vicinity". We keep the paper's stated value. |
| Batch size | 400 per round, i.e. 2 × foci (p. 15, 17) | 400. |
| Stopping rule | Sub-pool reaches 20% of the unlabeled data (p. 15, 17) | Stop when \|L\| = 20% of the original \|U\| (the 70% pool): 40,523 × 0.2 = 8,104 labels, i.e. 200 foci plus 20 rounds (the last batch is trimmed so every method uses exactly the same number of labels). |
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
| What gets weighted | `weighted_feature = feature × importance` (Eq. 10) | Applied to the feature space used for **similarity scoring**. Tree classifiers are unaffected by per-feature scaling, so the classifier uses unweighted features. **Note:** the foci are chosen once, in the *unweighted* space, before any SHAP weights exist. From the first round onward, similarity to those foci is measured in the *weighted* space. This is intentional and the same for both methods. |

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
| Pre-check (do first) | Most features are discrete: `gender`, `hypertension`, `heart_disease` and `smoking_history` are categorical or binary, and `age` and `HbA1c_level` take a limited set of values. DBSCAN may therefore just split the data along these category combinations, and duplicate points can make the k-distance plot flat at 0. **Before building the rest of §4**, run DBSCAN on the dev subset. Record the number of clusters, the noise rate, cluster sizes, and how much the clusters match category combinations. Bring the results to the team before going further. |
| `min_samples` | **Proposed:** start at 2 × number of features = 16. |
| `eps` | **Proposed:** choose a starting value from the bend in a k-distance plot (k = `min_samples`). |
| How `eps` / `min_samples` are tuned | **Proposed:** using **only unsupervised criteria on the training pool**, never downstream validation F1. Search a small grid around the starting values. Pick the setting with the highest silhouette score (computed on a sample of 10,000 rows, excluding noise), among settings with a noise rate ≤ 10% and at least 20 clusters. **Freeze the setting before any active-learning run.** This matches K-Means, whose settings come from the paper and are never tuned on downstream results. Neither method gets to see validation labels when choosing its foci. |
| Focus of each cluster | **Medoid:** the real point with the smallest total distance to the rest of its cluster. For large clusters, compute it on a random sample of at most 2,000 points. |
| Noise points | **Proposed:** never chosen as foci, but they stay in U and can still be picked by the scoring step. Report the noise rate. |
| Keeping the label budget equal | DBSCAN picks its own number of clusters C, while K-Means uses 200. **Proposed:** always use exactly 200 foci. If C ≥ 200, take the medoids of the 200 largest clusters. If C < 200, give each cluster foci in proportion to its size (at least 1 each), starting with its medoid and then adding the cluster's points farthest from the foci already chosen. **Team should discuss this one.** Adding farthest points is a separate selection method of its own, so it can hide what DBSCAN itself contributes. R7 (§6) checks this by letting DBSCAN use its natural cluster count. |
| Memory backup plan | If DBSCAN on the full training set is too slow or memory-heavy, fit it on a stratified sample (for example 20,000 rows). If we do this, **do the same for K-Means** so the two methods stay comparable. |

---

## 5. Final classifier and tuning

| Item | Paper says | Our decision |
|---|---|---|
| Classifiers | 10 models, best was LightGBM (Table 3) | **Proposed:** primary = **LightGBM**. Optionally add XGBoost and Random Forest. Skip AutoSklearn, which doesn't run on Windows. |
| Hyperparameter tuning | Grid search with 5-fold cross-validation, judged on precision/recall/F1 (p. 9, 16); AutoSklearn for model selection (p. 17). The LightGBM grid is not given. | Grid search with **5-fold stratified cross-validation on F1** over the final training set, the same grid for every run: `n_estimators ∈ {200, 500}`, `max_depth ∈ {-1, 6}`, `learning_rate ∈ {0.05, 0.1}`. Report validation F1 as a check. |
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
| R7: DBSCAN, natural C, `realistic` variant | Optional. One medoid per DBSCAN cluster, with no top-up to 200, so the initial label count differs from the other runs (report it). Shows how much of R5's result comes from DBSCAN itself and how much from the farthest-point top-up. |

- **Seeds:** `0, 1, 2, 3, 4` for clustering and active learning. The test split stays fixed (seed `42`).
- **Report:** the mean ± standard deviation across seeds.

## 7. Metrics

The positive class is **diabetic (1)**.

- **Primary:** recall, F1
- **Secondary:** precision, accuracy, PR-AUC, runtime, number of clusters, DBSCAN noise rate, and the percentage of diabetic rows in the final sub-pool

Accuracy alone isn't meaningful here: always predicting "not diabetic" already scores about 91.5%.

## 8. Replication target

The paper's best result (Table 3, LightGBM): **accuracy 0.9741, precision 0.9806, recall 0.9470, F1 0.9632.**

**These numbers cannot be reproduced on this dataset, by any LightGBM pipeline.**
- Accuracy 0.9741, precision 0.9806 and recall 0.9470 only fit together if the test set was about **36% diabetic**, with a **false-positive rate of about 1.1%**. That isn't the paper's own split, which keeps the natural 8.5% (or 6.9% after cleaning).
- A model's best recall at a given false-positive rate doesn't depend on the test set's class balance or the decision threshold. So we checked it directly: LightGBM trained on **every** training label reaches at most **0.74** recall at a 1.1% false-positive rate on the raw data, and **0.56** after the paper's own cleaning. Active learning trains on a subset of those same labels, so it shouldn't do better.
- A likely reason: about 18–27% of diabetics have HbA1c ≤ 6.2 and glucose ≤ 159, values that non-diabetics also take. No model can separate those rows from non-diabetics using these 8 features.
- The paper does mention evaluating on resampled sets ("1:4 ratio between non-diabetic and diabetic cases", p. 20, for its Table 5). That could explain the high precision and accuracy, but not the recall, because recall doesn't depend on class balance.

**Success criterion, revised:** the baseline must match the paper's **method** step by step (§1–§3, §5), with every unstated gap filled as documented here. It does **not** need to match Table 3's numbers. The project's question, K-Means vs. DBSCAN, only needs both arms built on the same faithful baseline. The report should present the infeasibility evidence above.

---

## Sign-off

| Teammate | Reviewed | Comments |
|---|---|---|
| Amber Ramey | ☑ 2026-10-27 | Approved, including the review edits in commit 260f301. |
| Nairuti Vora | ☐ | |
| Jonus Ronquillo | ☐ | |
