# Project Overview

## The original paper

**Zhang et al. (2025), "Enhancing diabetes risk prediction through focal active learning and machine learning models"** ([PDF](paper/Zhang2025_FocalActiveLearning_PLOSOne.pdf))

Summary of the Focal Active Learning pipeline:

1. **Focal selection.** Run K-Means on the training data (the paper uses k = 200) with **cosine similarity** to assign points to clusters. For each cluster, the sample most similar to the centroid becomes that cluster's *focus*.
2. **Sub-pool construction.** Starting from the foci, iteratively add similar samples to a smaller labeled sub-pool. This gradually raises the share of minority-class samples. Iteration stops once the sub-pool reaches a set **proportion of the unlabeled data**, not after a fixed number of rounds.
3. **Feature weighting.** At each iteration, SHAP values measure feature importance, and an attention mechanism adjusts the feature weights.
4. **Model training.** Train classifiers on the sub-pool: Logistic Regression, SVM, Random Forest, XGBoost, and others. Tune hyperparameters, e.g. estimators and max depth for XGBoost, kernel and C for SVM.

Reported result: 97.41% accuracy and 94.70% recall on the diabetes prediction dataset. Generalization was also tested on the PIMA Indians Diabetes Database.

The paper itself notes a limitation: K-Means *assumes spherical clusters and is sensitive to outliers*. That limitation motivates our project.

## Our project

### ML formulation

- **Task:** Binary classification with an unsupervised sample-selection step
- **Baseline:** K-Means focal selection. Its centroids naturally serve as focal points.
- **Proposed method:** DBSCAN focal selection. DBSCAN forms density-based clusters. We use a **medoid** as each cluster's focus and document how noise points are handled.

### Experiment design: change ONE factor

The only thing that differs between the two arms is how foci are selected: K-Means vs. DBSCAN. Everything else stays the same:

- Same cleaned and standardized data
- Same classifier and hyperparameter search
- Same label budget, stopping rule, and random seeds
- Same test set, untouched until final evaluation

### Evaluation

- **Primary metrics:** Recall and F1-score (diabetic class)
- **Secondary metrics:** Precision, accuracy, runtime, cluster count, DBSCAN noise rate

### Why the comparison is nontrivial

- DBSCAN needs `eps` and `min_samples` tuning.
- DBSCAN labels some points as noise and has no centroids. Scaling, feature encoding, and the rule for picking a representative point all affect the result.

### Open decisions to settle before coding

Proposed answers to all of these are in the [replication spec](replication_spec.md).

- [ ] How to treat DBSCAN noise points: exclude, assign to the nearest cluster, or treat each as its own focus?
- [ ] How to tune `eps` / `min_samples`: k-distance plot, grid search, or both?
- [ ] Which distance metric DBSCAN uses: cosine to match the paper, or Euclidean?
- [ ] How to encode categorical features (gender, smoking history)
- [ ] Which downstream classifier(s) to hold fixed across both arms
- [ ] Label budget / stopping proportion and the set of random seeds
- [ ] Train/test split ratio, and whether to stratify

## Dataset

Diabetes Prediction Dataset from [AbiDanalyst/Diabetes-Prediction-Analysis](https://github.com/AbiDanalyst/Diabetes-Prediction-Analysis), the same source the paper uses. See [../data/README.md](../data/README.md).
