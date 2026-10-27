# DBSCAN pre-check: findings (replication spec §4)

> **Stale data:** this was run *before* preprocessing was changed to follow the paper, which now drops "No Info" rows, duplicates and IQR outliers, and uses a 70/15/15 split. The numbers below are from the earlier 6-category, outlier-included data. Rerun `src/dbscan_precheck.py` before using them. The main finding (clusters follow the binary features) is likely to hold, because hypertension and heart_disease are unchanged.

Run on the 10,000-row stratified dev subset (from the training pool), using standardized, L2-normalized features as in spec §3–4.
Reproduce with `python src/preprocess.py`, then `python src/dbscan_precheck.py` (run from `src/`).

Files: `k_distance.png`, `dbscan_grid.csv` (grid around the knee), `dbscan_fine_sweep.csv` (finer eps steps through the transition range), `hdbscan_check.csv`, `summary.json`.

## Bottom line

**DBSCAN finds no real density-based clusters in this data. With the current spec, it would not give a meaningful comparison against K-Means.** The team needs to decide how to change §4 before the DBSCAN arm is built.

## What we saw

1. **The k-distance plot has no bend.** The curve rises smoothly for k = 8, 16 and 32, and the only "knee" is the last few outliers at the very end (`k_distance.png`). There's no clear `eps` that separates dense regions from sparse ones.

2. **DBSCAN jumps from "mostly noise" to "one giant cluster" with no useful setting in between.** For example, with `min_samples = 16`:

   | eps | clusters | noise | largest cluster | silhouette |
   |---|---|---|---|---|
   | 0.20 | 33 | 77% | 6% | 0.22 |
   | 0.26 | 40 | 39% | 27% | −0.12 |
   | 0.30 | 21 | 19% | 41% | −0.12 |
   | 0.34 | 8 | 6% | 50% | 0.08 |
   | 0.40 | 5 | 0.5% | 89% | 0.00 |

   **The spec's selection rule** (noise ≤ 10%, at least 20 clusters, then best silhouette):
   - Nothing passes in the main grid.
   - In the finer sweep, only `min_samples` 4 and 8 pass, around eps ≈ 0.24–0.28. Those settings have **negative silhouette** (−0.23 to −0.30) and one cluster holding about 45% of the points.
   - Silhouette below zero means the clusters are worse than a random split.

3. **The clusters just follow the binary features.** At eps = 0.40 the four clusters are exactly:

   | hypertension | heart_disease | share of points |
   |---|---|---|
   | no | no | 89% |
   | yes | no | 6% |
   | no | yes | 3% |
   | yes | yes | 1% |

   At eps = 0.30, the big cluster also splits by gender. DBSCAN is recovering the encoded categories, not any structure in the data. (K-Means with k = 200 also mostly follows categories: 71% stratum purity, meaning 71% of points share the most common category combination in their cluster.)

4. **Diabetic patients sit in the sparse regions.** Across the grid, noise points are **15–20% diabetic**, versus **3–8%** in clusters (the overall rate is 8.5%). The spec never picks noise points as foci, so DBSCAN's foci would systematically **under-sample diabetics**. That works against our hypothesis that DBSCAN improves recall on the diabetic class.

5. **The 200-foci rule would turn DBSCAN into farthest-point sampling.** DBSCAN never gives 200 clusters at a reasonable noise rate. That forces the "C < 200" path in §4. With half or more of the points in one cluster, about 100 of the 200 foci would come from farthest-point top-up inside that one cluster. The result would mostly measure farthest-point sampling, not DBSCAN.

6. **HDBSCAN doesn't fix it.** It handles regions of different density without a single eps, but it finds the same 3–8 category clusters, with about 89% of points in one (`hdbscan_check.csv`).

**Caveat:** this is the 10,000-row dev subset. The full pool (about 68,000 rows) is denser, so the eps values will shift. But the structure behind these results (binary features that split the data into separate groups, with continuous features in between) doesn't depend on sample size.

## Options for the team

| Option | What changes | Trade-off |
|---|---|---|
| **A. Cluster on continuous features only** (age, BMI, HbA1c, glucose). Leave the categorical features out of clustering; they still feed the classifier and scoring. | §4 input space, and §3 too, to keep the two methods comparable | The most likely way to find real density structure. But it departs from the paper's 8-feature clustering, and K-Means must change too. |
| **B. Cluster separately within each category stratum** (for example, per hypertension × heart_disease), giving each stratum foci in proportion to its size | §4 | Uses the categories directly instead of letting DBSCAN rediscover them. More complex to build. |
| **C. Keep the spec, but make noise points eligible as foci** (for example, add the highest-scoring noise points) | §4 noise row | Fixes the diabetic under-sampling. Still no real clusters. |
| **D. Keep DBSCAN as specified and report the negative result** | Add a silhouette > 0 requirement to the selection rule, and treat R7 as a main run | An honest null result, but a weak project if this is the only finding. |

**Suggestion:** test A on the dev subset first (a quick rerun of this script with a different feature list). If it shows real clusters, adopt A for both methods and update the spec through a PR.
