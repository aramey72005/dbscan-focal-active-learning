"""Focal sample selection (replication spec §3, and later §4).

Every method takes the standardized feature matrix of the training pool and
returns the row positions of exactly `n_foci` foci. The rest of the pipeline
does not care which method produced them; that is the one thing that differs
between the two experiment arms.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

N_FOCI = 200  # paper p. 11, 17


def kmeans_foci(Z, n_foci=N_FOCI, seed=0):
    """Paper's method (Eq. 2-5): cosine K-Means, then the real sample closest to each centroid.

    Rows are L2-normalized first; on unit vectors, Euclidean K-Means ranks points
    the same way cosine similarity does.
    """
    X = normalize(Z)
    km = KMeans(n_clusters=n_foci, n_init=10, random_state=seed).fit(X)
    centroids = normalize(km.cluster_centers_)

    foci = []
    for c in range(n_foci):
        members = np.flatnonzero(km.labels_ == c)
        foci.append(members[np.argmax(X[members] @ centroids[c])])
    return np.asarray(foci)


METHODS = {"kmeans": kmeans_foci}


def select_foci(method, Z, n_foci=N_FOCI, seed=0):
    foci = METHODS[method](Z, n_foci=n_foci, seed=seed)
    if len(set(foci)) != n_foci:
        raise ValueError(f"{method} returned {len(set(foci))} unique foci, expected {n_foci}")
    return foci
