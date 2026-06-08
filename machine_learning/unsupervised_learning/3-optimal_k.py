#!/usr/bin/env python3
"""
This module provides a function to determine the optimal number of clusters
using Inertia and Silhouette scores.
"""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """
    Evaluates K-Means clustering quality to find the optimal k.

    Parameters:
    X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
    max_clusters (int): Maximum number of clusters to evaluate (>=2)
    random_state (int): Random seed for reproducibility

    Returns:
    list[int]: Evaluated cluster numbers
    list[float]: Inertia values corresponding to each cluster number
    list[float]: Silhouette scores corresponding to each cluster number
    """
    ks = []
    inertia_values = []
    silhouette_values = []

    for k in range(2, max_clusters + 1):
        model = K_Means(X, n_clusters=k, random_state=random_state)
        labels = model.labels_

        ks.append(k)
        inertia_values.append(model.inertia_)

        score = metrics.silhouette_score(X, labels, random_state=random_state)
        silhouette_values.append(score)

    return ks, inertia_values, silhouette_values
