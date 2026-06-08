#!/usr/bin/env python3
"""
This module provides a function to standardize tabular data.
Standardization scales features to have a mean of 0 and variance of 1.
"""
from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes tabular data using Scikit-learn's StandardScaler.

    Parameters:
    X (numpy.ndarray): Tabular data of shape (n_samples, n_features)

    Returns:
    numpy.ndarray: The standardized version of the input data.
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
