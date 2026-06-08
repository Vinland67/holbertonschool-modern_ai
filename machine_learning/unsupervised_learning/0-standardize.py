#!/usr/bin/env python3
"""
This module provides a function to standardize tabular data.
"""
from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes tabular data using Scikit-learn's StandardScaler.
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
