#!/usr/bin/env python3
"""
Module to standardize tabular data using Scikit-learn
"""
from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes tabular data using StandardScaler
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
