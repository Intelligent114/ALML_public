"""LAB3. Submit this file only. Implement with NumPy; see lab3.pdf."""
import numpy as np


def get_config():
    """Fixed choices shared by all instances; representation holds your JSON settings."""
    return {"clusters": 3, "representation": {}}


def fit_pca(X, retain):
    """Return training mean (d,) and orthonormal axes (d,r), retaining >= retain variance."""
    raise NotImplementedError


def lloyd_step(X, centers):
    """Return assignments to OLD centers and one matrix of updated centers."""
    raise NotImplementedError


def label_clusters(X, centers, y):
    """Return one integer class per center. Unlabelled records have y=-1."""
    raise NotImplementedError


def learn_representation(X, pairs, settings):
    """Use training data and pairs to return a mean (d,) and linear map (d,r)."""
    raise NotImplementedError
