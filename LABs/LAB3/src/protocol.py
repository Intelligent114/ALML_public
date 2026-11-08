"""Shared single-observation classifier and reproducible training protocol."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import importlib.util
import json
import numpy as np

SHIFTS = ("same", "neutral", "reverse")
FORMS = ("single", "double")
MODES = ("raw_single", "pca_single", "raw_multi", "paired_single", "paired_multi", "shuffled_multi")
RESTARTS, MAX_STEPS, FIT_SEED = 5, 60, 31415
AVERAGE_TARGET, WORST_TARGET = 0.945, 0.925


def load_student(path):
    spec = importlib.util.spec_from_file_location("student_impl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(student):
    cfg = student.get_config()
    if not isinstance(cfg, dict) or set(cfg) != {"clusters", "representation"}:
        raise ValueError("get_config requires clusters and representation")
    if type(cfg["clusters"]) is not int or not 2 <= cfg["clusters"] <= 18:
        raise ValueError("clusters must be an integer in [2,18]")
    if not isinstance(cfg["representation"], dict):
        raise ValueError("representation must be a JSON dictionary")
    if len(json.dumps(cfg, allow_nan=False)) > 4096:
        raise ValueError("configuration too large")
    return cfg


def distances(X, C):
    return np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)


def multi_centers(student, Z, y, k, seed):
    rng = np.random.default_rng(seed)
    best_loss = np.inf
    for _ in range(RESTARTS):
        centers = [Z[rng.integers(len(Z))]]
        near = np.sum((Z - centers[0]) ** 2, axis=1)
        for _ in range(1, k):
            idx = rng.choice(len(Z), p=near / near.sum()) if near.sum() else 0
            centers.append(Z[idx])
            near = np.minimum(near, np.sum((Z - Z[idx]) ** 2, axis=1))
        C = np.stack(centers)
        for _ in range(MAX_STEPS):
            _, updated = student.lloyd_step(Z, C.copy())
            updated = np.asarray(updated, dtype=float)
            if updated.shape != C.shape or not np.isfinite(updated).all():
                raise ValueError("invalid updated centers")
            stationary = np.allclose(updated, C, atol=1e-9, rtol=0)
            C = updated
            if stationary:
                break
        loss = distances(Z, C).min(axis=1).sum()
        if loss < best_loss:
            best_loss, best = loss, C.copy()
    return best, student.label_clusters(Z, best.copy(), y)


def validate_model(model, d=64):
    mu, P, C, labels = (np.asarray(v) for v in model)
    if P.ndim != 2 or P.shape[0] != d or not 1 <= P.shape[1] <= d or mu.shape != (d,):
        raise ValueError("invalid projection or mean shape")
    if C.ndim != 2 or not 1 <= len(C) <= 18 or C.shape[1] != P.shape[1]:
        raise ValueError("invalid prototype shape")
    if labels.shape != (len(C),) or labels.dtype.kind not in "iu" or not np.isin(labels, [0,1,2]).all():
        raise ValueError("invalid prototype labels")
    for v in (mu, P, C):
        if v.dtype.kind not in "fiu" or not np.isfinite(v).all():
            raise ValueError("invalid model coefficients")
    return mu, P, C, labels


def fit(student, X, y, pairs, mode="paired_multi", seed=FIT_SEED):
    if mode not in MODES:
        raise ValueError("unknown experiment")
    cfg = config(student) if mode not in ("raw_single", "pca_single") else None
    if mode == "shuffled_multi":
        pairs = pairs.copy()
        # Sattolo cycle: no object retains its true partner.
        rng = np.random.default_rng(973)
        for j in range(len(pairs)-1, 0, -1):
            i = rng.integers(j)
            pairs[[i,j], 1] = pairs[[j,i], 1]
    if mode.startswith("raw"):
        mu, P = np.zeros(X.shape[1]), np.eye(X.shape[1])
    elif mode == "pca_single":
        mu, P = student.fit_pca(X, 0.90)
    else:
        mu, P = student.learn_representation(X, pairs, cfg["representation"])
    mu, P = np.asarray(mu), np.asarray(P)
    Z = (X-mu) @ P
    if mode.endswith("_single"):
        C = np.stack([Z[y == c].mean(axis=0) for c in range(3)])
        labels = np.arange(3)
    else:
        C, labels = multi_centers(student, Z, y, cfg["clusters"], seed)
    return validate_model((mu, P, C, labels), X.shape[1])


def predict(model, X):
    mu, P, C, labels = model
    with np.errstate(over="raise", invalid="raise"):
        Z = (X-mu) @ P
        return labels[distances(Z, C).argmin(axis=1)]


def evaluate(model, tests, truth):
    matrices = []
    for X, y in zip(tests, truth, strict=True):
        pred = predict(model, X)
        matrices.append(np.bincount(3*y + pred, minlength=9).reshape(3,3))
    mats = np.array(matrices)
    recall = (mats.diagonal(axis1=1, axis2=2) / mats.sum(axis=2)).mean(axis=1)
    return recall, mats


def summarize(rows):
    groups = []
    for form in FORMS:
        selected = [r["recall"] for r in rows if r["form"] == form]
        if not selected:
            raise ValueError("missing form")
        groups.extend(np.mean(selected, axis=0).tolist())
    a, w = float(np.mean(groups)), min(groups)
    # Averaging recalls can put an exact threshold one rounding unit below it.
    points = sum(10.0 if abs(val-cut) <= 1e-14 else 10 * np.clip((val-1/3)/(cut-1/3), 0, 1)
                 for val, cut in ((a, AVERAGE_TARGET), (w, WORST_TARGET)))
    return dict(groups=groups, average=a, worst=w, points=float(points))
