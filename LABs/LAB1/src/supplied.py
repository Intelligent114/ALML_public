"""Supplied inputs for LAB1: linear regression and trustworthy evaluation.

This file contains no regression solver, main-task generation mechanism,
test data or reference scores. Python + NumPy + scikit-learn are sufficient.
"""
import csv
from pathlib import Path
import numpy as np

SENSOR_NAMES = ["load", "temperature", "setting", "load_near", "temperature_milli"] + [
    f"nuisance_{j:02d}" for j in range(24)
]


def load_development(data_dir=None):
    """Return development arrays sorted by unique row_id."""
    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "data"
    with (data_dir / "dev.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["row_id"]))
    ids = [int(row['row_id']) for row in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate row_id')
    return {
        "row_id": np.array(ids, dtype=int),
        "X": np.array([[float(r[name]) for name in SENSOR_NAMES] for r in rows]),
        "y": np.array([float(r["y"]) for r in rows]),
        "device_id": np.array([int(r["device_id"]) for r in rows]),
        "post_cycle_meter": np.array([float(r["post_cycle_meter"]) for r in rows]),
    }


def make_solver_check_data():
    """Return raw X:(80,7), y:(80,). Last two columns repeat col 0 and equal 1."""
    rng = np.random.default_rng(99)
    X = rng.normal(size=(80, 6))
    X[:, 5] = X[:, 0]
    y = 7 + X @ np.array([1, -2, 0, .5, 0, 1]) + rng.normal(0, .1, 80)
    return np.column_stack([X, np.ones(80)]), y

