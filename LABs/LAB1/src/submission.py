"""LAB1 student implementation. Fill the seven TODO functions and get_config() after tuning.

Use NumPy; sklearn estimators and automatic differentiation are not allowed
inside these functions. Input arrays are floating point, finite, and nonempty.
See lab1.pdf for formulas, tasks, and grading. run.py supplies orchestration.
"""
import numpy as np

def get_config():
    """Return your frozen selection; no file reads or parameter search here.

    basis: 'linear' or 'quadratic'; the mean baseline is not submittable.
    lambda: finite nonnegative regularization strength.
    lr_factor: joint-validation selection in config.json (not the raw rate).
    Fill these only after both refinement experiments. No extra score.
    """
    return {"basis": None, "lambda": None, "lr_factor": None}



def make_features(X, basis):
    """X: (n,29), SENSOR_NAMES order. Return (n,29) for 'linear';
    for 'quadratic', append load**2, load*temperature, temperature**2 -> (n,32).
    Do not mutate X. No fitting/statistics here.
    """
    raise NotImplementedError("Task 1.1(a): make_features")


def fit_preprocessor(F, y):
    """F: (n,p), y: (n,). Return dict mu:(p,), scale:(p,), y_mean: scalar.
    Population standard deviation (ddof=0); replace scale < 1e-12 by 1.
    Fit ONLY on the supplied training rows. No changes to F/y.
    """
    raise NotImplementedError("Task 1.1(b): fit_preprocessor")


def transform(F, prep):
    """Return (F-prep['mu'])/prep['scale']; never refit statistics."""
    raise NotImplementedError("Task 1.1(b): transform")


def solve_weights(Z, yc, lam):
    """Z:(n,p), yc:(n,), lam>=0. Return w:(p,).
    lam==0: use np.linalg.lstsq(..., rcond=None).
    lam>0: solve the Ridge normal equations for the objective in lab1.pdf.
    np.linalg.solve allowed; explicit matrix inversion is unnecessary.
    """
    raise NotImplementedError("Task 1.2: solve_weights")


def loss_gradient(Z, yc, w, lam):
    """Return (objective: scalar, gradient:(p,)).
    Objective = squared residual / (2*n) + lam * squared norm(w) / 2.
    """
    raise NotImplementedError("Task 1.3: loss_gradient")


def train_gd(Z, yc, lam, learning_rate, max_iter=20000, tol=1e-7):
    """Full batch GD from zeros, calling loss_gradient. Return (w, history).
    history: float array (k,3), columns [iteration, objective, gradient_inf_norm].
    Log initial state and each updated state. Stop when gradient_inf_norm<=tol,
    or after max_iter updates. Return the last logged coefficients.
    Also log and stop if objective/gradient is nonfinite or objective exceeds
    1e6 * max(1, initial_objective); preserve this failed-trial evidence.
    """
    raise NotImplementedError("Task 1.4: train_gd")


def evaluate(y, pred):
    """Return dict with float mse, rmse, mae. y/pred: matching (m,) arrays.
    Evaluate prediction error ONLY; do not add any regularization penalty.
    """
    raise NotImplementedError("Task 1.5: evaluate")
