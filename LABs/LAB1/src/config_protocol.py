"""Provided configuration validation and final training shared by local/OJ runners."""
import numbers
import numpy as np

GD_MAX_ITER = 200
GD_TOL = 1e-7
TRAINING_PROTOCOL = "gd200-v1"


def read_config(submission):
    raw = submission.get_config()
    if not isinstance(raw, dict) or set(raw) != {'basis', 'lambda', 'lr_factor'}:
        raise ValueError('get_config() must return basis, lambda, lr_factor only.')
    basis = raw['basis']
    if not isinstance(basis, str) or basis not in ('linear', 'quadratic'):
        raise ValueError('Choose basis from linear, quadratic after validation; mean is a comparison only.')
    values = {}
    for key in ('lambda', 'lr_factor'):
        value = raw[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value):
            raise ValueError(f'{key} must be a finite real number, not a placeholder.')
        values[key] = float(value)
    if values['lambda'] < 0:
        raise ValueError('lambda must be nonnegative.')
    if not 0 < values['lr_factor'] < 2:
        raise ValueError('Selected lr_factor must be in (0,2).')
    return dict(basis=basis, **values)


def check_selection(config, model_selection, rate_selection):
    if (config['basis'] != model_selection['basis'] or
            config['lambda'] != model_selection['lam'] or
            config['lr_factor'] != model_selection['lr_factor'] or
            config['lr_factor'] not in (.01, 1., rate_selection['extra_factor']) or
            model_selection.get('training_protocol') != TRAINING_PROTOCOL or
            model_selection.get('rate_factors') != sorted([.01, 1., float(rate_selection['extra_factor'])])):
        raise ValueError('get_config() differs from config.json / lr_config.json. '
                         'Copy the selected values exactly before final testing.')


def train_features(submission, F, y, lam, factor):
    """Zero-start, finite-budget GD. Direct solvers are not used here."""
    prep = submission.fit_preprocessor(F, y)
    Z = submission.transform(F, prep)
    L = float(np.linalg.eigvalsh(Z.T @ Z / len(y)).max() + lam)
    rate = factor / L if L > 0 else factor
    w, history = submission.train_gd(Z, y-prep['y_mean'], lam, rate,
                                     max_iter=GD_MAX_ITER, tol=GD_TOL)
    w, history = np.asarray(w), np.asarray(history)
    if (w.shape != (Z.shape[1],) or history.ndim != 2 or history.shape[1] != 3
            or not 1 <= len(history) <= GD_MAX_ITER+1
            or not np.isfinite(w).all() or not np.isfinite(history).all()):
        raise ValueError('Invalid/nonfinite GD weights or history.')
    if not np.array_equal(history[:, 0], np.arange(len(history))):
        raise ValueError('GD history must include each step starting at zero.')
    if history[-1, 1] > 1e6 * max(1., history[0, 1]):
        raise ValueError('Final-training gradient descent diverged.')
    status = 'converged' if history[-1, 2] <= GD_TOL else 'budget'
    if status == 'budget' and history[-1, 0] != GD_MAX_ITER:
        raise ValueError('GD stopped before tolerance or iteration budget.')
    info = dict(learning_rate=rate, L=L, iterations=int(history[-1, 0]),
                status=status, gradient_inf_norm=float(history[-1, 2]),
                training_protocol=TRAINING_PROTOCOL, max_iter=GD_MAX_ITER, tol=GD_TOL)
    return prep, w, info


def train_artifact(submission, X, y, config):
    """Use only supplied training rows; recompute L for each fit."""
    F = submission.make_features(X, config['basis'])
    prep, w, info = train_features(submission, F, y, config['lambda'], config['lr_factor'])
    return dict(basis=config['basis'], lam=config['lambda'],
                lr_factor=config['lr_factor'], mu=np.asarray(prep['mu']),
                scale=np.asarray(prep['scale']), y_mean=float(prep['y_mean']),
                w=w, **info)


def predict_trained(submission, model, X):
    F = submission.make_features(X, model['basis'])
    return model['y_mean'] + submission.transform(F, model) @ model['w']
