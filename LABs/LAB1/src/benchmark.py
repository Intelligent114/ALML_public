"""Provided fixed sklearn baseline and public performance scoring rule.

This code must not be modified or called inside the seven student functions.
The baseline is independent of the student implementation and selected lambda.
"""
import numpy as np
from sklearn.linear_model import Ridge


def features(X,basis='quadratic'):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or X.shape[1]!=29 or not np.isfinite(X).all():
        raise ValueError('Expected finite raw sensor matrix of shape (n,29).')
    if basis=='linear':
        return X.copy()
    if basis!='quadratic':
        raise ValueError('Unsupported feature basis')
    return np.column_stack([X,X[:,0]**2,X[:,0]*X[:,1],X[:,1]**2])


def baseline_predict(train_X,train_y,new_X):
    F=features(train_X)
    mu=F.mean(0); scale=F.std(0,ddof=0)
    scale=np.where(scale<1e-12,1.,scale)
    y_mean=float(np.mean(train_y))
    model=Ridge(alpha=len(train_y)*.1,fit_intercept=False,solver='svd')
    model.fit((F-mu)/scale,np.asarray(train_y)-y_mean)
    return y_mean+model.predict((features(new_X)-mu)/scale)


def performance_score(student_rmse,baseline_rmse):
    """Return 0..10 points. Equal/better within numerical tolerance gets 10."""
    if not np.isfinite(baseline_rmse) or baseline_rmse<0:
        raise ValueError('Invalid baseline score; evaluator must be repaired.')
    if not np.isfinite(student_rmse) or student_rmse<0:
        return 0.
    tolerance=1e-8*max(1.,baseline_rmse)
    if student_rmse<=baseline_rmse+tolerance:
        return 10.
    return float(10*baseline_rmse/student_rmse)


def predict_artifact(model_path,raw_X):
    """Teacher-controlled prediction from submitted arrays, no pickle/student code."""
    with np.load(model_path,allow_pickle=False) as model:
        basis=str(model['basis']); y_mean=float(model['y_mean'])
        if not np.isfinite(y_mean):
            raise ValueError('Nonfinite model mean')
        if basis=='none':
            return np.full(len(raw_X),y_mean)
        F=features(raw_X,basis); p=F.shape[1]
        mu,scale,w=(model[k] for k in ('mu','scale','w'))
        if any(a.shape!=(p,) or not np.isfinite(a).all() for a in (mu,scale,w)) or (scale<=0).any():
            raise ValueError('Invalid saved model arrays')
        return y_mean+(F-mu)/scale@w
