"""Provided LAB1 workflow; students edit submission.py, not this file.

check: implementation checks. tune-lr: coarse/refined learning-rate trials.
compare: coarse/refined regularization selection on fixed training/validation.
final: use selected configuration, evaluate reserved local test ONCE, then
       refit all development rows and save final model. No network required.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import submission as S
from supplied import load_development, make_solver_check_data
from benchmark import baseline_predict, performance_score
from config_protocol import (read_config, check_selection, train_artifact, train_features,
                             predict_trained, GD_MAX_ITER, GD_TOL, TRAINING_PROTOCOL)

SEED = 20260918


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def splits(data, mode='group'):
    rng = np.random.default_rng(SEED)
    if mode == 'group':
        ids = rng.permutation(np.unique(data['device_id']))
        return tuple(np.flatnonzero(np.isin(data['device_id'], g))
                     for g in (ids[:48], ids[48:54], ids[54:]))
    train, val, test = splits(data, 'group')
    order = rng.permutation(np.concatenate([train, val]))
    return order[:480], order[480:], test


def fit(F, y, lam):
    prep = S.fit_preprocessor(F, y)
    w = S.solve_weights(S.transform(F, prep), y-prep['y_mean'], lam)
    return prep, w


def predict(F, prep, w):
    return prep['y_mean']+S.transform(F, prep)@w


def check(out):
    X, y = make_solver_check_data()
    prep = S.fit_preprocessor(X, y)
    Z, yc = S.transform(X, prep), y-prep['y_mean']
    lam = .1
    rows = []
    def record(name, value, tolerance):
        rows.append(dict(check=name, value=float(value), tolerance=tolerance,
                         passed=bool(np.isfinite(value) and value <= tolerance)))
    record('training_center', np.max(np.abs(Z.mean(0))), 1e-12)
    record('constant_column', np.max(np.abs(Z[:, -1])), 1e-12)
    record('standard_scale', np.max(np.abs(Z[:, :-1].std(0)-1)), 1e-12)
    Fnew = X[:3]+2
    record('reuse_train_statistics', np.max(np.abs(S.transform(Fnew, prep)-(Fnew-X.mean(0))/np.where(X.std(0)<1e-12, 1, X.std(0)))), 1e-12)
    raw = np.arange(3*29, dtype=float).reshape(3, 29)/10
    q = S.make_features(raw, 'quadratic')
    expected = np.column_stack([raw, raw[:, 0]**2, raw[:, 0]*raw[:, 1], raw[:, 1]**2])
    record('quadratic_features', np.max(np.abs(q-expected)), 1e-12)
    metrics = S.evaluate(np.array([1., 2., 3., 4.]), np.array([1., 2., 3., 8.]))
    record('metrics', max(abs(metrics[k]-v) for k, v in dict(mse=4., rmse=2., mae=1.).items()), 1e-12)
    w0 = np.linspace(-.2, .2, Z.shape[1])
    loss, grad = S.loss_gradient(Z, yc, w0, lam)
    expected_loss = np.mean((Z@w0-yc)**2)/2+lam*np.sum(w0*w0)/2
    record('objective_scaling', abs(loss-expected_loss), 1e-12)
    h = 1e-6
    num = np.zeros_like(w0)
    for j in range(len(w0)):
        direction = np.zeros_like(w0); direction[j] = h
        num[j] = (S.loss_gradient(Z, yc, w0+direction, lam)[0]-S.loss_gradient(Z, yc, w0-direction, lam)[0])/(2*h)
    record('gradient_finite_difference', np.max(np.abs(num-grad)), 1e-6)
    direct = S.solve_weights(Z, yc, lam)
    ref = np.linalg.solve(Z.T@Z+len(y)*lam*np.eye(Z.shape[1]), Z.T@yc)
    record('ridge_direct_prediction', np.max(np.abs(Z@(direct-ref))), 1e-8)
    ols = S.solve_weights(Z, yc, 0.)
    record('ols_duplicate_prediction', np.max(np.abs(Z@(ols-np.linalg.lstsq(Z, yc, rcond=None)[0]))), 1e-8)
    rate = 1/(np.linalg.eigvalsh(Z.T@Z/len(y)).max()+lam)
    w, history = S.train_gd(Z, yc, lam, rate)
    # A direct-solver shortcut can match the converged answer but must fail
    # these zero/one-update cases; max_iter is part of the student contract.
    zero, hzero = S.train_gd(Z, yc, lam, rate, max_iter=0, tol=1e-7)
    one, hone = S.train_gd(Z, yc, lam, rate, max_iter=1, tol=1e-7)
    first_step = rate * (Z.T @ yc / len(yc))
    short_error = max(float(np.max(np.abs(zero))), float(np.max(np.abs(one-first_step))))
    if np.shape(hzero) != (1, 3) or np.shape(hone) != (2, 3):
        short_error = float('inf')
    record('gd_prediction', max(float(np.max(np.abs(Z@(w-direct)))), short_error), 1e-5)
    record('gd_terminal_gradient', np.max(np.abs(S.loss_gradient(Z, yc, w, lam)[1])), 1e-7)
    record('gd_loss_increase', max(0., float(np.max(np.diff(history[:, 1])))), 1e-10)
    prep2, w2 = fit(X, y+100., lam)
    record('label_shift', np.max(np.abs(predict(X, prep2, w2)-predict(X, prep, direct)-100)), 1e-8)
    write_csv(out/'checks.csv', rows)
    write_csv(out/'gd_history.csv', [dict(iteration=int(t), objective=float(a), gradient_inf_norm=float(b)) for t, a, b in history])
    plt.figure(figsize=(6, 3.5))
    plt.plot(history[:, 0], history[:, 1]); plt.xlabel('Iteration'); plt.ylabel('Objective')
    plt.tight_layout(); plt.savefig(out/'loss_curve.png', dpi=160); plt.close()
    print(f'Checks passed: {sum(r["passed"] for r in rows)}/{len(rows)}')
    if not all(r['passed'] for r in rows):
        raise SystemExit('Some checks failed. Read checks.csv and fix submission.py.')


def tune_lr(data, out, extra_factor=None):
    """Same training objective/initialization; three coarse rates plus one refinement."""
    if (out/'config.json').exists():
        raise SystemExit('Model comparison already started; finish rate trials first. Use a fresh output directory only to correct an invalid experiment.')
    if (out/'test_metrics.csv').exists():
        raise SystemExit('Learning-rate experiments must precede final testing.')
    factors = [.01, 1., 2.1]
    if extra_factor is not None:
        if not np.isfinite(extra_factor) or not 0 < extra_factor < 2 or extra_factor in factors:
            raise ValueError('Choose a new finite factor in (0,2), different from 0.01 and 1.')
        if not (out/'lr_coarse.csv').exists():
            raise ValueError('Run tune-lr without --lr-factor first; inspect the coarse results.')
        factors.append(float(extra_factor))
    train, _, _ = splits(data)
    F = S.make_features(data['X'][train], 'quadratic')
    prep = S.fit_preprocessor(F, data['y'][train])
    Z, yc = S.transform(F, prep), data['y'][train]-prep['y_mean']
    lam = .1
    L = float(np.linalg.eigvalsh(Z.T@Z/len(yc)).max()+lam)
    direct = S.solve_weights(Z, yc, lam)
    optimum = float(S.loss_gradient(Z, yc, direct, lam)[0])
    rows, histories = [], []
    plt.figure(figsize=(7,4))
    for factor in factors:
        with np.errstate(over='ignore', invalid='ignore'):
            w, history = S.train_gd(Z, yc, lam, factor/L, max_iter=2000, tol=1e-7)
        history = np.asarray(history)
        last = history[-1]
        finite = bool(np.all(np.isfinite(w)) and np.all(np.isfinite(last)))
        diverged = not finite or last[1] > 1e6*max(1., history[0,1])
        reached = np.flatnonzero(np.isfinite(history[:,1]) &
                                 (np.abs(history[:,1]-optimum) <= 1e-6*max(1.,abs(optimum))))
        hit = int(history[reached[0],0]) if len(reached) and not diverged else None
        rows.append(dict(factor=factor, learning_rate=factor/L, iterations=int(last[0]),
                         status='diverged' if diverged else ('converged' if last[2]<=1e-7 else 'budget'),
                         final_objective=float(last[1]) if np.isfinite(last[1]) else '',
                         objective_gap=float(last[1]-optimum) if np.isfinite(last[1]) else '',
                         gradient_inf_norm=float(last[2]) if np.isfinite(last[2]) else '',
                         first_target_iteration=hit if hit is not None else '',
                         prediction_difference=float(np.max(np.abs(Z@(w-direct)))) if finite else ''))
        for t, objective, grad in history:
            histories.append(dict(factor=factor, iteration=int(t),
                                  objective=float(objective) if np.isfinite(objective) else '',
                                  gradient_inf_norm=float(grad) if np.isfinite(grad) else ''))
        gap = np.maximum(np.abs(history[:,1]-optimum),1e-14)
        valid = np.isfinite(gap)
        plt.semilogy(history[valid,0],gap[valid],label=f'factor={factor:g}')
    plt.xlabel('Iteration'); plt.ylabel('Absolute objective gap'); plt.legend()
    plt.tight_layout(); plt.savefig(out/'lr_curves.png',dpi=160); plt.close()
    write_csv(out/'lr_tuning.csv',rows)
    write_csv(out/'lr_history.csv',histories)
    if extra_factor is None:
        write_csv(out/'lr_coarse.csv',rows)
        (out/'lr_config.json').unlink(missing_ok=True)
    else:
        eligible=[r for r in rows if r['first_target_iteration'] != '' and r['status'] != 'diverged']
        if not eligible:
            raise ValueError('No learning rate reached the fixed target; check the implementation.')
        chosen=min(eligible,key=lambda r:(r['first_target_iteration'],r['factor']))
        write_json(out/'lr_config.json',dict(extra_factor=extra_factor,chosen_factor=chosen['factor'],
                   chosen_learning_rate=chosen['learning_rate'],L=L,lam=lam,basis='quadratic',
                   objective_target=1e-6*max(1.,abs(optimum)),selection='fewest iterations to fixed objective gap'))
    print('Learning-rate comparison saved; divergent trials are recorded, not treated as task failure.')


def compare(data, out, extra_lambdas=None):
    if (out/'test_metrics.csv').exists():
        raise SystemExit('Configuration is frozen after final test; do not rerun model selection.')
    lambdas = [0., .1, 1.]
    if extra_lambdas is not None:
        extra_lambdas = list(map(float,extra_lambdas))
        if (len(extra_lambdas)!=2 or len(set(extra_lambdas))!=2 or
            any(not np.isfinite(x) or x<=0 or x in lambdas for x in extra_lambdas)):
            raise ValueError('Provide two distinct finite positive NEW regularization parameters.')
        if not (out/'coarse_comparison.csv').exists():
            raise ValueError('Run compare without --extra-lambdas first; inspect its validation results.')
        lambdas=sorted(lambdas+extra_lambdas)
    if not (out/'lr_config.json').exists():
        raise ValueError('Complete learning-rate refinement before comparing models.')
    rate_selection = json.loads((out/'lr_config.json').read_text(encoding='utf-8'))
    factors = sorted([.01, 1., float(rate_selection['extra_factor'])])
    if len(set(factors)) != 3 or any(not 0 < c < 2 for c in factors):
        raise ValueError('Invalid learning-rate refinement record.')
    train, val, test = splits(data)
    write_csv(out/'split.csv', [dict(row_id=int(data['row_id'][i]), device_id=int(data['device_id'][i]), split=name)
                              for name, idx in zip(('train', 'validation', 'test'), (train, val, test)) for i in idx])
    rows, joint, predictions, audit = [], [], [], []
    mean_pred = np.full(len(val), data['y'][train].mean())
    rows.append(dict(candidate='mean', basis='none', lam=0., lr_factor='', learning_rate='',
                     iterations=0, status='comparison_only',
                     train_rmse=S.evaluate(data['y'][train],np.full(len(train),data['y'][train].mean()))['rmse'],
                     coefficient_norm=0., **S.evaluate(data['y'][val], mean_pred)))
    for basis in ('linear', 'quadratic'):
        F = S.make_features(data['X'], basis)
        for lam in lambdas:
            candidates = []
            for factor in factors:
                prep, w, info = train_features(S, F[train], data['y'][train], lam, factor)
                pred = predict(F[val], prep, w)
                name = f'{basis}_{lam:.17g}_c{factor:.17g}'
                row = dict(candidate=name, basis=basis, lam=lam, lr_factor=factor,
                           learning_rate=info['learning_rate'], iterations=info['iterations'], status=info['status'],
                           train_rmse=S.evaluate(data['y'][train], predict(F[train], prep, w))['rmse'],
                           coefficient_norm=float(np.linalg.norm(w)), **S.evaluate(data['y'][val], pred))
                joint.append(row)
                candidates.append((row, pred))
            chosen, pred = min(candidates, key=lambda pair: (pair[0]['rmse'], pair[0]['lr_factor']))
            rows.append(chosen)
            predictions.extend(dict(candidate=chosen['candidate'], row_id=int(data['row_id'][i]),
                                    y_true=float(data['y'][i]), y_pred=float(v)) for i,v in zip(val,pred))
    predictions.extend(dict(candidate='mean', row_id=int(data['row_id'][i]), y_true=float(data['y'][i]), y_pred=float(p)) for i,p in zip(val,mean_pred))
    best = min(joint, key=lambda r:(r['rmse'], ('linear','quadratic').index(r['basis']), -r['lam'], r['lr_factor']))
    fingerprint = hashlib.sha256(data['X'].tobytes()+data['y'].tobytes()+data['row_id'].tobytes()).hexdigest()
    write_json(out/'config.json',dict(candidate=best['candidate'],basis=best['basis'],lam=best['lam'],lr_factor=best['lr_factor'],
                seed=SEED,selection_metric='validation_rmse',data_sha256=fingerprint,
                refinement_complete=extra_lambdas is not None,extra_lambdas=extra_lambdas,
                rate_factors=factors,training_protocol=TRAINING_PROTOCOL,max_iter=GD_MAX_ITER,tol=GD_TOL))
    write_csv(out/'comparison.csv', rows)
    write_csv(out/'joint_comparison.csv', joint)
    if extra_lambdas is None:
        write_csv(out/'coarse_comparison.csv',rows)
        write_csv(out/'coarse_joint_comparison.csv',joint)
    write_csv(out/'validation_predictions.csv', predictions)
    Fbase = S.make_features(data['X'], 'quadratic')
    for name, mode, post in [('group', 'group', False), ('random', 'random', False), ('post', 'group', True)]:
        tr, va, _ = splits(data, mode)
        F = np.column_stack([Fbase, data['post_cycle_meter']]) if post else Fbase
        prep, w, _ = train_features(S, F[tr], data['y'][tr], .1, 1.)
        pred = predict(F[va], prep, w)
        overlap = len(set(data['device_id'][tr]) & set(data['device_id'][va]))
        audit.append(dict(condition=name, train_rows=len(tr), validation_rows=len(va), shared_devices=overlap,
                          **S.evaluate(data['y'][va], pred)))
    write_csv(out/'audit.csv', audit)
    plt.figure(figsize=(7, 3.8))
    labels=['mean' if r['basis']=='none' else f"{r['basis']}/{r['lam']:g}" for r in rows]
    plt.bar(labels, [r['rmse'] for r in rows])
    plt.xticks(rotation=30, ha='right'); plt.ylabel('Validation RMSE')
    plt.tight_layout(); plt.savefig(out/'comparison.png', dpi=160); plt.close()
    fig, axes=plt.subplots(2,2,figsize=(9,6))
    for col,basis in enumerate(('linear','quadratic')):
        selected=sorted((r for r in joint if r['basis']==basis and r['lr_factor']==1.),key=lambda r:r['lam'])
        x=[r['lam'] for r in selected]
        axes[0,col].plot(x,[r['train_rmse'] for r in selected],'o-',label='Train RMSE')
        axes[0,col].plot(x,[r['rmse'] for r in selected],'s-',label='Validation RMSE')
        axes[0,col].set_title(basis+' (factor=1, GD budget=200)'); axes[0,col].legend(); axes[0,col].set_ylabel('RMSE')
        axes[1,col].plot(x,[r['coefficient_norm'] for r in selected],'o-')
        axes[1,col].set_ylabel('Coefficient L2 norm')
        for ax in axes[:,col]:
            ax.set_xscale('symlog',linthresh=min(v for v in x if v>0)/2)
            ax.set_xlabel('lambda')
    fig.tight_layout(); fig.savefig(out/'regularization_curves.png',dpi=160); plt.close(fig)
    print('Selected:', best['candidate'], '; local test labels have not been used for evaluation.')


def final(data, out):
    if (out/'test_metrics.csv').exists():
        raise SystemExit('Test result already exists; do not repeatedly select using local test scores.')
    config = json.loads((out/'config.json').read_text(encoding='utf-8'))
    if not config.get('refinement_complete') or not (out/'lr_config.json').exists():
        raise SystemExit('Complete both learning-rate and regularization refinement before final testing.')
    fingerprint = hashlib.sha256(data['X'].tobytes()+data['y'].tobytes()+data['row_id'].tobytes()).hexdigest()
    if config['seed'] != SEED or config['data_sha256'] != fingerprint:
        raise ValueError('Data/split differs from the selected configuration.')
    selected = read_config(S)
    rate_selection = json.loads((out/'lr_config.json').read_text(encoding='utf-8'))
    check_selection(selected, config, rate_selection)
    tr, va, te = splits(data); train = np.concatenate([tr, va])
    local_model = train_artifact(S, data['X'][train], data['y'][train], selected)
    pred = predict_trained(S, local_model, data['X'][te])
    write_json(out/'local_training.json', {key:local_model[key] for key in
               ('training_protocol','learning_rate','L','iterations','status','max_iter','tol','lr_factor','lam','basis')})
    np.savez(out/'final_model.npz', **train_artifact(S, data['X'], data['y'], selected))
    write_json(out/'submitted_config.json', selected)
    write_csv(out/'test_metrics.csv', [dict(candidate=config['candidate'], **S.evaluate(data['y'][te], pred))])
    baseline_pred = baseline_predict(data['X'][train],data['y'][train],data['X'][te])
    student_rmse=float(np.sqrt(np.mean((pred-data['y'][te])**2)))
    baseline_rmse=float(np.sqrt(np.mean((baseline_pred-data['y'][te])**2)))
    write_csv(out/'performance_preview.csv',[dict(student_rmse=student_rmse,
              baseline_rmse=baseline_rmse,preview_points=performance_score(student_rmse,baseline_rmse),
              note='local test preview only; official 10 points use teacher held-out devices')])
    write_csv(out/'test_predictions.csv', [dict(row_id=int(data['row_id'][i]), y_true=float(data['y'][i]), y_pred=float(p)) for i, p in zip(te, pred)])
    print('Local test evaluated once; final model refit on all 600 rows.')


def predict_saved(model_path, raw_X):
    with np.load(model_path, allow_pickle=False) as model:
        basis = str(model['basis'])
        if basis == 'none':
            return np.full(len(raw_X), float(model['y_mean']))
        return predict(S.make_features(raw_X, basis), model, model['w'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['check', 'tune-lr', 'compare', 'final'])
    parser.add_argument('--lr-factor',type=float,help='One student-selected new rate multiplier in (0,2).')
    parser.add_argument('--extra-lambdas',type=float,nargs=2,help='Two student-selected positive NEW regularization parameters.')
    parser.add_argument('--data', default=str(Path(__file__).resolve().parents[1]/'data'))
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'check':
        check(out)
    else:
        data = load_development(args.data)
        if args.stage=='tune-lr':
            tune_lr(data,out,args.lr_factor)
        elif args.stage=='compare':
            compare(data,out,args.extra_lambdas)
        else:
            final(data,out)


if __name__ == '__main__':
    main()
