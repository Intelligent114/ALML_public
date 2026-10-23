"""Supplied experiment CLI. Execute from LAB2 root. Do not edit."""
import argparse,csv,json,platform,time,hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from protocol import load_submission,digest,validate_config,train,predict,SEED
from checks import check
ROOT=Path(__file__).resolve().parents[1]
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def cases():return [dict(np.load(ROOT/'data'/f'signal{i}.npz',allow_pickle=False)) for i in range(3)]
def configuration(scale=1,lr=.001,lam=0):return dict(scale=float(scale),lr=float(lr),lam=float(lam))
def record_fit(S,data,c,out,raw=False,width=64,sparse=False):
    xx,yy=(data['X_sparse'],data['y_sparse']) if sparse else (data['X_train'],data['y_train'])
    es={'validation':(data['X_val'],data['y_val']),'diagnostic':(data['X_grid'],data['y_grid'],data['freqs'])}
    m,rec=train(S,xx,yy,c,es,raw=raw,width=width)
    out.mkdir(parents=True,exist_ok=True);save(out/'history.json',rec)
    hist=rec['history'];grid=data['X_grid'].ravel()
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    for ax,step in zip(axes.ravel(),(0,50,200,800)):
        ax.plot(grid,data['y_grid'].ravel(),label='target',lw=1)
        ax.plot(grid,np.array(rec['snapshots']['diagnostic'][str(step)]).ravel(),label='prediction',lw=1)
        ax.set_title(f'Update {step}');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'predictions.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3))
    for j,k in enumerate(data['freqs']):ax.semilogy([r['step'] for r in hist],[max(r['diagnostic_frequency_error'][j],1e-14) for r in hist],label=f'k={k}')
    ax.set(xlabel='Update',ylabel='Projected squared residual');ax.legend();fig.tight_layout();fig.savefig(out/'frequencies.png',dpi=140);plt.close(fig)
    np.savez_compressed(out/'predictions.npz',X=data['X_grid'],target=data['y_grid'],prediction=predict(S,m,data['X_grid'],c,raw))
    return dict(hist[-1],seconds=rec['seconds'],parameter_count=rec['parameter_count'])
def run_trial(S,ds,c,out):
    row=dict(config=c,status='ok',cases=[]);tick=time.perf_counter()
    try:
        for i,data in enumerate(ds):row['cases'].append(record_fit(S,data,c,out/f'signal{i}'))
        row['validation_mse']=float(np.mean([r['validation_mse'] for r in row['cases']]))
    except FloatingPointError as e:row.update(status='failed',error=str(e))
    row['wall_seconds']=time.perf_counter()-tick;return row
def choose(rows):
    good=[r for r in rows if r['status']=='ok']
    if not good:raise ValueError('All trials failed')
    return min(good,key=lambda r:(r['validation_mse'],r['config']['scale'],r['config']['lr'],r['config']['lam']))
def current(path,stamp):
    d=json.loads(path.read_text(encoding='utf-8'))
    if d['implementation_sha256']!=stamp:raise ValueError(f'Implementation changed: rerun {path.stem}')
    return d
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['check','observe','tune','stress','transfer','final'])
    ap.add_argument('--submission',default=str(ROOT/'src/submission.py'));ap.add_argument('--out',default='results')
    ap.add_argument('--extra-lambdas',nargs=2,type=float);ap.add_argument('--reason',default='')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    S=load_submission(args.submission);stamp=digest(args.submission);tick=time.perf_counter()
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()!=expected:raise ValueError('Provided data changed: '+name)
    if args.stage=='check':
        result=check(S);result['implementation_sha256']=stamp;save(out/'check.json',result);print(result);return
    current(out/'check.json',stamp);ds=cases();result={'implementation_sha256':stamp}
    if args.stage=='observe':
        result['runs']={}
        for name,raw,width in [('raw64',True,64),('raw70',True,70),('fourier64',False,64)]:
            result['runs'][name]=[record_fit(S,d,configuration(),out/'observe'/name/f'signal{i}',raw,width) for i,d in enumerate(ds)]
    elif args.stage=='tune':
        if args.extra_lambdas:
            prev=current(out/'tune_coarse.json',stamp);rows=prev['trials'];anchor=choose(rows)['config']
            extra=args.extra_lambdas
            if len(set(extra))!=2 or not all(1e-6<=v<=.1 for v in extra) or not args.reason.strip():raise ValueError('Two distinct positive lambdas and --reason required')
            # Do not append repeatedly: exactly the frozen six coarse + two current extras.
            for i,lam in enumerate(extra):
                c=dict(anchor,lam=lam);rows.append(run_trial(S,ds,c,out/'tune'/f'extra{i}'))
            result.update(trials=rows,extra_reason=args.reason,recommended_config=choose(rows)['config'])
        else:
            rows=[]
            for i,(scale,lr) in enumerate((s,l) for s in (.25,1,4) for l in (.001,.003)):
                rows.append(run_trial(S,ds,configuration(scale,lr),out/'tune'/f'coarse{i}'))
            result.update(trials=rows,recommended_config=choose(rows)['config'])
            save(out/'tune_coarse.json',result)
        with (out/'trials.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['scale','lr','lam','status','validation_mse_0','validation_mse_1','validation_mse_2','mean_validation_mse','mean_train_mse','wall_seconds'])
            for r in rows:
                vals=[v['validation_mse'] for v in r['cases']];vals+=['']*(3-len(vals))
                tm=float(np.mean([v['train_mse'] for v in r['cases']])) if r['cases'] else ''
                w.writerow([*r['config'].values(),r['status'],*vals,r.get('validation_mse',''),tm,r['wall_seconds']])
    else:
        tuned=current(out/'tune.json',stamp)
        if len(tuned['trials'])!=8:raise ValueError('Complete six coarse plus two extra trials first')
        selected=tuned['recommended_config'];positive=[r for r in tuned['trials'] if r['config']['lam']>0 and r['status']=='ok']
        if not positive:raise ValueError('No usable positive lambda trial')
        positive_config=choose(positive)['config']
        if args.stage=='stress':
            # If final selection has lam=0, use the best positive validation trial for this controlled comparison.
            lam=selected['lam'] if selected['lam']>0 else positive_config['lam']
            result.update(selected_config=selected,stress_lambda=lam,runs={})
            for name,c in [('medium',configuration(1,selected['lr'],0)),('high',configuration(4,selected['lr'],0)),('high_regularized',configuration(4,selected['lr'],lam))]:
                result['runs'][name]=[record_fit(S,d,c,out/'stress'/name/f'signal{i}',sparse=True) for i,d in enumerate(ds)]
        elif args.stage=='transfer':
            d=dict(np.load(ROOT/'data/image.npz'));ix=d['train_idx'];test=d['test_idx']
            m,rec=train(S,d['X'][ix],d['y'][ix],selected);pred=predict(S,m,d['X'],selected)
            result.update(config=selected,train_mse=float(np.mean((pred[ix]-d['y'][ix])**2)),unobserved_mse=float(np.mean((pred[test]-d['y'][test])**2)),fit_seconds=rec['seconds'])
            mask=np.full(4096,np.nan);mask[ix]=d['y'][ix,0]
            fig,axes=plt.subplots(1,4,figsize=(12,3))
            for ax,title,im in zip(axes,['Target','Observed','Reconstruction','Absolute error'],[d['image'],mask.reshape(64,64),pred.reshape(64,64),abs(pred-d['y']).reshape(64,64)]):
                ax.imshow(im,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
            fig.tight_layout();fig.savefig(out/'transfer.png',dpi=160);plt.close(fig);np.savez_compressed(out/'transfer_predictions.npz',prediction=pred)
        elif args.stage=='final':
            for name in ['observe','stress','transfer']:current(out/(name+'.json'),stamp)
            c=validate_config(S.get_config())
            if c!=selected:raise ValueError(f'get_config must match best mean-validation config: {selected}')
            if current(out/'transfer.json',stamp)['config']!=c or current(out/'stress.json',stamp)['selected_config']!=c:raise ValueError('Selected configuration changed: rerun stress/transfer')
            matches=[r for r in tuned['trials'] if r['status']=='ok' and r['config']==c]
            if not matches:raise ValueError('Configuration not in recorded trials')
            repeat=run_trial(S,ds,c,out/'final')
            if repeat['status']!='ok':raise FloatingPointError('Final retraining failed')
            delta=abs(repeat['validation_mse']-matches[0]['validation_mse'])
            if delta>1e-6*max(1.,matches[0]['validation_mse']):raise ValueError('Final retraining inconsistent with trials')
            result.update(config=c,mean_validation_mse=repeat['validation_mse'],reproduction_delta=delta,submission_sha256=hashlib.sha256(Path(args.submission).read_bytes()).hexdigest(),seed=SEED,steps=800)
    result['wall_seconds']=time.perf_counter()-tick;result['environment']=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),platform=platform.platform(),threads=1,device='cpu')
    save(out/(args.stage+'.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ('trials','runs')},ensure_ascii=False))
if __name__=='__main__':main()
