"""Supplied local/OJ training protocol. Do not edit."""
import ast, hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
STEPS=800
SEED=20261021
WIDTH=64
SNAPSHOTS=(0,50,200,800)
def load_submission(path):
    spec=importlib.util.spec_from_file_location('lab2_submission',str(path)); s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);return s
def digest(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='get_config']
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()
def validate_config(c):
    if not isinstance(c,dict) or set(c)!={'scale','lr','lam'}:raise ValueError('Exactly scale, lr, lam required')
    if not all(type(v) in (int,float) and np.isfinite(v) for v in c.values()):raise ValueError('Finite scalar configuration required')
    if c['scale'] not in (.25,1,4) or c['lr'] not in (.001,.003) or not (c['lam']==0 or 1e-6<=c['lam']<=.1):raise ValueError('Out-of-range config')
    return {k:float(c[k]) for k in ('scale','lr','lam')}
def initial_params(d,h=64,seed=SEED,dtype=np.float32):
    rng=np.random.default_rng(seed);p={}
    for j,(a,b) in enumerate(((d,h),(h,h),(h,1)),1):
        p['W'+str(j)]=(rng.standard_normal((a,b))*np.sqrt((2 if j<3 else 1)/a)).astype(dtype)
        p['b'+str(j)]=np.zeros(b,dtype=dtype)
    return p
def assert_model(m,p):
    if type(m)!=nn.Sequential or len(m)!=5 or [type(l) for l in m]!=[nn.Linear,nn.ReLU,nn.Linear,nn.ReLU,nn.Linear]:raise ValueError('Architecture must be Linear/ReLU/Linear/ReLU/Linear')
    for j,i in enumerate((0,2,4),1):
        if not torch.equal(m[i].weight.detach().cpu(),torch.as_tensor(p['W'+str(j)].T)) or not torch.equal(m[i].bias.detach().cpu(),torch.as_tensor(p['b'+str(j)])):
            raise ValueError('Supplied parameters not loaded exactly')
def project_error(x,pred,truth,freqs):
    residual=np.asarray(pred).reshape(-1)-np.asarray(truth).reshape(-1);u=np.asarray(x).reshape(-1)
    # Uniform periodic diagnostic grid: squared sine/cosine coefficients / 2.
    a=2*np.mean(residual[:,None]*np.sin(np.pi*u[:,None]*freqs),axis=0)
    b=2*np.mean(residual[:,None]*np.cos(np.pi*u[:,None]*freqs),axis=0)
    return ((a*a+b*b)/2).tolist()
def train(S,x,y,c,eval_sets=None,raw=False,width=WIDTH,device='cpu',steps=STEPS):
    c=validate_config(c);torch.set_num_threads(1);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    tick=time.perf_counter();x=torch.as_tensor(x,dtype=torch.float32,device=device);y=torch.as_tensor(y,dtype=torch.float32,device=device)
    with torch.no_grad():features=x if raw else S.fourier_features(x,c['scale'])
    p=initial_params(features.shape[1],width);model=S.build_torch_model(p);assert_model(model,p);model=model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=c['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
    history=[];snapshots={}; evaluations={}
    for name,ds in (eval_sets or {}).items():
        xx=torch.as_tensor(ds[0],dtype=torch.float32,device=device)
        with torch.no_grad():ff=xx if raw else S.fourier_features(xx,c['scale'])
        evaluations[name]=(ff,np.asarray(ds[1]),np.asarray(ds[0]),ds[2] if len(ds)>2 else None)
    for step in range(steps+1):
        if step%50==0 or step==steps:
            with torch.no_grad():
                pred=model(features);row={'step':step,'train_mse':float(((pred-y)**2).mean()),'objective':float(S.torch_objective(pred,y,model,c['lam']))}
                for name,(ff,yy,xx,freq) in evaluations.items():
                    out=model(ff).cpu().numpy();row[name+'_mse']=float(np.mean((out-yy)**2))
                    if freq is not None:row[name+'_frequency_error']=project_error(xx,out,yy,freq)
                    if step in SNAPSHOTS:snapshots.setdefault(name,{})[str(step)]=out.tolist()
                if not all(np.isfinite(v) for v in row.values() if np.isscalar(v)):raise FloatingPointError('Nonfinite training/evaluation')
                history.append(row)
        if step==steps:break
        opt.zero_grad(set_to_none=True);loss=S.torch_objective(model(features),y,model,c['lam'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();opt.step()
    return model,{'history':history,'snapshots':snapshots,'seconds':time.perf_counter()-tick,'parameter_count':sum(v.numel() for v in model.parameters())}
def predict(S,model,x,c,raw=False):
    device=next(model.parameters()).device
    with torch.no_grad():
        x=torch.as_tensor(x,dtype=torch.float32,device=device);return model(x if raw else S.fourier_features(x,c['scale'])).cpu().numpy()
def score(student_rmse,baseline_rmse):
    if not np.isfinite(student_rmse) or student_rmse<0:return 0.
    if not np.isfinite(baseline_rmse) or baseline_rmse<=0:raise ValueError('Invalid baseline')
    return float(10*min(1,baseline_rmse/max(student_rmse,1e-12)))
