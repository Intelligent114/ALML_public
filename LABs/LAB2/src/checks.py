"""Public three-way checks. Full grader is teacher-only."""
import numpy as np
import torch
from protocol import initial_params
def check(S):
    rng=np.random.default_rng(51);p=initial_params(3,4,seed=59,dtype=np.float64)
    p['b1'][:]=.11;p['b2'][:]=.17
    x=rng.normal(size=(5,3));y=rng.normal(size=(5,1));pr,cache=S.numpy_forward(x,p)
    before={k:v.copy() for k,v in p.items()};reports={}
    # Fixed tiny network, away from ReLU boundaries; finite differences all parameters.
    a1=x@p['W1']+p['b1'];a2=np.maximum(a1,0)@p['W2']+p['b2']
    assert min(np.min(abs(a1)),np.min(abs(a2)))>1e-4
    for lam in (0.,.03):
        value,g=S.numpy_loss_backward(pr,y,cache,p,lam)
        model=S.build_torch_model(p);xt=torch.tensor(x);yt=torch.tensor(y)
        out=model(xt);loss=S.torch_objective(out,yt,model,lam);loss.backward()
        assert np.allclose(out.detach().numpy(),pr,atol=1e-12)
        assert abs(float(loss.detach())-value)<1e-10
        errors={};epsilon=1e-6
        for j,i in enumerate((0,2,4),1):
            for key,tg in [('W'+str(j),model[i].weight.grad.numpy().T),('b'+str(j),model[i].bias.grad.numpy())]:
                numeric=np.empty_like(p[key])
                for idx in np.ndindex(p[key].shape):
                    vals=[]
                    for sign in (1,-1):
                        q={k:v.copy() for k,v in p.items()};q[key][idx]+=sign*epsilon
                        o,ca=S.numpy_forward(x,q);val,_=S.numpy_loss_backward(o,y,ca,q,lam);vals.append(val)
                    numeric[idx]=(vals[0]-vals[1])/(2*epsilon)
                denom=max(1.,np.max(abs(numeric)),np.max(abs(g[key])))
                err=float(np.max(abs(numeric-g[key]))/denom);errors[key]=err
                assert err<1e-6,(key,err)
                assert np.allclose(tg,g[key],rtol=1e-9,atol=1e-10),(key,'torch mismatch')
        reports[str(lam)]=errors
    for d in (1,2):
        z=rng.normal(size=(7,d));zt=torch.tensor(z,requires_grad=True)
        a=S.fourier_features(z,.25);b=S.fourier_features(zt,.25)
        assert a.shape==(7,13*d) and np.allclose(a,b.detach().numpy(),atol=1e-12)
        b.sum().backward();assert zt.grad is not None and torch.isfinite(zt.grad).all()
    assert all(np.array_equal(p[k],before[k]) for k in p)
    return {'passed':True,'relative_errors':reports,'threshold':1e-6}
