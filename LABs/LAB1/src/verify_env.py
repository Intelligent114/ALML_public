"""Non-graded LAB1 environment check. Run from LAB1 after conda activation."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import sklearn
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    if sys.version_info[:2] != (3,11):
        raise SystemExit('Activate ai3002-lab1 with Python 3.11 before running this check.')
    env_name=Path(os.environ.get('CONDA_DEFAULT_ENV','')).name
    if env_name != 'ai3002-lab1' or Path(sys.prefix).name != 'ai3002-lab1':
        raise SystemExit('Expected the NEW conda environment ai3002-lab1; check the selected interpreter.')
    X=np.arange(5,dtype=float).reshape(-1,1)
    assert np.all(np.isfinite(Ridge().fit(X,X[:,0]).predict(X)))
    root=Path(__file__).resolve().parents[1]
    out=root/'results'; out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(); ax.plot(X[:,0],X[:,0]); fig.savefig(out/'environment_check.png'); plt.close(fig)
    result=dict(environment=env_name,python=sys.version,executable=sys.executable,
                numpy=np.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,status='PASS')
    (out/'environment_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('LAB1 environment ready. This check carries no marks; no OJ hash is required.')


if __name__=='__main__':
    main()
