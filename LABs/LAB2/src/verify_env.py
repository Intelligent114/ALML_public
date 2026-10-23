import sys,json,platform
from pathlib import Path
import numpy as np,torch,matplotlib
assert sys.version_info[:2]==(3,12),'Use the supplied Python 3.12 environment'
assert Path('data/manifest.json').exists(),'Run from LAB2 root'
out=Path('results');out.mkdir(exist_ok=True)
info=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,torch=torch.__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),device='cpu')
(out/'environment.json').write_text(json.dumps(info,indent=2));print('LAB2 environment ready');print(info)
