"""Public CLI; all paths default to LAB3 independent of shell working directory."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import protocol as p
import numpy as np
from check import checks

LAB = Path(__file__).resolve().parents[1]
STAGES = {"baseline":("raw_single","pca_single"),
          "compare":("raw_single","raw_multi","paired_single","paired_multi"),
          "stress":("paired_multi","shuffled_multi"), "final":("paired_multi",)}


def write(path, obj):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")
    temp.replace(path)


def plot(rows, summary, out, stage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes=list(summary)
    fig,ax=plt.subplots(figsize=(8,3+.36*len(modes)),layout="constrained")
    M=np.array([summary[m]["groups"] for m in modes])
    im=ax.imshow(M,vmin=0,vmax=1,cmap="Blues",aspect="auto")
    for i,j in np.ndindex(M.shape):
        ax.text(j,i,f"{M[i,j]:.1%}",ha="center",va="center",color="white" if M[i,j]>.62 else "black")
    ax.set_xticks(range(6),[f+"\n"+s for f in p.FORMS for s in p.SHIFTS])
    ax.set_yticks(range(len(modes)),modes); fig.colorbar(im,ax=ax,label="Macro recall")
    fig.savefig(out/(stage+".png"),dpi=150); plt.close(fig)
    if stage=="baseline":
        row=next(r for r in rows if r["mode"]=="raw_single" and r["form"]=="double")
        fig,axes=plt.subplots(1,3,figsize=(7.5,3),layout="constrained")
        for ax,mat,cond in zip(axes,row["confusion"],p.SHIFTS):
            ax.imshow(mat,vmin=0,vmax=80,cmap="Blues")
            for i,j in np.ndindex(3,3):
                n=mat[i][j]; ax.text(j,i,str(n),ha="center",va="center",color="white" if n>45 else "black")
            ax.set(xticks=[0,1,2],yticks=[0,1,2],title=cond,xlabel="Predicted class")
        axes[0].set_ylabel("True class")
        fig.savefig(out/"confusion.png",dpi=150); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["env","check",*STAGES,"all"])
    parser.add_argument("--submission",type=Path,default=LAB/"src/submission.py")
    parser.add_argument("--out",type=Path,default=LAB/"results")
    args=parser.parse_args()
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    student=p.load_student(args.submission)
    stamp=hashlib.sha256(args.submission.read_bytes()).hexdigest()
    import matplotlib
    env=dict(python=sys.version.split()[0],executable=sys.executable,numpy=np.__version__,
             matplotlib=matplotlib.__version__,platform=platform.platform(),threads=1)
    stages=["env","check",*STAGES] if args.stage=="all" else [args.stage]
    files=sorted((LAB/"data").glob("public*.npz"))
    if (any(s in STAGES for s in stages) and len(files)!=6):
        raise ValueError("Expected six public data files")
    for stage in stages:
        t=time.perf_counter()
        rec=dict(stage=stage,environment=env,submission_sha256=stamp,fit_seed=p.FIT_SEED)
        if stage=="check":
            rec["checks"]=checks(student)
        elif stage in STAGES:
            cfg=p.config(student)
            if stage=="final":
                previous=json.loads((out/"compare.json").read_text(encoding="utf8"))
                if previous["submission_sha256"]!=stamp or previous["config"]!=cfg:
                    raise ValueError("Re-run compare with the frozen submission")
            rows=[]
            for file in files:
                form=file.stem.split("_")[1]
                with np.load(file,allow_pickle=False) as ds:
                    for mode in STAGES[stage]:
                        model=p.fit(student,ds["X"],ds["y"],ds["pairs"],mode)
                        scores,mats=p.evaluate(model,ds["tests"],ds["truth"])
                        rows.append(dict(instance=file.stem,form=form,mode=mode,
                                         dimension=model[1].shape[1],recall=scores.tolist(),confusion=mats.tolist()))
            summary={m:p.summarize([r for r in rows if r["mode"]==m]) for m in STAGES[stage]}
            if stage=="final":
                old=previous["summary"]["paired_multi"]["groups"]
                np.testing.assert_allclose(summary["paired_multi"]["groups"],old,rtol=0,atol=1e-12)
            rec.update(config=cfg,rows=rows,summary=summary)
            plot(rows,summary,out,stage)
        rec["seconds"]=time.perf_counter()-t
        write(out/(stage+".json"),rec)
        print(stage,"OK",round(rec["seconds"],2),"s",out/(stage+".json"),flush=True)


if __name__=="__main__":
    main()
