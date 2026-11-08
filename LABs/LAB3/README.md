# LAB3 少量标签能否跨设备识别？

先阅读 lab3.pdf。个人完成，截止时间为 2026 年 12 月 21 日 06:00。环境、运行器与六份公开数据都在本目录。Windows PowerShell 或 macOS 终端进入本目录后执行：

```
conda env create -f environment.yml
conda activate ai3002-lab3
python src/run.py env
```

只编辑 src/submission.py。依次运行：

```
python src/run.py check
python src/run.py baseline
python src/run.py compare
python src/run.py stress
python src/run.py final
```

比较不同配置时可加 `--out results/candidate1`，将各次运行分开。最终的 compare 和 final 须使用同一份代码与输出目录。运行器会在 results/ 生成证据表、混淆矩阵、图和配置摘要。提交报告 PDF 至 [Blackboard](https://bb.ustc.edu.cn)，单个 submission.py 至 [OJ](https://oj.temaurinum.moe)。报告 50 分、代码 30 分、性能 20 分。
