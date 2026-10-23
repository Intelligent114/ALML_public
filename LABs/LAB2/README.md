# LAB2：神经网络如何学习不同频率的信号

个人实验，预计 10–12 小时。正式任务见 `lab2.pdf`，截止时间为 2026 年 11 月 23 日 06:00。CPU 可以完成；只修改 `src/submission.py`。

## 环境与运行

在 LAB2 根目录执行；先按 LAB0 安装 Conda，并新建独立环境：

```bash
conda env create -f environment.yml
conda activate ai3002-lab2
python src/verify_env.py
python src/run.py check
python src/run.py observe
python src/run.py tune
```

阅读六组基础试验后，自行补选两个不同的正则化系数并写明理由。下面仅演示命令格式：

```bash
python src/run.py tune --extra-lambdas 0.0001 0.01 --reason "Compare weak and strong penalties after inspecting coarse trials"
python src/run.py stress
python src/run.py transfer
```

将 `results/tune.json` 的 `recommended_config` 原样写入 `submission.py` 的 `get_config()`，再执行：

```bash
python src/run.py final
```

## 任务与产物

| 阶段 | 工作 | 主要证据 |
|---|---|---|
| check | NumPy、PyTorch 与有限差分三方核验 | `check.json` |
| observe | 原始坐标、近似参数量匹配、Fourier 表示三种对照 | `observe.json`、预测与频率曲线 |
| tune | 六组基础比较及两组自主补选，只按验证集选择 | `trials.csv`、`tune.json` |
| stress | 32 点含噪训练，中等尺度、高尺度及高尺度正则化 | `stress.json`、预测图 |
| transfer | 一次二维图像重建，不重新调参 | `transfer.png`、`transfer.json` |
| final | 配置与记录核对、固定预算重训 | `final.json` |

数据和代码放在 `data/`、`src/`，输出放在 `results/`。不修改提供的数据、初始化、优化器、网络宽度或 800 次更新预算。除核验用 NumPy 小网络外，主实验由 PyTorch 自动微分训练。若修改计算实现，重新运行前面的各阶段；只改 `get_config()` 无需重做实现核验。

## 提交

- 报告 PDF 上传 [Blackboard](https://bb.ustc.edu.cn)，文件名为 `学号-姓名-LAB2.pdf`，占 50 分。
- 单个 `submission.py` 上传 [OJ](https://oj.temaurinum.moe)，代码占 40 分、性能占 10 分。
- 不接收其他文件。图表、参数选择、命令、耗时与 AI 使用/独立核验记录直接写入报告。结果目录和模型只在本地保留。

正式性能在未公开的一维信号实例上按统一 CPU 协议重训评测；不劣于固定基准得满性能分，否则按 RMSE 比例折算。二维任务只计报告分。

2026-10-23 更新：最终性能基准采用 `scale=0.25, lr=0.003, lam=0`，由六组基础对照的公开验证均值选出；已有提交统一按新基准重评。计分公式、代码 40 分与报告 50 分、800 次更新预算均不变。`observe` 阶段仍使用原规定的对照配置；学生最终配置仍须按自己的完整选参记录确定。
