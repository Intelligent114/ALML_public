# LAB1：线性回归与正则化

实验正文见 [lab1.pdf](lab1.pdf)。按正文第 1 部分补全 `src/submission.py` 中的七个函数，再完成学习率和正则化参数的两轮调整，填写同一文件中的 `get_config()`，最后测试并保存。运行脚本和数据已提供，不需要自建实验框架。

## 截止时间

截止时间：2026 年 10 月 19 日 06:00（北京时间）。实验报告上传至 [Blackboard 系统（bb.ustc.edu.cn）](https://bb.ustc.edu.cn)，`submission.py` 上传至 [OJ 系统（oj.temaurinum.moe）](https://oj.temaurinum.moe)，不接收其他文件。

## 实验环境（不计分）

参照 LAB0 的操作，为本实验新建独立环境 `ai3002-lab1`，不要继续使用 LAB0 的 `ai3002` 环境。Windows 与 macOS 的完整步骤、Shell 初始化和 VS Code 解释器选择见实验正文。先进入 LAB1 根目录，执行：

```bash
conda env create -f environment.yml
conda activate ai3002-lab1
python --version
python src/verify_env.py
```

环境包含 Python 3.11、NumPy、scikit-learn 和 Matplotlib，CPU 即可。检查通过后显示 `LAB1 environment ready`，并生成环境记录；这部分不计分，不提交 OJ 字符串。以后运行前激活该环境即可，不必重复创建。

所有代码位于 `src/`，所有命令从 LAB1 根目录执行；数据位于 `data/`，输出位于 `results/`。`requirements.txt` 记录对应 Python 依赖范围，正常使用 `environment.yml` 创建环境即可。

## 任务清单

| 顺序 | 要完成的工作 | 验收产物 |
|---|---|---|
| 1 | 补全特征、预处理、直接求解、损失梯度、梯度下降和指标，共七个函数 | submission.py |
| 2 | 运行 check，修正实现问题 | checks.csv、gd_history.csv、loss_curve.png |
| 3 | 运行 tune-lr 比较三种学习率，依据结果自行补选一个新值 | lr_coarse.csv、lr_tuning.csv、lr_curves.png、lr_config.json |
| 4 | 运行 compare 做粗比较，再自行补选两个正则化参数 | comparison.csv（11 行汇总）、joint_comparison.csv（30 行）、regularization_curves.png、config.json |
| 5 | 查看评估对照，将所选值写入 get_config()，运行 final 核验、测试与保存 | audit.csv、submitted_config.json、test_metrics.csv、test_predictions.csv、final_model.npz |
| 6 | 回答四个问题，记录调参依据、流程、自检和耗时 | report.pdf，正文建议 4–6 页 |

```bash
python src/run.py --stage check --out results
python src/run.py --stage tune-lr --out results
python src/run.py --stage tune-lr --lr-factor 0.5 --out results
python src/run.py --stage compare --data data --out results
python src/run.py --stage compare --extra-lambdas 0.02 0.05 --out results
```

完成上述调参后，将 `config.json` 的 `basis`、`lam`、`lr_factor`，分别原样填入 `submission.py` 的 `get_config()` 三个字段 `basis`、`lambda`、`lr_factor`。再执行：

```bash
python src/run.py --stage final --data data --out results
```

上述补选数值只展示命令格式，应分别在阅读第一轮结果后决定自己的参数，再执行下一轮。学习率新倍数须在正文规定的稳定范围内，不能重复已试值；两个新正则化参数须为不同的正数，不能重复已有值。报告记录实际使用的完整命令。默认输出不足以完成调参任务。

先完成实现与调参并填写 `get_config()`，再运行 `final`；缺少任一项参数补选记录时，脚本会拒绝最终测试。代码中的配置与选参记录不一致时也会拒绝测试。不要根据测试结果改参数；若需要修复使测试无效的程序错误，必须在报告中记录原因和重跑过程。学习率观察阶段最多更新 2000 次；联合模型选择、本地测试重训与 OJ 最终训练均调用 `train_gd()`，从零开始最多更新 200 次，梯度最大绝对分量不超过 $10^{-7}$ 时可提前停止。每次拟合重新计算步长尺度，将所选倍数除以该尺度作为实际学习率。禁止用直接解替代训练、增加预算或热启动。

## 文件与接口

只需修改 `src/submission.py`；`src/run.py`、`src/supplied.py`、`data/` 为提供的运行材料。绘图、固定设备划分、参数比较、保存结果均由脚本完成。`src/config_protocol.py` 提供本地与 OJ 共用的配置校验和最终训练流程，不得修改。`src/benchmark.py` 提供独立的 sklearn 基准及公开计分函数，不得修改；`src/verify_env.py` 仅用于环境检查。

- `get_config()`：仅返回固定的 `basis`、`lambda`、`lr_factor`；禁止读取外部文件。`basis` 允许 `linear`、`quadratic`；均值基准仅作对照，不允许提交；正则化参数有限且非负；学习率倍数在开区间 $(0,2)$ 内。它不新增分值。
- `make_features(X, basis)`：输入原始特征，构造 linear 或 quadratic 特征。
- `fit_preprocessor(F, y)`：只在传入的训练数据上拟合均值和尺度。
- `transform(F, prep)`：应用已保存的预处理，不重新拟合。
- `solve_weights(Z, yc, lam)`：普通最小二乘或 Ridge 直接求解。
- `loss_gradient(Z, yc, w, lam)`：计算目标与解析梯度。
- `train_gd(...)`：零初始化的全批次梯度下降，返回系数和训练历史。
- `evaluate(y, pred)`：返回 MSE、RMSE、MAE。

具体输入形状和返回要求见函数注释及正文。`supplied.load_development()` 返回传感器矩阵、标签、行编号、设备编号和事后字段等；传感器矩阵本身不含事后字段。`make_solver_check_data()` 提供含重复列和常数列的小样例。

模型比较阶段自动运行三个稳定范围内学习率倍数（0.01、1 和补选值）与特征、正则化参数的组合。第一轮有 18 个组合，第二轮有 30 个；汇总表仍为 7 行与 11 行，含不参与最终选择的均值对照。最终三项配置全部来自 `config.json`，不能用 `lr_config.json` 的效率最优倍数替换。正则化曲线固定倍数为 1，便于控制变量。

固定 8:1:1 划分由 `src/run.py` 中的 `splits` 提供。

## 成绩评定

| 类别 | 分数 | 内容 |
|---|---|---|
| 代码（OJ） | 40 | 预处理 8、直接求解 8、损失梯度 8、梯度下降 8、评价指标 8 |
| 调参与实验结果（报告） | 30 | 学习率调整 10、正则化参数调整 12、两项评估对照 4、测试保存 4 |
| 问题回答、流程与反馈（报告） | 20 | 四个问题各 4 分，流程 2、AI 或自检记录 1、用时反馈 1 |
| 最终性能（OJ） | 10 | 教师测试集上与固定 sklearn Ridge 的 RMSE 比较 |
| 环境创建 | 0 | 新环境创建与检查，不计分 |

OJ 系统负责代码 40 分与最终性能 10 分的评分，共 50 分；Blackboard 实验报告占其余 50 分（调参与实验结果 30 分，问题回答、流程与反馈 20 分）。

最终性能不低于固定基准得 10 分，否则按“基准 RMSE / 学生 RMSE”的比例计分；正式性能分由 OJ 系统评定。仅这 10 分按最终性能计分，调参过程仍按证据与方法评分。允许使用 AI，但必须理解提交内容；报告需有独立核验记录。

## 提交要求

仅接收以下两项，分别上传至指定平台：

| 提交内容 | 平台 | 要求 |
|---|---|---|
| 实验报告（PDF） | [Blackboard 系统](https://bb.ustc.edu.cn) | 占 50 分；文件名为 `学号-姓名-LAB1.pdf`，图表与运行记录直接写入报告 |
| `submission.py` | [OJ 系统](https://oj.temaurinum.moe) | 代码 40 分、最终性能 10 分；单独上传 `src/submission.py`，文件名保持 `submission.py` |

不接收其他文件。不要上传压缩包、`results/` 目录、模型文件、数据、环境文件或其他代码。

报告包含四组学习率试验、最终 11 行汇总表（标注第一轮与补选项，并列出最佳学习率倍数）、同一特征与正则化参数下不同学习率的验证误差比较、三项测试指标及三行对照；放入学习率曲线与正则化曲线，回答四个问题。补选参数前的观察与理由必须记录；失败或未改善的试验如实保留。

报告还须明确记录补选学习率、两个补选正则化参数、最终配置及完整运行命令。OJ 直接读取 `submission.py` 的 `get_config()`，按统一的 200 次梯度下降预算使用开发数据训练并计算性能分，不依赖报告或外部配置文件。报告中的最终配置须与该接口一致；教师可按报告记录复核调参过程。所有结果必须由该实现和所记录的参数可复现地产生。

`results/` 中的结果与 `final_model.npz` 仅留存本地，不上传。OJ 独立测试七个计算函数获得代码 40 分；最终性能 10 分由 OJ 使用代码中的配置在未公开的新设备数据上评定；`performance_preview.csv` 只是本地测试预估，不可据此继续调参。不能用已包含全部开发数据的最终模型在开发数据上计算误差，再称为独立测试成绩。

内容疑问通过课程仓库 Issues 提问，注明 LAB1 和任务编号，不公开完整答案或个人信息。
