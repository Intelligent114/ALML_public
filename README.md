# AI3002 人工智能与机器学习基础：学生资料

本仓库发布中国科学技术大学 AI3002《人工智能与机器学习基础》2026 年秋季学期的学生版作业与实验材料。

实验文档提供 Windows 与 macOS 两套完整步骤。课程不要求学生使用 Linux；使用其他系统的同学可以参考命令，但应自行处理平台差异。

## 获取与更新

首次使用：

```bash
git clone https://github.com/Intelligent114/ALML_public.git
cd ALML_public
```

课程材料更新后，在仓库目录运行：

```bash
git pull
```

请将自己的作业答案和实验代码保存在单独的私有目录或私有仓库中，避免在拉取更新时产生冲突，也不要将解答公开上传。

## 当前材料

| 类型 | 编号 | 内容 | 主要文件 |
| --- | --- | --- | --- |
| 书面作业 | HW0 | 数学基础诊断 | `HWs/HW0/hw0.pdf` |
| 书面作业 | HW1 | 线性模型、正则化与性能评估 | `HWs/HW1/hw1.pdf` |
| 实验 | LAB0 | Git、Conda、VS Code 与机器学习环境配置 | `LABs/LAB0/lab0.pdf`、`verify_env.py`、`environment.yml` |
| 实验 | LAB1 | 线性回归、梯度下降、联合调参与性能评估 | `LABs/LAB1/lab1.pdf`、`README.md`、`src/`、`environment.yml` |

后续作业与实验将按课程进度陆续加入。

## 目录结构

```text
ALML_public/
|-- HWs/
|   |-- HW0/
|   |   |-- hw0.pdf
|   |   `-- hw0.tex
|   `-- HW1/
|       |-- hw1.pdf
|       |-- hw1.tex
|       `-- hw1.md
`-- LABs/
    |-- LAB0/
    |   |-- lab0.pdf
    |   |-- lab0.tex
    |   |-- environment.yml
    |   `-- verify_env.py
    `-- LAB1/
        |-- lab1.pdf
        |-- lab1.tex
        |-- lab1.md
        |-- README.md
        |-- environment.yml
        |-- requirements.txt
        |-- src/
        `-- data/dev.csv
```

PDF 是正式发布版本；同时提供 TeX 源文件便于无障碍阅读、检索和报告排版参考。实验所需代码和环境文件位于对应 LAB 目录。

## LAB0 提交

1. 阅读 `LABs/LAB0/lab0.pdf`。
2. 使用 `environment.yml` 创建并激活 `ai3002` 环境。
3. 在 `LABs/LAB0` 目录运行 `verify_env.py`。
4. 将脚本生成的 64 位 SHA256 提交至 [TensorJudge 课程 OJ](https://oj.temaurinum.moe/) 中的 LAB0。

## HW1 与 LAB1 提交

- HW1：将 `学号-姓名-HW1.pdf` 上传至 [Blackboard](https://bb.ustc.edu.cn)。
- LAB1：将实验报告 `学号-姓名-LAB1.pdf` 上传至 [Blackboard](https://bb.ustc.edu.cn)，将 `submission.py` 单独上传至 [课程 OJ](https://oj.temaurinum.moe)，不接收其他文件。OJ 负责代码 40 分与最终性能 10 分，报告占 50 分。
- LAB1 须新建 `ai3002-lab1` 环境，具体任务与运行命令见 `LABs/LAB1/README.md` 和实验 PDF。
- HW1 与 LAB1 的截止时间均为 **2026 年 10 月 19 日 06:00（北京时间）**；LAB1 的报告与代码须在此时间前分别提交。提交入口以课程通知为准。

## 通过 GitHub Issues 提问

对作业题意、实验步骤、环境安装、代码框架或 OJ 使用有疑问时，请先搜索已有问题，然后通过 [GitHub Issues](https://github.com/Intelligent114/ALML_public/issues/new/choose) 发布。不要通过公开仓库以外的临时渠道重复提问；集中讨论便于其他同学检索，也便于教师和助教统一更新说明。

Issue 应注明 HW/LAB 编号、Windows 或 macOS、系统版本、CPU 架构、执行过的命令、完整错误文本和已经尝试的方法。错误信息优先使用代码块粘贴，不要只发无法检索的截图。

## 学术诚信与隐私

本仓库只包含学生版题目和实验框架，不包含参考答案、隐藏测试、评分器或教师备注。请独立完成课程要求；不要公开传播自己的作业答案、实验实现或他人的提交。

GitHub Issue 是公开页面。不要发布真实学号、个人邮箱、验证码、密码、Cookie、SHA256 提交值、未公开答案或完整个人作业。需要处理账户信息时，只需在 Issue 中描述问题类型，教师或助教会说明后续安全渠道。

## 版权

除非具体文件另有说明，本仓库材料版权归课程教学团队所有。公开可见不等于授予复制、再发布或商业使用许可。
