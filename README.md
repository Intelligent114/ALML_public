# AI3002 人工智能与机器学习基础：学生资料

本仓库发布中国科学技术大学 AI3002《人工智能与机器学习基础》2026 年秋季学期的学生版作业与实验材料。

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
| 实验 | LAB0 | Git、Conda、VS Code 与机器学习环境配置 | `LABs/LAB0/lab0.pdf`、`verify_env.py`、`environment.yml` |

后续 HW1--HW5/6 与 LAB1--LAB4 将按课程进度陆续加入。

## 目录结构

```text
ALML_public/
|-- HWs/
|   `-- HW0/
|       |-- hw0.pdf
|       `-- hw0.tex
`-- LABs/
    `-- LAB0/
        |-- lab0.pdf
        |-- lab0.tex
        |-- environment.yml
        `-- verify_env.py
```

PDF 是正式发布版本；同时提供 TeX 源文件便于无障碍阅读、检索和报告排版参考。实验所需代码和环境文件位于对应 LAB 目录。

## LAB0 提交

1. 阅读 `LABs/LAB0/lab0.pdf`。
2. 使用 `environment.yml` 创建并激活 `ai3002` 环境。
3. 在 `LABs/LAB0` 目录运行 `verify_env.py`。
4. 将脚本生成的 64 位 SHA256 提交至 [TensorJudge 课程 OJ](https://oj.temaurinum.moe/) 中的 LAB0。

## 学术诚信与问题反馈

本仓库只包含学生版题目和实验框架，不包含参考答案、隐藏测试、评分器或教师备注。请独立完成课程要求；不要公开传播自己的作业答案、实验实现或他人的提交。

发现题目、代码或环境配置问题时，请通过课程公布的答疑渠道联系教师或助教，并附上操作系统、相关命令和完整错误信息。不要在公开 Issue 中发布个人学号、邮箱验证码、密码、SHA256 提交值或未公开答案。

## 版权

除非具体文件另有说明，本仓库材料版权归课程教学团队所有。公开可见不等于授予复制、再发布或商业使用许可。
