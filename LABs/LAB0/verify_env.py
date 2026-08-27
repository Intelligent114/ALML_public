#!/usr/bin/env python3
"""AI3002 LAB0 local environment verifier.

The final token depends only on the normalized student number, the protocol
version, and deterministic integer results. Package versions, paths, GPU
availability, and platform information are printed for diagnostics but are
never included in the token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any


PROTOCOL = "AI3002-2026F-LAB0-v1"
EXPECTED_ENV_NAME = "ai3002"
STUDENT_ID_PATTERN = re.compile(r"^[A-Z0-9]{6,20}$")
SELF_TEST_STUDENT_ID = "TEST000001"
SELF_TEST_TOKEN = "8c75124e59aa921a1c2feb388f12e31b0cde97957d2522206ad01cc108a83480"


class VerificationError(RuntimeError):
    """An actionable environment verification failure."""


def pass_message(message: str) -> None:
    print(f"[PASS] {message}")


def info_message(message: str) -> None:
    print(f"[INFO] {message}")


def run_command(arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        raise VerificationError(f"找不到命令：{arguments[0]}") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise VerificationError(f"命令执行失败：{' '.join(arguments)}") from exc
    return completed.stdout.strip()


def normalize_student_id(raw_student_id: str) -> str:
    student_id = raw_student_id.strip().upper()
    if not STUDENT_ID_PATTERN.fullmatch(student_id):
        raise VerificationError("学号应由 6--20 位英文字母或数字组成。")
    return student_id


def check_python_and_conda() -> None:
    if sys.version_info[:2] != (3, 11):
        raise VerificationError(
            f"当前 Python 为 {sys.version.split()[0]}，本实验要求使用 Python 3.11。"
        )
    pass_message(f"Python {sys.version.split()[0]}")

    active_environment = os.environ.get("CONDA_DEFAULT_ENV", "")
    if active_environment != EXPECTED_ENV_NAME:
        shown = active_environment or "未检测到 Conda 环境"
        raise VerificationError(
            f"当前环境为 {shown}，请先运行 conda activate {EXPECTED_ENV_NAME}。"
        )
    pass_message(f"Active conda environment: {active_environment}")
    info_message(f"Python executable: {sys.executable}")


def check_git() -> None:
    version = run_command(["git", "--version"])
    pass_message(version)

    user_name = run_command(["git", "config", "--global", "--get", "user.name"])
    if not user_name:
        raise VerificationError(
            'Git user.name 尚未配置。请运行 git config --global user.name "Your Name"。'
        )
    pass_message("Git user.name is configured")

    user_email = run_command(["git", "config", "--global", "--get", "user.email"])
    if not user_email:
        raise VerificationError(
            'Git user.email 尚未配置。请运行 git config --global user.email "your_email@example.com"。'
        )
    pass_message("Git user.email is configured")


def compute_results() -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise VerificationError(
            "无法导入 NumPy。请确认已激活 ai3002，并重新创建 environment.yml 中的环境。"
        ) from exc

    try:
        import sklearn
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:
        raise VerificationError(
            "无法导入 scikit-learn。安装名是 scikit-learn，导入名是 sklearn。"
        ) from exc

    try:
        import torch
    except ImportError as exc:
        raise VerificationError(
            "无法导入 PyTorch。请确认 environment.yml 已完整安装。"
        ) from exc

    numpy_x = np.array([[2, -1, 3], [0, 4, 5]], dtype=np.int64)
    numpy_y = np.array([[1, 2], [3, 0], [-2, 1]], dtype=np.int64)
    numpy_result = (numpy_x @ numpy_y).reshape(-1).tolist()

    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 2, 2, 1, 0, 1]
    sklearn_result = (
        confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        .astype(np.int64)
        .reshape(-1)
        .tolist()
    )

    torch_a = torch.tensor([[1, -2], [3, 4]], dtype=torch.int64, device="cpu")
    torch_b = torch.tensor([[2, 1], [1, 3]], dtype=torch.int64, device="cpu")
    torch_result = (torch_a @ torch_b).reshape(-1).tolist()

    expected_numpy = [-7, 7, 2, 5]
    expected_sklearn = [2, 0, 0, 0, 1, 1, 0, 1, 1]
    expected_torch = [0, -5, 10, 15]
    if numpy_result != expected_numpy:
        raise VerificationError(f"NumPy 计算结果异常：{numpy_result}")
    if sklearn_result != expected_sklearn:
        raise VerificationError(f"scikit-learn 计算结果异常：{sklearn_result}")
    if torch_result != expected_torch:
        raise VerificationError(f"PyTorch 计算结果异常：{torch_result}")

    pass_message(f"NumPy {np.__version__}")
    pass_message(f"scikit-learn {sklearn.__version__}")
    pass_message(f"PyTorch {torch.__version__}")
    info_message(f"CUDA available: {torch.cuda.is_available()} (LAB0 does not require CUDA)")
    pass_message("All deterministic computations matched")

    return {
        "python": sum(i * i for i in range(11)),
        "numpy": numpy_result,
        "sklearn": sklearn_result,
        "torch": torch_result,
    }


def generate_token(student_id: str, results: dict[str, Any]) -> tuple[str, str]:
    payload = {
        **results,
        "protocol": PROTOCOL,
        "student_id": normalize_student_id(student_id),
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    token = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return token, canonical_json


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI3002 LAB0 环境验证")
    parser.add_argument("--student-id", help="本人学号；字母不区分大小写")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="使用课程内置测试学号检查脚本版本和 SHA256 协议",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    print("AI3002 LAB0 Environment Verification")
    print("-" * 40)
    try:
        check_python_and_conda()
        check_git()
        results = compute_results()
        student_id = (
            SELF_TEST_STUDENT_ID
            if arguments.self_test
            else arguments.student_id or input("请输入本人学号：")
        )
        normalized_student_id = normalize_student_id(student_id)
        token, _canonical_json = generate_token(normalized_student_id, results)
        if arguments.self_test:
            if token != SELF_TEST_TOKEN:
                raise VerificationError(
                    "脚本自检失败：文件可能被修改，或当前脚本与实验协议版本不一致。"
                )
            pass_message(f"Protocol self-test: {PROTOCOL}")
            print("\n自检通过。请去掉 --self-test，并使用本人学号再次运行。")
            return 0

        print(f"[PASS] Student ID: {normalized_student_id}")
        print(f"[PASS] Protocol: {PROTOCOL}")
        print("\n请将下面一行的 64 位字符串提交到 OJ：\n")
        print(token)
        return 0
    except VerificationError as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        print("请根据上面的提示修复环境后重新运行；不要修改 verify_env.py。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
