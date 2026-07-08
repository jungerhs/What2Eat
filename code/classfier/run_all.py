"""
一键跑完数据集构建全流程（无 LLM 阶段时也能跑完模板部分）。
"""
import argparse
import subprocess
import sys
from pathlib import Path

CLASSFIER_ROOT = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("parse_dishes", [PY, "-m", "classfier.dataset.parse_dishes"]),
    ("generate_from_kg", [PY, "-m", "classfier.dataset.generate_from_kg"]),
    ("generate_by_llm (full)", [PY, "-m", "classfier.dataset.generate_by_llm", "--n", "2000"]),
    ("double_check", [PY, "-m", "classfier.dataset.double_check"]),
    ("clean_and_split", [PY, "-m", "classfier.dataset.clean_and_split"]),
    ("stats", [PY, "-m", "classfier.dataset.stats"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true",
                        help="跳过 LLM 合成阶段（仅跑模板 + 清洗 + 切分）")
    args = parser.parse_args()

    for name, cmd in STEPS:
        if args.skip_llm and "generate_by_llm" in name:
            print(f"==== [SKIP] {name} ====")
            continue
        if "double_check" in name and args.skip_llm:
            print(f"==== [SKIP] {name} (依赖 llm 合成) ====")
            continue
        print(f"\n==== [{name}] ====\n  $ {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=CLASSFIER_ROOT.parent)
        if r.returncode != 0:
            print(f"!! 步骤失败: {name}, 退出码 {r.returncode}")
            sys.exit(r.returncode)

    print("\nAll done.")


if __name__ == "__main__":
    main()
