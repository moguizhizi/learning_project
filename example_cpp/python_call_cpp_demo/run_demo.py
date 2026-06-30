from pathlib import Path
import importlib
import subprocess
import sys

import torch


THIS_DIR = Path(__file__).resolve().parent


def build_extension():
    subprocess.check_call(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=THIS_DIR,
    )
    return importlib.import_module("cpp_ops_demo")


def main() -> None:
    cpp_ops = build_extension()

    print("Python calls C++ add_int(3, 5):")
    print("result =", cpp_ops.add_int(3, 5))

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])

    print("\nPython passes torch.Tensor to C++ add_tensor(a, b):")
    print("a =", a)
    print("b =", b)
    print("result =", cpp_ops.add_tensor(a, b))


if __name__ == "__main__":
    main()
