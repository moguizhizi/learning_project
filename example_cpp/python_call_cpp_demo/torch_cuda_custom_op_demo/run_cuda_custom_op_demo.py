from pathlib import Path
import subprocess
import sys

import torch


THIS_DIR = Path(__file__).resolve().parent


def build_extension() -> None:
    subprocess.check_call(
        [sys.executable, "setup_cuda.py", "build_ext", "--inplace"],
        cwd=THIS_DIR,
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available in this environment; skip running demo.")
        return

    build_extension()
    torch.ops.load_library(str(next(THIS_DIR.glob("my_cuda_ops_lib*.so"))))

    a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    b = torch.tensor([10.0, 20.0, 30.0], device="cuda")

    print("Python calls torch.ops.my_cuda_ops.add_cuda(a, b):")
    print("a =", a)
    print("b =", b)
    print("result =", torch.ops.my_cuda_ops.add_cuda(a, b))


if __name__ == "__main__":
    main()
