import os
from safetensors import safe_open

def inspect_safetensors(file_path):
    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        return

    total_params = 0
    print(f"Inspecting {file_path}")
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if not keys:
            print("没有参数 (empty file)")
            return
        for key in keys:
            tensor = f.get_tensor(key)
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            numel = tensor.numel()
            total_params += numel
            print(f"{key:60} shape={str(shape):25} dtype={str(dtype):10} params={numel:,}")

    print("-" * 100)
    print(f"Total parameters: {total_params:,}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python inspect_safetensors_ex.py path_to_model.safetensors")
    else:
        inspect_safetensors(sys.argv[1])
