# Pybind11 C++ Demo

This demo shows Python calling C++ functions through a PyTorch C++ extension.

Run from the project root:

```bash
python3 example_cpp/python_call_cpp_demo/pybind11_cpp_demo/run_demo.py
```

Call path:

```text
Python run_demo.py
  -> setup.py build_ext --inplace
  -> compile cpp_ops.cpp
  -> import generated extension module
  -> call C++ functions add_int and add_tensor
```

This is simpler than vLLM's `torch.ops._C.xxx` custom-op registration, but the
core idea is the same: Python enters compiled C++ code.
