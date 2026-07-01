# Python Call C++ Demos

This directory contains small demos for Python calling C++/CUDA code.

## Pybind11 C++ Demo

```bash
python3 example_cpp/python_call_cpp_demo/pybind11_cpp_demo/run_demo.py
```

This demo imports a generated Python extension module and calls C++ functions.

## Torch CUDA Custom Op Demo

```bash
python3 example_cpp/python_call_cpp_demo/torch_cuda_custom_op_demo/run_cuda_custom_op_demo.py
```

This demo registers a CUDA implementation with PyTorch's dispatcher and calls it
through `torch.ops.my_cuda_ops.add_cuda(...)`.
