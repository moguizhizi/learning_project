# Torch CUDA Custom Op Demo

This demo shows a CUDA implementation registered through PyTorch's dispatcher.
It is closer to vLLM's `STABLE_TORCH_LIBRARY_IMPL(..., CUDA, m)` pattern than
the pybind11 demo.

Core registration:

```cpp
TORCH_LIBRARY(my_cuda_ops, m) {
  m.def("add_cuda(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_cuda_ops, CUDA, m) {
  m.impl("add_cuda", &add_cuda);
}
```

Run it on a machine with CUDA available:

```bash
python3 example_cpp/python_call_cpp_demo/torch_cuda_custom_op_demo/run_cuda_custom_op_demo.py
```

Call path:

```text
Python
  -> torch.ops.load_library(...)
  -> torch.ops.my_cuda_ops.add_cuda(a, b)
  -> PyTorch dispatcher
  -> CUDA implementation add_cuda
  -> add_kernel<<<...>>>()
```
