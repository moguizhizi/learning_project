#include <torch/extension.h>

int add_int(int a, int b) {
  return a + b;
}

torch::Tensor add_tensor(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_int", &add_int, "Add two integers in C++");
  m.def("add_tensor", &add_tensor, "Add two tensors in C++");
}
