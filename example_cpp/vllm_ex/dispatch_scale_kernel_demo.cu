#include <ATen/Dispatch.h>
#include <torch/torch.h>

#include <iostream>

template <typename scalar_t>
__global__ void scale_kernel(scalar_t *data, int n, scalar_t factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        data[i] *= factor;
    }
}

void scale_tensor(torch::Tensor input) {
    int n = input.numel();
    auto dtype = input.scalar_type();

    int block = 256;
    int grid = (n + block - 1) / block;

    AT_DISPATCH_SWITCH(dtype, "scale_kernel",

        AT_DISPATCH_CASE(at::ScalarType::Float,
            [&] {
                using scalar_t = float;

                scale_kernel<scalar_t><<<grid, block>>>(input.data_ptr<scalar_t>(), n, 2.0f);
            })

            AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
                using scalar_t = at::Half;

                scale_kernel<scalar_t><<<grid, block>>>(input.data_ptr<scalar_t>(), n, (at::Half)2);
            }));
}

int main() {
    torch::manual_seed(0);

    torch::Tensor a = torch::rand({10}, torch::dtype(torch::kFloat).device(torch::kCUDA));

    std::cout << "before:\n" << a << std::endl;

    scale_tensor(a);

    cudaDeviceSynchronize();

    std::cout << "after:\n" << a << std::endl;

    return 0;
}