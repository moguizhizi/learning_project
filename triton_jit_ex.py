import triton
import triton.language as tl
import torch


@triton.jit
def add_(x_ptr, y_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offset < num_elements
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)

    output = x + y

    tl.store(output_ptr + offset, output, mask=mask)


x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")
output = torch.zeros_like(x, device="cuda")

BLOCK_SIZE = 256
num_thread = (1024 // BLOCK_SIZE,)

add_[num_thread](x, y, output, 1024, BLOCK_SIZE)
print(output)
