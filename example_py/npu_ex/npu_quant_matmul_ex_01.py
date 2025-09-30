import torch
import torch_npu

cpu_x1 = torch.tensor(
    [[[ 0,  0,  0,  0],
      [ 0,  0,  0,  0]],
     [[ 1,  1,  1,  1],
      [ 1,  1,  1,  1]],
     [[ 2,  2,  2,  2],
      [ 2,  2,  2,  2]]], dtype=torch.int8)
cpu_x1 = torch.tensor(cpu_x1, dtype=torch.int8)


cpu_x2 = torch.randint(-128, 127, (4, 4), dtype=torch.int8)

scale = [1,2,3,4]
scale = torch.tensor(scale, dtype=torch.float32)

pertoken_scale = torch.rand(2, dtype=torch.float32)

output = torch_npu.npu_quant_matmul(
            cpu_x1.npu(),
            cpu_x2.npu(),
            scale.npu(),
            pertoken_scale=pertoken_scale.npu(),

            output_dtype=torch.float16,
        )

print(output.size())
        