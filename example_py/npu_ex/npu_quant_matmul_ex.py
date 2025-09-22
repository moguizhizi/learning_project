import torch
import torch_npu
import logging
import os

# cpu_x1 = torch.randint(-5, 5, (1, 3, 4), dtype=torch.int8)
# cpu_x2 = torch.randint(-5, 5, (1, 4, 5), dtype=torch.int8)
# scale = torch.randn(5, dtype=torch.float32)
# offset = torch.randn(5, dtype=torch.float32)
# bias = torch.randint(-5, 5, (1, 1, 5), dtype=torch.int32)
# # Method 1：You can directly call npu_quant_matmul
# npu_out = torch_npu.npu_quant_matmul(
#    cpu_x1.npu(), cpu_x2.npu(), scale.npu(), offset=offset.npu(), bias=bias.npu()
# )

# print(f"scale:{scale.size()}")
# print(f"offset:{offset.size()}")
# print(f"bias:{bias.size()}")

cpu_x1 = [[0,0,0,0],
          [1,1,1,1],
          [2,2,2,2]]
cpu_x1 = torch.tensor(cpu_x1, dtype=torch.int8)

cpu_x2 = [[3,3,3,3,3],
          [4,4,4,4,4],
          [5,5,5,5,5],
          [6,6,6,6,6]]
cpu_x2 = torch.tensor(cpu_x2, dtype=torch.int8)

scale = [1,2,3,4,5]
scale = torch.tensor(scale, dtype=torch.float32)

offset = [0,0,0,0,0]
offset = torch.tensor(offset, dtype=torch.float32)

bias = [0,0,0,0,0]
bias = torch.tensor(bias, dtype=torch.int32)

npu_out = torch_npu.npu_quant_matmul(
   cpu_x1.npu(), cpu_x2.npu(), scale=scale.npu(), offset=offset.npu(), bias=bias.npu(), 
)

print(f"output_dtype is none(int8):{npu_out}")

npu_out = torch_npu.npu_quant_matmul(
   cpu_x1.npu(), cpu_x2.npu(), scale=scale.npu(), bias=bias.npu(), output_dtype=torch.int8
)

print(f"output_dtype is int8 for no offset:{npu_out}")

npu_out = torch_npu.npu_quant_matmul(
   cpu_x1.npu(), cpu_x2.npu(), scale=scale.npu(), bias=bias.npu()
)

print(f"output_dtype is none for no offset:{npu_out}")


npu_out = torch_npu.npu_quant_matmul(
   cpu_x1.npu(), cpu_x2.npu(), scale=scale.npu(), bias=bias.npu(), output_dtype=torch.float16
)

print(f"output_dtype is float16 for no offset:{npu_out}")


npu_out = torch_npu.npu_quant_matmul(
   cpu_x1.npu(), cpu_x2.npu(), scale=scale.npu(), bias=bias.npu(), output_dtype=torch.int32
)

print(f"output_dtype is int32 for no offset:{npu_out}")


bias = [6,6,6,6,6]
bias = torch.tensor(bias, dtype=torch.int32)


npu_out = torch_npu.npu_quant_matmul(
   cpu_x1.npu(), cpu_x2.npu(), scale=scale.npu(), offset=offset.npu(), bias=bias.npu(), 
)

print(f"output_dtype is none(int8) for bias is no_zero:{npu_out}")

npu_out = torch_npu.npu_quant_matmul(
   cpu_x1.npu(), cpu_x2.npu(), scale=scale.npu(), bias=bias.npu(), output_dtype=torch.int8
)

print(f"output_dtype is int8 for no offset,  bias is no_zero:{npu_out}")

npu_out = torch_npu.npu_quant_matmul(
   cpu_x1.npu(), cpu_x2.npu(), scale=scale.npu(), bias=bias.npu()
)

print(f"output_dtype is none for no offset,  bias is no_zero:{npu_out}")


npu_out = torch_npu.npu_quant_matmul(
   cpu_x1.npu(), cpu_x2.npu(), scale=scale.npu(), bias=bias.npu(), output_dtype=torch.float16
)

print(f"output_dtype is float16 for no offset,  bias is no_zero:{npu_out}")


npu_out = torch_npu.npu_quant_matmul(
   cpu_x1.npu(), cpu_x2.npu(), scale=scale.npu(), bias=bias.npu(), output_dtype=torch.int32
)

print(f"output_dtype is int32 for no offset,  bias is no_zero:{npu_out}")




