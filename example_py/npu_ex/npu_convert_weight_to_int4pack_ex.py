import torch
import torch_npu

m = 128
k = 64
n = 32
trans_weight = False

cpu_x = torch.randn((m, k), dtype=torch.float16)
if trans_weight:
    cpu_weight = torch.randint(low=-8, high=8, size=(n, k), dtype=torch.int32)
    cpu_antiquantscale = torch.randn((n, 1), dtype=torch.float16)
    cpu_antiquantoffset = torch.randn((n, 1), dtype=torch.float16)
else:
    cpu_weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int32)
    cpu_antiquantscale = torch.randn((1, n), dtype=torch.float16)
    cpu_antiquantoffset = torch.randn((1, n), dtype=torch.float16)

cpu_x2 = [[40,3,3,3,3,3,3,3],
          [4,4,4,4,4,4,4,4],
          [5,5,5,5,5,5,5,5],
          [6,6,6,6,6,6,6,6]]
cpu_x2 = torch.tensor(cpu_x2, dtype=torch.int32)

weight_int4 = torch_npu.npu_convert_weight_to_int4pack(cpu_x2.npu())


print(cpu_x2.npu())
print(cpu_x2.npu().size())

print(weight_int4)
print(weight_int4.size())

exit(0)

if trans_weight:
    cpu_weight = cpu_weight.transpose(-1, -2)
    weight_int4 = weight_int4.transpose(-1, -2)
    cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)
    cpu_antiquantoffset = cpu_antiquantoffset.transpose(-1, -2)

npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), weight_int4.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu())
print(npu_out)