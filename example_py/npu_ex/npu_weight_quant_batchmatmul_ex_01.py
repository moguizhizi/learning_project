import torch
import torch_npu
cpu_x = torch.randn((96, 320),dtype=torch.float16)
cpu_weight = torch.randint(low=-8, high=8, size=(256, 320),dtype=torch.int8)
cpu_antiquantscale = torch.randn((256),dtype=torch.float16)
# 构建int64类型的scale参数
antiquant_scale = torch_npu.npu_trans_quant_param(cpu_antiquantscale.to(torch.float32).npu()).reshape(256, 1)
cpu_antiquantoffset = torch.randint(-128, 127, (256, 1), dtype=torch.int32)
npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.transpose(-1,-2).npu(), antiquant_scale.transpose(-1,-2).npu(), cpu_antiquantoffset.transpose(-1,-2).npu())
print(npu_out)