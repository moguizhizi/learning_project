import torch
import torch_npu

# scale = torch.randn(16, dtype=torch.float32)
# offset = torch.randn(16, dtype=torch.float32)
# round_mode = 1
# npu_out = torch_npu.npu_trans_quant_param(scale.npu(), offset.npu(), round_mode)
# print(npu_out)

scale = [1,2,3,4,5]
scale = torch.tensor(scale, dtype=torch.float32)

offset = [0,0,0,0,0]
offset = torch.tensor(offset, dtype=torch.float32)

round_mode = 1
npu_out = torch_npu.npu_trans_quant_param(scale.npu(), offset.npu(), round_mode)
print(npu_out)