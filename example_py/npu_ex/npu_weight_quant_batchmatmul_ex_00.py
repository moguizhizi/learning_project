import torch
import torch_npu
# 输入int8+ND 
# cpu_x = torch.randn((3, 4),dtype=torch.float16)
# cpu_weight = torch.randint(low=-8, high=8, size=(4, 5),dtype=torch.int8)
# cpu_antiquantscale = torch.randn((1, 5),dtype=torch.float16)
# cpu_antiquantoffset = torch.randn((1, 5),dtype=torch.float16)
# cpu_quantscale = torch.randn((1, 5),dtype=torch.float32)
# cpu_quantoffset = torch.randn((1, 5),dtype=torch.float32)


# cpu_quantscale = [1,1,1,1,1]
# cpu_quantscale = torch.tensor(cpu_quantscale, dtype=torch.float32).unsqueeze(0)

# cpu_quantoffset = [0,0,0,0,0]
# cpu_quantoffset = torch.tensor(cpu_quantoffset, dtype=torch.float32).unsqueeze(0)

# quantscale= torch_npu.npu_trans_quant_param(cpu_quantscale.npu(), cpu_quantoffset.npu())



# npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(),quantscale.npu())


# print(npu_out)
# print(quantscale.dtype)
# print(quantscale)


# cpu_x = torch.randn((3, 4),dtype=torch.float16)
# cpu_weight = torch.randint(low=-8, high=8, size=(4, 5),dtype=torch.int8)
# cpu_antiquantscale = torch.randn((1, 5),dtype=torch.float16)
# cpu_antiquantoffset = torch.randn((1, 5),dtype=torch.float16)
# cpu_quantscale = torch.randn((1, 5),dtype=torch.float32)
# cpu_quantoffset = torch.randn((1, 5),dtype=torch.float32)


# cpu_quantscale = [1,1,1,1,1]
# cpu_quantscale = torch.tensor(cpu_quantscale, dtype=torch.float32).unsqueeze(0)

# cpu_quantoffset = [0,0,0,0,0]
# cpu_quantoffset = torch.tensor(cpu_quantoffset, dtype=torch.float32).unsqueeze(0)

# quantscale= torch_npu.npu_trans_quant_param(cpu_quantscale.npu(), cpu_quantoffset.npu())



# npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(),quantscale.npu())


# print(npu_out)
# print(quantscale.dtype)
# print(quantscale)

cpu_x1 = [[0,0,0,0],
          [1,1,1,1],
          [2,2,2,2]]
cpu_x1 = torch.tensor(cpu_x1, dtype=torch.float16)

cpu_x2 = [[3,3,3,3,3],
          [4,4,4,4,4],
          [5,5,5,5,5],
          [6,6,6,6,6]]
cpu_x2 = torch.tensor(cpu_x2, dtype=torch.int8)

cpu_antiquantscale = [1,1,1,1,1]
cpu_antiquantscale = torch.tensor(cpu_antiquantscale, dtype=torch.float16).unsqueeze(0)
print(cpu_antiquantscale.size())

cpu_antiquantoffset = [-1,-1,-1,-1,-1]
cpu_antiquantoffset = torch.tensor(cpu_antiquantoffset, dtype=torch.float16).unsqueeze(0)
print(cpu_antiquantoffset.size())

cpu_quantscale = [1,1,1,1,1]
cpu_quantscale = torch.tensor(cpu_quantscale, dtype=torch.float32).unsqueeze(0)

cpu_quantoffset = [1,1,1,1,1]
cpu_quantoffset = torch.tensor(cpu_quantoffset, dtype=torch.float32).unsqueeze(0)

quantscale= torch_npu.npu_trans_quant_param(cpu_quantscale.npu(), cpu_quantoffset.npu())

npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x1.npu(), cpu_x2.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), quantscale.npu())
print(npu_out)
print(npu_out.size())