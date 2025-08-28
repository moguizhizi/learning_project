import torch

is_eos = torch.tensor([[True, False, True, False],
                            [False, True, False, True],
                            [True, True, False, False]], dtype=torch.bool)

print(is_eos.int().argmax(dim=1))
print(is_eos.any(dim=1))