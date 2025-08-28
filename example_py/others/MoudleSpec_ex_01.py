from megatron.core.transformer.spec_utils import ModuleSpec

from torch.nn import Linear

linear_spec = ModuleSpec(module=Linear, params={
                         "in_features": 768, "out_features": 2024, "bias": True})

instance = linear_spec.module(**linear_spec.params)
print(instance)
