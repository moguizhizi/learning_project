from megatron.core.transformer.spec_utils import ModuleSpec

# 定义一个线性层的 ModuleSpec 示例
from torch.nn import Linear

linear_spec = ModuleSpec(
    module=Linear,
    params={"in_features": 768, "out_features": 3072, "bias": True}
)

# 使用这个 ModuleSpec 来实例化模块
linear_layer = linear_spec.module(**linear_spec.params)
print(linear_layer)

a = Linear(in_features=768, out_features=3072, bias=True)
print(a)
