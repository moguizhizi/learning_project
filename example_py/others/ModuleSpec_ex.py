from dataclasses import dataclass
from typing import Union, Type
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP

# 子模块定义类，用于包装 submodules
@dataclass
class TransformerLayerSubmodules:
    self_attention: ModuleSpec
    mlp: ModuleSpec

# 配置 SelfAttention 子模块
self_attention_spec = ModuleSpec(
    constructor=SelfAttention,
    kwargs={"hidden_size": 256, "num_attention_heads": 4}
)

# 配置 MLP 子模块
mlp_spec = ModuleSpec(
    constructor=MLP,
    kwargs={"hidden_size": 256, "ffn_hidden_size": 1024}
)

# 包装成 TransformerLayerSubmodules
layer_submodules = TransformerLayerSubmodules(
    self_attention=self_attention_spec,
    mlp=mlp_spec
)

# 配置 TransformerLayer 本身
layer_spec = ModuleSpec(
    constructor=TransformerLayer,
    submodules=layer_submodules
)

# 配置 TransformerConfig
transformer_config = TransformerConfig(
    hidden_size=256,
    num_attention_heads=4,
    num_layers=1,
    ffn_hidden_size=1024
)

# 构建 TransformerLayer
transformer_layer = build_module(
    layer_spec,
    config=transformer_config
)

# 打印层结构
print(transformer_layer)
