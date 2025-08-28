from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

# 模拟 state_dict 中的 key 名称映射
sharded_state_dict_keys_map = {
    TransformerLayerSubmodules.ATTENTION: {
        "qkv.weight": "transformer.layers.0.attention.query_key_value.weight",
        "qkv.bias": "transformer.layers.0.attention.query_key_value.bias"
    },
    TransformerLayerSubmodules.MLP: {
        "dense_h_to_4h.weight": "transformer.layers.0.mlp.dense_h_to_4h.weight",
        "dense_4h_to_h.weight": "transformer.layers.0.mlp.dense_4h_to_h.weight"
    }
}

# 使用这个映射加载 state_dict 中的权重
def map_weights_from_shard(shard_state_dict, mapping):
    model_weights = {}
    for submodule, param_map in mapping.items():
        for local_name, full_name in param_map.items():
            if full_name in shard_state_dict:
                model_weights[f"{submodule}.{local_name}"] = shard_state_dict[full_name]
    return model_weights

# 模拟一个 shard 的 state_dict
shard_state_dict = {
    "transformer.layers.0.attention.query_key_value.weight": "QKV_WEIGHT",
    "transformer.layers.0.attention.query_key_value.bias": "QKV_BIAS",
    "transformer.layers.0.mlp.dense_h_to_4h.weight": "MLP_1_WEIGHT",
    "transformer.layers.0.mlp.dense_4h_to_h.weight": "MLP_2_WEIGHT",
}

mapped = map_weights_from_shard(shard_state_dict, sharded_state_dict_keys_map)
for k, v in mapped.items():
    print(f"{k} => {v}")
