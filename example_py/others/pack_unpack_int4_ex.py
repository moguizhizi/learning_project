import torch

def _unpack_int4(packed_weight: torch.Tensor, original_n: int) -> torch.Tensor:
    """
    Unpack int8 weight back to int4 weight
    @param packed_weight: torch.Tensor, int8 packed weight (from _pack_int4)
    @param original_n: 原始 n 维度（必须是偶数）
    @return: torch.Tensor, int4 weight (值范围 0~15)
    """
    e, k, n_new = 0, 0, 0
    if len(packed_weight.shape) == 2:
        k, n_new = packed_weight.shape
    elif len(packed_weight.shape) == 3:
        e, k, n_new = packed_weight.shape
    else:
        raise ValueError("packed_weight shape must be 2D or 3D")

    # 解出两个 int4
    high = torch.bitwise_right_shift(packed_weight, 4) & 0b00001111  # 高 4 bit
    low  = packed_weight & 0b00001111                               # 低 4 bit

    # 拼回原始维度
    unpacked = torch.stack([low, high], dim=-1).reshape(-1, original_n)
    if e == 0:
        return unpacked.reshape(k, original_n).to(torch.int8)
    else:
        return unpacked.reshape(e, k, original_n).to(torch.int8)

def _pack_int4(weight) -> torch.Tensor:
    """
    Pack int4 weight to int8 weight
    @param weight: torch.Tensor, int4 weight
    @return: torch.Tensor, int8 weight
    """
    weight = weight.to(torch.int8)
    e = 0  # number of experts
    if len(weight.shape) == 2:
        k, n = weight.shape
    elif len(weight.shape) == 3:
        e, k, n = weight.shape
    n_new = n // 2 + n % 2

    if n_new != n // 2:
        raise AssertionError("n dimension should be even")
    
    weight = weight.reshape(-1, 2)
    weight0 = weight[:, :1]
    weight1 = weight[:, 1:]

    weight1_4 = torch.bitwise_left_shift(weight1, 4)
    weight2_4 = weight0 & 0b00001111

    weight_add = torch.bitwise_or(weight1_4, weight2_4)
    if e == 0:
        weight_res = weight_add.reshape(k, n_new)
    else:
        weight_res = weight_add.reshape(e, k, n_new)
    return weight_res

# 原始 int4 权重
w = torch.tensor([[1, 7, 2, 15],[2, 4, 6, 8]], dtype=torch.int8)  # shape (1, 4)

packed = _pack_int4(w)
print("Packed:", packed)  # 例如 tensor([[113, 242]], dtype=torch.int8)

unpacked = _unpack_int4(packed, original_n=4)
print("Unpacked:", unpacked)  # 回到 [[1, 7, 2, 15]]
