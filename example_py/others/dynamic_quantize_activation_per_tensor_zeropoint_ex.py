import torch


def quant_dequant(t, do_align_zp):
    quant_min = -127
    quant_max = 127

    t_min = t.min()
    t_max = t.max()

    scale = (t_max - t_min) / (quant_max - quant_min)
    zp = (t_max + t_min) / 2

    if do_align_zp:
        zp = (zp / scale).round() * scale  # ⬅️ 关键对齐操作

    # 中心化
    t_centered = t - zp

    # 量化
    t_quant = (t_centered / scale).round().clamp(quant_min, quant_max)

    # 反量化
    t_dequant = t_quant * scale + zp

    return t_dequant


# 测试样本（不对称，放大误差差异）
torch.manual_seed(50)
t = torch.rand(1000) * 5 - 2  # [-2, 3] 区间内

# 分别处理
out_align = quant_dequant(t, do_align_zp=True)
out_no_align = quant_dequant(t, do_align_zp=False)

# 误差对比
mae_align = (out_align - t).abs().mean()
mae_no_align = (out_no_align - t).abs().mean()

print(f"使用 zp 对齐后的 MAE:      {mae_align.item():.6f}")
print(f"未使用 zp 对齐时的 MAE:   {mae_no_align.item():.6f}")
print(f"误差差值（未对齐 - 对齐）: {(mae_no_align - mae_align).item():.6f}")
