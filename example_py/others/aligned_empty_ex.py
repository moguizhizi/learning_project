import torch


def aligned_empty(shape, dtype=torch.float32, alignment=64, device="cpu"):
    """
    创建一个对齐的 Tensor，不丢数据。
    shape: 目标张量的形状
    dtype: 数据类型
    alignment: 对齐字节数 (必须是 element_size 的倍数)
    device: "cpu" 或 "cuda"
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    if alignment % element_size != 0:
        raise ValueError(
            f"alignment ({alignment}) 必须是 element_size ({element_size}) 的倍数")

    # 目标需要的字节数
    nbytes = torch.Size(shape).numel() * element_size

    # 申请多余的 buffer，确保有空间找到对齐地址
    buf = torch.empty(nbytes + alignment, dtype=torch.uint8, device=device)

    addr = buf.data_ptr()
    offset = (alignment - (addr % alignment)) % alignment

    # 切片对齐后的部分
    aligned_buf = buf[offset: offset + nbytes]

    # 转成目标 dtype
    aligned_tensor = aligned_buf.view(dtype).reshape(shape)

    return aligned_tensor
