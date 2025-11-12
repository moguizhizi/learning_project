import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# 模拟一个包含 3 个 Expert 的简单 MoE


class SimpleMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=3, k=1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        # gating 网络，用来决定哪些 expert 被激活
        self.gate = nn.Linear(input_dim, num_experts)

        # 每个 expert 是一个独立的线性层
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        # gate 输出 logits
        logits = self.gate(x)                 # [batch, num_experts]
        scores = F.softmax(logits, dim=-1)    # [batch, num_experts]

        # Top-k 选出激活的 expert
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)

        # 重点：Top-k 是不可导的！但是我们用它来选择路径
        # 只对选中的 expert 执行前向计算
        outputs = []
        for i in range(self.k):
            expert_idx = topk_idx[:, i]
            expert_out = torch.zeros(x.size(0), self.experts[0].out_features)

            for b in range(x.size(0)):
                expert = self.experts[expert_idx[b]]
                expert_out[b] = expert(x[b]) * topk_scores[b, i]

            outputs.append(expert_out)

        # 聚合选中 expert 的结果
        return sum(outputs)


# 测试
moe = SimpleMoE(input_dim=4, hidden_dim=2, num_experts=3, k=1)
x = torch.randn(5, 4)
out = moe(x)

loss = out.sum()
loss.backward()

print("✅ 反向传播完成，没有报错。")
