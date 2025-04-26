import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

print(model.config)

# 启用 GPU（如果可用）
if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU not available, using CPU")

# 禁用 SDPA（使用 eager 模式）
# model.config._attn_implementation = "eager"

# 初始输入
text = "Hello  why?"
inputs = tokenizer(text, return_tensors="pt")
inputs["attention_mask"] = torch.ones_like(
    inputs["input_ids"], dtype=torch.long)
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

# 验证初始形状
print("Initial input_ids shape:", inputs["input_ids"].shape)
print("Initial attention_mask shape:", inputs["attention_mask"].shape)
assert inputs["input_ids"].shape == inputs["attention_mask"].shape

# 生成函数


def generate_token_with_past(model, inputs, past_key_values=None):
    # 如果有 past_key_values，只传入最新 token
    if past_key_values is not None:
        input_ids = inputs["input_ids"][:, -1:]  # 只取最后一个 token
        attention_mask = inputs["attention_mask"][:, -1:]
    else:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

    print("Input IDs shape:", input_ids.shape)
    print("Attention Mask shape:", attention_mask.shape)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            return_dict=True
        )
    last_logits = output.logits[:, -1, :]
    next_token_id = torch.argmax(last_logits, dim=-1)
    return next_token_id, output.past_key_values


# 生成循环
durations_s = []
past_key_values = None
for _ in range(10):
    t0 = time.time()
    next_token_id, past_key_values = generate_token_with_past(
        model, inputs, past_key_values)
    durations_s.append(time.time() - t0)

    print("Past Key Values:", [kv.shape for kv in past_key_values[0]])

    # 更新输入
    next_token_id = next_token_id.unsqueeze(-1)  # [batch_size, 1]
    inputs["input_ids"] = torch.cat(
        (inputs["input_ids"], next_token_id), dim=1)
    inputs["attention_mask"] = torch.cat(
        (inputs["attention_mask"], torch.ones_like(next_token_id, dtype=torch.long)), dim=1)

print("Total duration:", sum(durations_s))
