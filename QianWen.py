import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader

# 加载模型和分词器
model_name = "Qwen/Qwen2.5-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载和预处理 WikiText 数据集
dataset_path = "/data1/project/learning_project/tokenized_wikitext"
try:
    # 尝试加载预处理数据集
    dataset = load_from_disk(dataset_path)
    print("数据集加载成功！")
except FileNotFoundError:
    print("未找到数据集，正在预处理 WikiText 数据集...")
    # 加载原始 WikiText 数据集
    dataset = load_dataset("wikitext", "wikitext-103-v1")

    # 分词函数
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    # 预处理数据集
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 保存到磁盘
    tokenized_dataset.save_to_disk(dataset_path)
    print(f"数据集已保存至 {dataset_path}")

    # 重新加载
    dataset = load_from_disk(dataset_path)
    print("预处理后数据集加载成功！")

# 自定义 collate_fn 将列表转换为 PyTorch 张量
def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# 创建 DataLoader
train_dataset = dataset["train"]
train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn  # 添加 collate_fn
)

model.gradient_checkpointing_enable()

# 初始化 DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./qwen2.5-3b-pretrained",
    per_device_train_batch_size=1,  # 减到 1
    gradient_accumulation_steps=32,  # 增加到 32，保持有效批次大小
    num_train_epochs=1,
    logging_steps=100,
    save_steps=1000,
    deepspeed="ds_config.json"
)

# 自定义训练循环（简化版）
def train():
    model_engine.train()
    for epoch in range(training_args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            inputs = batch["input_ids"].to(model_engine.device)
            labels = inputs.clone()            
            
            outputs = model_engine(inputs, labels=labels)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            if step % 100 == 0:
                print(f"轮次 {epoch}, 步骤 {step}, 损失: {loss.item()}")

# 运行训练
train()

# 保存模型
model_engine.save_checkpoint("./qwen2.5-3b-pretrained")