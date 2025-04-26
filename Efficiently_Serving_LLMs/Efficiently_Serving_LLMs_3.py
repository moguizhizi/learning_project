from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
model.config.pad_token_id = model.config.eos_token_id

prompts = ["The quick brown fox jumped over the",
           "The rain in Spanin falls", "What comes up must"]
inputs = tokenizer(prompts, padding=True, return_tensors="pt")

print(f"input_ids:{inputs['input_ids']}")
print(f"input_ids shape:{inputs['input_ids'].size()}")

print(f"attention_mask:{inputs['attention_mask']}")
print(f"attention_mask shape:{inputs['attention_mask'].size()}")

attention_mask = inputs['attention_mask']
position_ids = attention_mask.cumsum(-1) - 1
position_ids = position_ids.masked_fill(attention_mask == 0, 1)

print(f"position_ids:{position_ids}")
print(f"position_ids shape:{position_ids.size()}")


def generate_batch_with_past(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)

    last_logits = outputs.logits[:, -1, :]

    next_token_ids = torch.argmax(last_logits, dim=-1)

    return next_token_ids, outputs.past_key_values


inputs["position_ids"] = position_ids
next_token_ids, past_key_values = generate_batch_with_past(model, inputs)
print(next_token_ids)

input_ids = next_token_ids.reshape(-1, 1)
attention_mask = torch.cat([inputs["attention_mask"], torch.ones(
    (inputs["attention_mask"].size(0), 1))], dim=-1)
position_ids = position_ids[:, -1:] + 1

next_inputs = {"input_ids": input_ids, "attention_mask": attention_mask,
               "position_ids": position_ids, "past_key_values": past_key_values}

next_token_ids, past_key_values = generate_batch_with_past(model, next_inputs)
print(next_token_ids)


def generate_batch(model ,inputs, max_tokens):
    # create a list of tokens for every input in the batch
    generated_tokens = [[] for _ in range(inputs["input_ids"].shape[0])]
    
    attention_mask = inputs["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    
    next_inputs = {
        "position_ids": position_ids,
        **inputs
    }
    for _ in range(max_tokens):
        next_token_ids, past_key_values = generate_batch_with_past(model, next_inputs)
        next_inputs = {
            "input_ids": next_token_ids.reshape((-1, 1)),  # '-1' here means the remaining elements for this dim
            "position_ids": next_inputs["position_ids"][:, -1].unsqueeze(-1) + 1,  # increment last, discard the rest
            "attention_mask": torch.cat([
                next_inputs["attention_mask"],
                torch.ones((next_token_ids.shape[0], 1)),  # concatenate vector of 1's with shape [batch_size]
            ], dim=1),
            "past_key_values": past_key_values,
        }

        next_tokens = tokenizer.batch_decode(next_token_ids)
        for i, token in enumerate(next_tokens):
            generated_tokens[i].append(token)
    return ["".join(tokens) for tokens in generated_tokens]


durations_s = []
throughput = []
latencies = []

batch_sizes = [2**p for p in range(8)]

for batch_size in batch_sizes:
    print(f"bs={batch_size}")
    
    batch_prompts = [prompts[i % len(prompts)] for i in range(batch_size)]
    inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt")
    
    start_time = time.time()
    
    generate_batch(model, inputs, 512)
    
    durations_s.append(time.time()-start_time)
    throughput.append(batch_size * inputs["input_ids"].size(0)/(time.time()-start_time))
    latencies.append((time.time()-start_time)/512)
    
    print(f"durations_s:{time.time()-start_time}")
    print(f"throughput:{batch_size * inputs['input_ids'].size(0)/(time.time()-start_time)}")
    print(f"latencies:{(time.time()-start_time)/512}")
    

    