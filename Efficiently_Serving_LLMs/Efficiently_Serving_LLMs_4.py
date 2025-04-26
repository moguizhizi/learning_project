import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
from learning_project.helpers import *

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
model.config.pad_token_id = model.config.eos_token_id


def generate_batch_with_past(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)

    last_logits = outputs.logits[:, -1, :]

    next_token_ids = torch.argmax(last_logits, dim=-1)

    return next_token_ids, outputs.past_key_values


def generate_batch(model, inputs, max_tokens):
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
        next_token_ids, past_key_values = generate_batch_with_past(
            model, next_inputs)
        next_inputs = {
            # '-1' here means the remaining elements for this dim
            "input_ids": next_token_ids.reshape((-1, 1)),
            # increment last, discard the rest
            "position_ids": next_inputs["position_ids"][:, -1].unsqueeze(-1) + 1,
            "attention_mask": torch.cat([
                next_inputs["attention_mask"],
                # concatenate vector of 1's with shape [batch_size]
                torch.ones((next_token_ids.shape[0], 1)),
            ], dim=1),
            "past_key_values": past_key_values,
        }

        next_tokens = tokenizer.batch_decode(next_token_ids)
        for i, token in enumerate(next_tokens):
            generated_tokens[i].append(token)
    return ["".join(tokens) for tokens in generated_tokens]


prompts = ["The quick brown fox jumped over the",
           "The rain in Spanin falls", "What comes up must"]


batch_size = 8
request_queue_len = 32

queues = [(prompts[0], 100 if i % batch_size == 0 else 10)
          for i in range(request_queue_len)]


batches = [queues[i:(i+1)*batch_size]
           for i in range(0, len(queues), batch_size)]

t0 = time.time()
with tqdm(total=len(batches), desc=f"bs={batch_size}") as pbar:
    for i, batch in enumerate(batches):
        max_token = [queue[1] for queue in batch]
        max_token = max(max_token)
        pbar.set_postfix({'max_token':max_token})

        inputs = [queue[0] for queue in batch]
        inputs = tokenizer(inputs, padding=True, return_tensors="pt")

        generate_batch(model, inputs, max_token)

        pbar.update(i)

print(f"durations:{time.time()-t0}")

# seed the random number generator so our results are deterministic
random.seed(42)

# constants
queue_size = 32
batch_size = 8

# requests waiting to be processed
# this time requests are tuples (prompt, max_tokens)
request_queue = [
    (prompts[0], 100 if i % batch_size == 0 else 10)
    for i in range(queue_size)
]

t0 = time.time()
with tqdm(total=len(request_queue), desc=f"bs={batch_size}") as pbar:
    # first, let's seed the initial cached_batch
    # with the first `batch_size` inputs
    # and run the initial prefill step
    batch = init_batch(request_queue[:batch_size])
    cached_batch = generate_next_token(batch)
    request_queue = request_queue[batch_size:]

    # continue until both the request queue is 
    # fully drained and every input
    # within the cached_batch has completed generation
    while (
        len(request_queue) > 0 or
        cached_batch["input_ids"].size(0) > 0
    ):
        batch_capacity = (
            batch_size - cached_batch["input_ids"].size(0)
        )
        if batch_capacity > 0 and len(request_queue) > 0:
            # prefill
            new_batch = init_batch(request_queue[:batch_capacity])
            new_batch = generate_next_token(new_batch)
            request_queue = request_queue[batch_capacity:]

            # merge
            cached_batch = merge_batches(cached_batch, new_batch)

        # decode
        cached_batch = generate_next_token(cached_batch)

        # remove any inputs that have finished generation
        cached_batch, removed_indices = filter_batch(cached_batch)
        pbar.update(len(removed_indices))

duration_s = time.time() - t0
print("duration", duration_s)