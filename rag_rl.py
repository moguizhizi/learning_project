import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("dev:", dev)

ds = load_dataset("code_search_net", "python")["train"]
codes = ds["func_code_string"][:1000]
docs = ds["func_documentation_string"][:1000]

emb = SentenceTransformer('all-MiniLM-L6-v2', device=dev)

tok = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
m = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono").to(dev)
m.eval()

def get_snips(q, k=3):
    q_emb = emb.encode(q, convert_to_tensor=True, device=dev)
    c_emb = emb.encode(codes, convert_to_tensor=True, device=dev)
    scores = util.cos_sim(q_emb, c_emb)[0]
    top = torch.topk(scores, k=k)
    snips = [f"Code:\n{codes[i.item()]}\nDocstring:\n{docs[i.item()]}\n" for i in top.indices]
    return snips

def gen_cands(q, snips, n=3):
    p = f"Query: {q}\n\nRelevant Code Snippets:\n" + "".join(f"Snippet {i+1}:\n{s}\n" for i, s in enumerate(snips)) + "\nAnswer the query based on the snippets above:\n"
    cands = []
    for _ in range(n):
        inp = tok(p, return_tensors="pt", truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            out = m.generate(**inp, max_length=1024, num_beams=5, do_sample=True, top_p=0.95, temperature=0.7, pad_token_id=tok.eos_token_id)
        cand = tok.decode(out[0], skip_special_tokens=True)
        cands.append(cand)
    return cands

def score(cand, q):
    kws = q.lower().split()
    c = cand.lower()
    cor = sum(1 for k in kws if k in c) / len(kws)
    l = len(cand.split())
    brev = max(0, 1 - (l - 50) / 150)
    return 0.7 * cor + 0.3 * brev

def rl_opt(cands, q):
    r = [score(c, q) for c in cands]
    idx = np.argsort(r)[::-1]
    sc = [cands[i] for i in idx]
    sr = [r[i] for i in idx]
    p = np.exp(sr) / np.sum(np.exp(sr))
    i = np.random.choice(len(cands), p=p)
    return sc[i], sr[i]

def assist(q):
    snips = get_snips(q)
    print("Retrieved Snippets:")
    for i, s in enumerate(snips):
        print(f"Snippet {i+1}:\n{s}")
    cands = gen_cands(q, snips)
    print("\nGenerated Candidates:")
    for i, c in enumerate(cands):
        print(f"Candidate {i+1}:\n{c}\n")
    best, r = rl_opt(cands, q)
    print(f"\nBest Candidate (Reward: {r:.2f}):\n{best}")

# q = """Find and fix the bug in this snippet:
# def average_positive_numbers(numbers):
#     total = 0
#     count = 0
#     for num in numbers:
#         if num > 0:
#             total += num
#             count += 1
#     return total / count
# """

# q = """Give an optimized version of this snippet:
# def sum_of_even_squares(numbers):
#     result = 0
#     for num in numbers:
#         if num % 2 == 0:  
#             square = num * num  
#             result += square  
#     return result"""

q = "Explain what retry_with_backoff does?"

assist(q)