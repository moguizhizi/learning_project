import torch, time

device = torch.device('cuda:0')
n = 16384
a = torch.randn(n, n, device=device, dtype=torch.float16)
b = torch.randn(n, n, device=device, dtype=torch.float16)

# warm-up
for _ in range(10): c = torch.matmul(a, b)

torch.cuda.synchronize()
t0 = time.time()
iters = 100
for _ in range(iters):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
t1 = time.time()

elapsed = t1 - t0
ops = 2 * n * n * n * iters  # FMA 算 2 次
tflops = ops / elapsed / 1e12
print(f"实测 FP16 TFLOPS: {tflops:.2f}")