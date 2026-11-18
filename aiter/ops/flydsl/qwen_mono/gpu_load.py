import torch, time, sys
dev = "cuda"; a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16); b = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
big = torch.empty(2 << 30, dtype=torch.uint8, device=dev); src = torch.randint(0, 255, (2 << 30,), dtype=torch.uint8, device=dev)
t = time.time()
while time.time() - t < float(sys.argv[1]):
    for _ in range(20):
        c = a @ b; big.copy_(src); c = None
    torch.cuda.synchronize()
