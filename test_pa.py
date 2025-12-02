import time
import torch
import torch.nn.functional as F

import aiter


def make_inputs(device="cuda", dtype=torch.bfloat16):
    seqlen = 1024
    headdim = 128
    numheads = 8
    shape   = (1, seqlen, numheads, headdim)

    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)

    return q, k, v


def aiter_flash_attn_only(q, k, v, scale):
    out, _ = aiter.flash_attn_func(
        q, k, v,
        dropout_p         = 0.0,
        softmax_scale     = scale,
        causal            = False,
        deterministic     = False,
        return_attn_probs = False,
        return_lse        = True,  
    )
    return out


def benchmark(fn, warmup=1, iters=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def clone_detached_requires_grad(t):
    return torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=t.device).copy_(t).requires_grad_()

def get_cloned_tensors(*tensors):
    return [clone_detached_requires_grad(tensor) for tensor in tensors]


def main():
    device, dtype, scale = "cuda", torch.bfloat16, 0.088388347648318447
    q, k, v = make_inputs(device, dtype)

    # ----------------------- latency (fwd + bwd) ---------------------------
    with torch.no_grad():
        t_fwd_aiter = benchmark(lambda: aiter_flash_attn_only(q, k, v, scale))

    q, k, v = get_cloned_tensors(q, k, v)
    aiter_out = aiter_flash_attn_only(q, k, v, scale)

    t_bwd_aiter = benchmark(lambda: aiter_out.backward(gradient=torch.randn_like(q), retain_graph=True))

    print("                   forward   backward")
    print(f"aiter            : {t_fwd_aiter:8.3f} {t_bwd_aiter:9.3f}")


if __name__ == "__main__":
    main()
    print("Exiting normally.")
