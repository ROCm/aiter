import torch
import aiter
from aiter import dtypes
# dtype = torch.float8_e4m3fn
dtype = torch.bfloat16
device = "cuda:0"
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim))
# @torch.compile(fullgraph=True)
def test_aiter_compile(q, k, v, do):
    out, _ = aiter.flash_attn_func(q, k, v,
        # dropout_p=0.0,
        causal=True,
        # window_size=(-1, -1),
        # bias=None,
        # alibi_slopes=None,
        deterministic=False,
        return_lse=True,
        # return_attn_probs=False,
    )
    return out
def test_case(
    batch_size,
    seq_len,
    num_heads_q,
    num_heads_kv,
    head_dim):
    tflop_fwd = 4 * batch_size * seq_len * seq_len * num_heads_q * head_dim / 1e12
    tflop_fwd = tflop_fwd * 0.5
    tflop_bwd = tflop_fwd * 2.5
    q = torch.randn(
        (batch_size, seq_len, num_heads_q, head_dim),
        device=device,
        requires_grad=True,
    ).to(dtype)
    k = torch.randn(
        (batch_size, seq_len, num_heads_kv, head_dim),
        device=device,
        requires_grad=True,
    ).to(dtype)
    v = torch.randn(
        (batch_size, seq_len, num_heads_kv, head_dim),
        device=device,
        requires_grad=True,
    ).to(dtype)
    o = torch.randn(
        (batch_size, seq_len, num_heads_q, head_dim),
        device=device,
        requires_grad=True,
    ).to(dtype)
    do = torch.randn(
        (batch_size, seq_len, num_heads_q, head_dim),
        device=device,
        requires_grad=True,
    ).to(dtype)
    k = repeat_kv(k, num_heads_q // num_heads_kv)
    v = repeat_kv(v, num_heads_q // num_heads_kv)
    torch.cuda.synchronize()
    print("Test compile")
    out = test_aiter_compile(q, k, v, do)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)
    print("Test compile end")
    # Warm-UP
    ITER = 100
    for _ in range(ITER):
        (
            o,
            _,
        ) = aiter.flash_attn_func(q, k, v,
            causal=True,
            return_lse=True,
            deterministic=False,
        )
        dq, dk, dv = torch.autograd.grad(o, (q, k, v), do)
    # event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # FWD
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(ITER):
        (
            o,
            _,
        ) = aiter.flash_attn_func(q, k, v,
            causal=True,
            return_lse=True,
            deterministic=False,
        )
    end_event.record()
    torch.cuda.synchronize()
    avg_time_fwd = start_event.elapsed_time(end_event) / 1000 / ITER
    # FWD + BWD
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(ITER):
        # q.grad = None
        # k.grad = None
        # v.grad = None
        (
            o,
            _,
        ) = aiter.flash_attn_func(q, k, v,
            causal=True,
            return_lse=True,
            deterministic=False,
        )
        dq, dk, dv = torch.autograd.grad(o, (q, k, v), do)
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / 1000 / ITER
    avg_time_bwd = avg_time - avg_time_fwd
    tflops_fwd = tflop_fwd / avg_time_fwd
    tflops_bwd = tflop_bwd / avg_time_bwd
    print(
        "B={}, Seq={}, HeadQ={}, HeadKV={}, Dim={} \nFWD: Time={}, TFLOPS={}\nBWD: Time={}, TFLOPS={}".format(
            batch_size,
            seq_len,
            num_heads_q,
            num_heads_kv,
            head_dim,
            avg_time_fwd,
            tflops_fwd,
            avg_time_bwd,
            tflops_bwd,
        )
    )
    return tflops_fwd, tflops_bwd
if __name__ == "__main__":
    # test_case(6, 8192, 64, 8, 128) # mbs, seq, q_head, kv_head, head_dim
    test_case(6, 8192, 32, 8, 128) # llama3 8B
    test_case(6, 8192, 64, 8, 128) # llama3 70B
