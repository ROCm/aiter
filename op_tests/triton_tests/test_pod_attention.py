import sys
import pytest
import torch

from aiter.ops.triton.pod_attention import pod_attention


def get_batch_num_block_n(n_ctx, BLOCK_N, batch):
    # N_CTX is a list of context lengthes for all the req in a batch
    # First, calculate #BLOCK_N for each context length "list_num_block_n"
    # Second, Convert it to a list of accumulative lengthes "list_sum_block_n"
    # Third, convert list to a tensor "batch_num_block_n"
    for s in n_ctx:
        # print(f"s={s}")
        list_num_block_n = [
            (int(str(s).strip()) + BLOCK_N - 1) // BLOCK_N for s in n_ctx
        ]
    len_sum = 0
    list_sum_block_n = []
    for i in range(batch):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, device="cuda", dtype=torch.int32)

    return batch_num_block_n


@pytest.mark.parametrize(
    "batch, h, n_ctx_q, n_ctx, d, total_programs, init_dtype, BLOCK_M, BLOCK_N, waves_per_eu, num_warps, \
        batch_pf, n_ctx_q_pf, n_ctx_pf, BLOCK_M_pf, BLOCK_N_pf",
    [
        (
            1,
            64,
            16,
            [65536],
            128,
            608,
            torch.float16,
            16,
            64,
            1,
            1,
            1,
            4096,
            [4096],
            128,
            64,
        ),
    ],
)
def test_pod_attention(
    request,
    batch,
    h,
    n_ctx_q,
    n_ctx,
    d,
    total_programs,
    init_dtype,
    BLOCK_M,
    BLOCK_N,
    waves_per_eu,
    num_warps,
    batch_pf,
    n_ctx_q_pf,
    n_ctx_pf,
    BLOCK_M_pf,
    BLOCK_N_pf,
):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.manual_seed(20)
    # Long seqlen (>512K) can hit memory access fault. Suspect compiler issue
    # WA with shorter d and longer BLOCK_N
    if any(item > 524288 for item in n_ctx):
        BLOCK_N = 256
        d = 16

    assert batch == len(n_ctx)
    assert batch_pf == len(n_ctx_pf)

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")
    batch_num_block_n = get_batch_num_block_n(n_ctx, BLOCK_N, batch)

    sum_n_ctx_pf = sum(int(n) for n in n_ctx_pf)
    batch_num_block_n_pf = get_batch_num_block_n(n_ctx_pf, BLOCK_N_pf, batch_pf)

    sm_scale = 0.5

    # Decode: Allocate Tensors
    q = torch.empty((n_ctx_q * batch, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((sum_n_ctx, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((sum_n_ctx, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # Decode LeanAttention Specific Parameters
    Mp = torch.empty((total_programs, BLOCK_M), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, BLOCK_M), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, BLOCK_M, d), device=q.device, dtype=torch.float32)

    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    # MI300 (gfx942) HW_ID layout:
    # XCC_ID: 4 bits
    # SE_ID: 2 bits
    # CU_ID: 4 bits, total 10 bits for a max value of 1024
    # MI300 has 8 XCD's, 4 SE per XCD, 10 CUs per SE, 2 CUs per XCD for harvesting. Total: (10CU's * 4 SE's - 2) * 8 XCD's = 304
    # Because the ID bits are shifted, CU_ID's are not continuous values
    max_num_cu = 1024
    cu_ctr = torch.zeros((max_num_cu,), device=q.device, dtype=torch.int32)

    # Prefill: Allocate Tensors
    q_pf = torch.empty(
        (n_ctx_q_pf * batch_pf, h, d), dtype=init_dtype, device="cuda"
    ).normal_(mean=0.0, std=0.5)
    k_pf = torch.empty((sum_n_ctx_pf, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v_pf = torch.empty((sum_n_ctx_pf, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # Prefill LeanAttention Specific Parameters
    Mp_pf = torch.empty(
        (total_programs, BLOCK_M_pf), device=q.device, dtype=torch.float32
    )
    Lp_pf = torch.empty(
        (total_programs, BLOCK_M_pf), device=q.device, dtype=torch.float32
    )
    Op_pf = torch.empty(
        (total_programs, BLOCK_M_pf, d), device=q.device, dtype=torch.float32
    )

    locks_pf = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    prefill_ratio = 1
    decode_ratio = 1

    # Calculate Pytorch refence output
    start = 0
    start_q = 0
    ref_out = torch.empty_like(q, dtype=v.dtype)
    causal = False  # Decode

    for b in n_ctx:
        qb = q[start_q : (start_q + int(n_ctx_q)), :, :]
        qb_reshaped = qb.transpose(0, 1)
        kb = k[start : (start + int(b)), :, :]
        kb_reshaped = kb.transpose(0, 1)
        vb = v[start : (start + int(b)), :, :]
        vb_reshaped = vb.transpose(0, 1)

        p = torch.matmul(qb_reshaped, kb_reshaped.transpose(-2, -1)) * sm_scale
        if causal:
            M = torch.tril(torch.ones((n_ctx_q, b), device="cuda"))
            mask = M == 0
            p[:, mask] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).to(q.dtype)
        refb = torch.matmul(p, vb_reshaped)
        ref_out[start_q : (start_q + int(n_ctx_q)), :, :] = refb.transpose(0, 1)
        start += b
        start_q += n_ctx_q

    start = 0
    start_q = 0
    ref_out_pf = torch.empty_like(q_pf, dtype=v.dtype)
    causal_pf = True  # Prefill

    for b in n_ctx_pf:
        qb = q_pf[start_q : (start_q + int(n_ctx_q_pf)), :, :]
        qb_reshaped = qb.transpose(0, 1)
        kb = k_pf[start : (start + int(b)), :, :]
        kb_reshaped = kb.transpose(0, 1)
        vb = v_pf[start : (start + int(b)), :, :]
        vb_reshaped = vb.transpose(0, 1)

        p = torch.matmul(qb_reshaped, kb_reshaped.transpose(-2, -1)) * sm_scale
        if causal_pf:
            M = torch.tril(torch.ones((n_ctx_q_pf, b)))
            mask = M == 0
            p[:, mask] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).to(q.dtype)
        refb = torch.matmul(p, vb_reshaped)
        ref_out_pf[start_q : (start_q + int(n_ctx_q_pf)), :, :] = refb.transpose(0, 1)
        start += b
        start_q += n_ctx_q_pf

    # Triton LeanAttention output
    la_decode_out, la_prefill_out = pod_attention(
        cu_ctr,
        # Decode params
        q,
        k,
        v,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        total_programs,  # common
        BLOCK_M,
        BLOCK_N,
        # False,      # decode causal,
        batch,
        sm_scale,  # common
        num_warps,  # common
        waves_per_eu,
        # Prefill params
        q_pf,
        k_pf,
        v_pf,
        Mp_pf,
        Lp_pf,
        Op_pf,
        locks_pf,
        batch_num_block_n_pf,
        BLOCK_M_pf,
        BLOCK_N_pf,
        # True,   # Prefill causal
        batch_pf,
        prefill_ratio,
        decode_ratio,
    )

    # Compare result
    atol = 1.4e-1 if init_dtype == "fp8" else 1e-2
    rtol = 1e-2 if init_dtype == "fp8" else 3e-3
    # print(f"ref_out_pf={ref_out_pf[0:15,0,0:7]}")
    # print(f"la_prefill_out={la_prefill_out[0:15,0,0:7]}")
    torch.testing.assert_close(ref_out, la_decode_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(ref_out_pf, la_prefill_out, atol=atol, rtol=rtol)


def main():

    batch = 1
    h = 64
    n_ctx_q = 16
    n_ctx = [65536]  # [262144]
    d = 128
    total_programs = 608
    init_dtype = torch.float16
    BLOCK_M = 16
    BLOCK_N = 64
    waves_per_eu = 2
    num_warps = 4
    batch_pf = 1
    n_ctx_q_pf = 4096  # 12288#1024#8192#2048#4096
    n_ctx_pf = [4096]  # [12288] #[1024] #[8192]#[2048]#[4096]
    BLOCK_M_pf = 128
    BLOCK_N_pf = 64
    assert batch == len(n_ctx)
    assert batch_pf == len(n_ctx_pf)

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")
    batch_num_block_n = get_batch_num_block_n(n_ctx, BLOCK_N, batch)

    sum_n_ctx_pf = sum(int(n) for n in n_ctx_pf)
    batch_num_block_n_pf = get_batch_num_block_n(n_ctx_pf, BLOCK_N_pf, batch_pf)

    sm_scale = 0.5

    # Decode: Allocate Tensors
    q = torch.empty((n_ctx_q * batch, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((sum_n_ctx, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((sum_n_ctx, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # Decode LeanAttention Specific Parameters
    Mp = torch.empty((total_programs, BLOCK_M), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, BLOCK_M), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, BLOCK_M, d), device=q.device, dtype=torch.float32)

    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)
    # cu_ctr = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    # MI300 (gfx942) HW_ID layout:
    # XCC_ID: 4 bits
    # SE_ID: 2 bits
    # CU_ID: 4 bits, total 10 bits for a max value of 1024
    # MI300 has 8 XCD's, 4 SE per XCD, 10 CUs per SE, 2 CUs per XCD for harvesting. Total: (10CU's * 4 SE's - 2) * 8 XCD's = 304
    # Because the ID bits are shifted, CU_ID's are not continuous values
    max_num_cu = 1024
    cu_ctr = torch.zeros((max_num_cu,), device=q.device, dtype=torch.int32)

    # Prefill: Allocate Tensors
    q_pf = torch.empty(
        (n_ctx_q_pf * batch_pf, h, d), dtype=init_dtype, device="cuda"
    ).normal_(mean=0.0, std=0.5)
    k_pf = torch.empty((sum_n_ctx_pf, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v_pf = torch.empty((sum_n_ctx_pf, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # Prefill LeanAttention Specific Parameters
    Mp_pf = torch.empty(
        (total_programs, BLOCK_M_pf), device=q.device, dtype=torch.float32
    )
    Lp_pf = torch.empty(
        (total_programs, BLOCK_M_pf), device=q.device, dtype=torch.float32
    )
    Op_pf = torch.empty(
        (total_programs, BLOCK_M_pf, d), device=q.device, dtype=torch.float32
    )

    locks_pf = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    prefill_ratio = 1
    decode_ratio = 1

    # Triton LeanAttention output
    la_decode_out, la_prefill_out = pod_attention(
        # la_decode_out, la_prefill_out, pod_cu_count = pod_attention(
        cu_ctr,
        # Decode params
        q,
        k,
        v,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        total_programs,  # common
        BLOCK_M,
        BLOCK_N,
        # False,      # decode causal,
        batch,
        sm_scale,  # common
        num_warps,  # common
        waves_per_eu,
        # Prefill params
        q_pf,
        k_pf,
        v_pf,
        Mp_pf,
        Lp_pf,
        Op_pf,
        locks_pf,
        batch_num_block_n_pf,
        BLOCK_M_pf,
        BLOCK_N_pf,
        # True,   # Prefill causal
        batch_pf,
        prefill_ratio,
        decode_ratio,
    )


if __name__ == "__main__":
    sys.exit(main())
#     benchmark_params = BenchmarkArgs()
#     args = benchmark_params.parse_args()
#     bench_streamk(args.m, args.n, args.k, args.total_programs_streamk, str_to_dtype(args.in_dtype), str_to_dtype(args.out_dtype), args.BLK_M, args.BLK_N, args.BLK_K, args.gsize_m)
