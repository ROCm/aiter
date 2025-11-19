import sys
import pytest
import torch
import random

from aiter.ops.triton.lean_atten_paged import persistent_lean_attention_paged


@pytest.mark.parametrize(
    "batch, h, n_ctx_q, n_ctx, d, total_programs, init_dtype, BLOCK_M, BLOCK_N, waves_per_eu, num_warps ",
    [
        (1, 64, 16, [65536], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 64, 16, [131072], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 64, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 64, 16, [524288], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 96, 16, [32768], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 96, 16, [65536], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 96, 16, [131072], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 96, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 96, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
        (1, 96, 16, [1048576], 16, 912, torch.float16, 16, 256, 1, 4),  #
        (1, 128, 16, [32768], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 128, 16, [65536], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 128, 16, [131072], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 128, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (1, 128, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
        (3, 64, 16, [4096, 32768, 65536], 64, 912, torch.float16, 16, 64, 2, 4),
        (
            8,
            64,
            16,
            [1024, 1024, 2048, 2048, 4096, 4096, 32768, 65536],
            64,
            912,
            torch.float16,
            16,
            64,
            2,
            4,
        ),
    ],
)
def test_persistent_lean_attention(
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
):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    torch.manual_seed(20)
    # Long seqlen (>512K) can hit memory access fault. Suspect compiler issue
    # WA with shorter d and longer BLOCK_N
    if any(item > 524288 for item in n_ctx):
        BLOCK_N = 256
        d = 16

    assert batch == len(n_ctx)

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")

    # N_CTX is a list of context lengthes for all the req in a batch
    # First, calculate #BLOCK_N for each context length "list_num_block_n"
    # Second, Convert it to a list of assumulative lengthes "list_sum_block_n"
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

    sm_scale = 0.5

    # Allocate Tensors
    q = torch.empty((h, n_ctx_q * batch, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((h, sum_n_ctx, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((h, sum_n_ctx, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    print(f"Q shape={q.shape}")
    print(f"K shape={k.shape}")
    print(f"V shape={v.shape}")
    torch.set_printoptions(threshold=10000)

    block_tables = []  # kv block tables used by lean attention
    ref_block_tables = []  # kv block tables used to compute reference results
    num_kv_blocks = sum_n_ctx // BLOCK_N + (1 if (sum_n_ctx % BLOCK_N != 0) else 0)
    kv_n_ctx = []
    last = 0
    for s in n_ctx:
        kv_blocks = s // BLOCK_N + (1 if (s % BLOCK_N != 0) else 0)
        last += kv_blocks
        kv_n_ctx.append(last)

    for head in range(h):
        ref_b = []
        ref_b_ctx = []
        kv_n_ctx_idx = 0

        r = random.sample(range(0, num_kv_blocks), num_kv_blocks)
        for i in range(num_kv_blocks):
            ref_b.append(r[i])
            if i == kv_n_ctx[kv_n_ctx_idx] - 1:
                ref_b_ctx.append(ref_b)
                ref_b = []
                kv_n_ctx_idx += 1
        block_tables.append(r)
        ref_block_tables.append(ref_b_ctx)
    kv_block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    # LeanAttention Specific Parameters
    Mp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, n_ctx_q, d), device=q.device, dtype=torch.float32)

    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    # Calculate Pytorch refence output
    start = 0
    start_q = 0
    ref_out = torch.empty_like(q, dtype=v.dtype)
    # qb = torch.empty((h, n_ctx_q*batch, d), dtype=init_dtype)
    for h in range(h):
        for b in range(len(n_ctx)):
            # print(f"h={h}")
            # print(f"n_ctx_q={N_CTX_Q}")
            # print(f"M shape: {M.shape}")
            qb = q[h, start_q : (start_q + int(n_ctx_q)), :]
            # print(f"qb shape: {qb.shape}")
            idxs = [
                ref_block_tables[h][b][kv_b_i] * BLOCK_N + b_i
                for kv_b_i in range(len(ref_block_tables[h][b]))
                for b_i in range(BLOCK_N)
            ]
            # print(f'idxs: {idxs}')
            idxs = torch.tensor(idxs, dtype=torch.int32, device="cuda")
            kb = torch.index_select(k[h], dim=0, index=idxs)
            # print(f"{kb} kb shape: {kb.shape}")
            vb = torch.index_select(v[h], dim=0, index=idxs)
            # print(f"{vb} vb shape: {vb.shape}")
            p = torch.matmul(qb, kb.transpose(0, 1)) * sm_scale
            # print(f"p shape: {p.shape}")
            p = torch.softmax(p.float(), dim=-1).to(q.dtype)
            refb = torch.matmul(p, vb)
            ref_out[h, start_q : (start_q + int(n_ctx_q)), :] = refb
            # print(f"refb={refb}")
            # print(f"refb shape: {refb.shape}")
            start += b
            start_q += n_ctx_q
        start = 0
        start_q = 0

    # Triton LeanAttention output
    la_out = persistent_lean_attention_paged(
        q,
        k,
        v,
        kv_block_tables,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        total_programs,
        BLOCK_M,
        BLOCK_N,
        batch,
        sm_scale,
        num_warps,
        waves_per_eu,
    )
    # Compare result
    atol = 1.4e-1 if init_dtype == "fp8" else 1e-2
    rtol = 1e-2 if init_dtype == "fp8" else 3e-3
    torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)


def main():
    batch = 1
    h = 64
    n_ctx_q = 16
    n_ctx = [4096]
    d = 64
    total_programs = 32
    init_dtype = torch.float16
    BLOCK_M = 16
    BLOCK_N = 64
    waves_per_eu = 1
    num_warps = 4
    assert batch == len(n_ctx)

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")

    # N_CTX is a list of context lengthes for all the req in a batch
    # First, calculate #BLOCK_N for each context length "list_num_block_n"
    # Second, Convert it to a list of assumulative lengthes "list_sum_block_n"
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

    sm_scale = 0.5

    # Allocate Tensors
    q = torch.empty((h, n_ctx_q * batch, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((h, sum_n_ctx, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((h, sum_n_ctx, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    print(f"Q shape={q.shape}")
    print(f"K shape={k.shape}")
    print(f"V shape={v.shape}")

    num_kv_blocks = sum_n_ctx // BLOCK_N + (1 if (sum_n_ctx % BLOCK_N != 0) else 0)

    block_tables = []
    for head in range(h):
        b = random.sample(range(0, num_kv_blocks), num_kv_blocks)
        block_tables.append(b)
    kv_block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")
    print(f"KV block tables shape={kv_block_tables.shape}")

    # LeanAttention Specific Parameters
    Mp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, n_ctx_q, d), device=q.device, dtype=torch.float32)

    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    # Triton LeanAttention output
    la_out = persistent_lean_attention_paged(
        q,
        k,
        v,
        kv_block_tables,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        total_programs,
        BLOCK_M,
        BLOCK_N,
        batch,
        sm_scale,
        num_warps,
        waves_per_eu,
    )

    print(f"la_out[0,0,:10]={la_out[0,0,:10]}")


if __name__ == "__main__":
    sys.exit(main())
