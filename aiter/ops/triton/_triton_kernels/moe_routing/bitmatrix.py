import torch
import triton
import triton.language as tl


@triton.jit
def vpopc(x):
    """
    Vertical popcount
    Input  x : uint32[..., N]
    Output y : uint32[..., 32]
    semantics : y[..., i] = sum_j((x[..., j] >> i) & 1)
    credits: @apgoucher
    """

    tl.static_assert(
        x.dtype == tl.uint32, "x should consist of 32-bit unsigned integers"
    )

    BLOCK_N: tl.constexpr = x.shape[-1]  # summation axis
    BATCHES: tl.constexpr = x.numel // BLOCK_N  # number of batches
    if BLOCK_N >= 8:
        sa1: tl.constexpr = 8
    else:
        sa1: tl.constexpr = BLOCK_N
    # create 8-way sums in 4-bit fields:
    y = tl.reshape(x, [BATCHES, BLOCK_N // sa1, sa1, 1])
    y = (y >> tl.arange(0, 4)[None, None, None, :]) & 0x11111111
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // sa1, 4]
    if BLOCK_N >= 128:
        sa2: tl.constexpr = 16
    else:
        sa2: tl.constexpr = BLOCK_N // sa1
    # create 128-way sums in 8-bit fields:
    y = tl.reshape(y, [BATCHES, BLOCK_N // (sa1 * sa2), sa2, 1, 4])
    y = (y >> (4 * tl.arange(0, 2))[None, None, None, :, None]) & 0x0F0F0F0F
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // (sa1 * sa2), 2, 4]
    sa3: tl.constexpr = BLOCK_N // (sa1 * sa2)
    # create N-way sums in 32-bit fields:
    y = tl.reshape(y, [BATCHES, 1, sa3, 8])
    y = (y >> (8 * tl.arange(0, 4))[None, :, None, None]) & 0x000000FF
    y = tl.sum(y, 2)  # [BATCHES, 4, 8]
    y = tl.reshape(y, x.shape[:-1] + [32])
    return y


@triton.jit
def _sum_bitmatrix_memset(Ret, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(Ret + offs, 0)


@triton.jit
def _sum_bitmatrix_rows(
    B,
    shape_bm,
    stride_bm,
    stride_bn,  # input bitmatrix
    Ret,
    Partials,
    stride_pm,
    stride_pn,
    shape_pn,
    num_pids_m,  # outputs
    BLOCK_MM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):

    tl.static_assert(BLOCK_MM % BLOCK_M == 0)
    TILE_SIZE: tl.constexpr = BLOCK_MM // BLOCK_M
    if isinstance(shape_bm, tl.tensor) and shape_bm.dtype.is_ptr():
        shape_bm = tl.load(shape_bm)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_MM + tl.arange(0, BLOCK_MM)
    offs_n = pid_n * 32 + tl.arange(0, 32)
    n_rows = shape_bm
    bits = tl.load(
        B + pid_n * stride_bn + offs_m * stride_bm, mask=offs_m < n_rows, other=0
    )
    bits = tl.reshape(bits, [TILE_SIZE, BLOCK_M])
    ret = vpopc(bits)  # [TILE_SIZE, 32]

    offs_t = pid_m * TILE_SIZE + tl.arange(0, TILE_SIZE)

    tl.atomic_add(Ret + offs_n, tl.sum(ret, 0), sem="relaxed")

    curr = tl.cumsum(ret, 0) - ret
    tl.atomic_add(
        Partials + offs_t[:, None] * stride_pm + offs_n[None, :] * stride_pn,
        curr,
        sem="relaxed",
    )
    curr = tl.sum(ret, 0, keep_dims=True)
    for i in range(pid_m + 1, num_pids_m):
        offs_t = i * TILE_SIZE + tl.arange(0, TILE_SIZE)
        tl.atomic_add(
            Partials + offs_t[:, None] * stride_pm + offs_n[None, :] * stride_pn,
            curr,
            sem="relaxed",
        )

    # tl.store(Partials + offs_t[:, None] * stride_pm + offs_n[None, :] * stride_pn, ret)


@triton.jit
def _sum_bitmatrix_rows_fused(
    B,
    shape_bm,
    stride_bm,
    stride_bn,
    Ret,
    N_BLKS_BITMATRIX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    if isinstance(shape_bm, tl.tensor) and shape_bm.dtype.is_ptr():
        shape_bm = tl.load(shape_bm)
    for i in tl.static_range(N_BLKS_BITMATRIX):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = i * 32 + tl.arange(0, 32)
        n_rows = shape_bm
        if EVEN_M:
            bits = tl.load(B + i * stride_bn + offs_m * stride_bm)
        else:
            bits = tl.load(
                B + i * stride_bn + offs_m * stride_bm, mask=offs_m < n_rows, other=0
            )
        bits = tl.reshape(bits, [1, BLOCK_M])
        ret = vpopc(bits)  # [1, 32]
        ret = tl.reshape(ret, [32])

        tl.store(Ret + offs_n, ret)
