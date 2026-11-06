# SPDX-License-Identifier: MIT
"""gfx1250 wave32 TDM HGEMM, including masked M/N tiles and cluster split-K.

Ported from xytpai/flydsl-examples @ 5017f4c16fc46badb5fce58c485e9bd090ffe325.
The source pipeline/rounding order is preserved; the wrapper accepts AITER W[N,K].
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl, tdm_ops
from flydsl.expr.rocdl.cluster import cluster_barrier
from flydsl.expr.typing import T

from .tensor_shim import _run_compiled


class BlockSwizzle:
    def __init__(self, NUM_XCDS, NUM_PIDS_THRESHOLD, GROUP_M):
        self.NUM_XCDS = NUM_XCDS
        self.NUM_PIDS_THRESHOLD = NUM_PIDS_THRESHOLD
        self.GROUP_M = GROUP_M

    @flyc.jit
    def swizzle(self, num_pid_m, num_pid_n, pid):
        simple_m = pid // num_pid_n
        simple_n = pid % num_pid_n
        if const_expr(self.GROUP_M <= 0):
            return simple_m, simple_n
        num_xcds = self.NUM_XCDS
        swizzle_threshold = self.NUM_PIDS_THRESHOLD
        num_wg = num_pid_m * num_pid_n
        linear_id = pid
        intra_xcd = linear_id // num_xcds
        xcd = linear_id % num_xcds
        wgid = xcd * (num_wg // num_xcds) + intra_xcd
        group_m = self.GROUP_M
        wgid_per_group = group_m * num_pid_n
        group_id = wgid // wgid_per_group
        intra_group = wgid % wgid_per_group
        first_pid_m = group_id * group_m
        remaining_m = num_pid_m - first_pid_m
        group_size_m = (remaining_m < group_m).select(remaining_m, group_m)
        swizzled_n = intra_group // group_size_m
        swizzled_m = first_pid_m + (intra_group % group_size_m)
        use_simple = (num_wg < swizzle_threshold) | ((num_wg % num_xcds) != 0)
        return (
            use_simple.select(simple_m, swizzled_m),
            use_simple.select(simple_n, swizzled_n),
        )


WAVE_SIZE = 32
LDS_PAD = 8
# MI455X SPX: 8 XCDs, 256 WGPs. Same pid threshold as the gfx950 kernel.
NUM_XCDS = 8
NUM_PIDS_THRESHOLD = 256


GEMM_DTYPE_FP32 = 1
GEMM_DTYPE_BF16 = 2
GEMM_DTYPE_FP16 = 3
GEMM_DTYPE_MXFP8 = 4
GEMM_DTYPE_MXFP4 = 5
GEMM_DTYPE_NAMES = {
    GEMM_DTYPE_FP32: "fp32",
    GEMM_DTYPE_BF16: "bf16",
    GEMM_DTYPE_FP16: "fp16",
    GEMM_DTYPE_MXFP8: "mxfp8",
    GEMM_DTYPE_MXFP4: "mxfp4",
}
SUPPORTED_MMA = {
    GEMM_DTYPE_BF16: ((16, 16, 32),),
    GEMM_DTYPE_FP16: ((16, 16, 32),),
}
GEMM_DTYPE_BITS = {
    GEMM_DTYPE_FP32: 32,
    GEMM_DTYPE_BF16: 16,
    GEMM_DTYPE_FP16: 16,
    GEMM_DTYPE_MXFP8: 8,
    GEMM_DTYPE_MXFP4: 4,
}
STORE_VEC_BITS = 16 * 8


def validate_gemm_gfx1250_config(
    m,
    n,
    k,
    block_m,
    block_n,
    block_k,
    stages,
    m_waves,
    n_waves,
    group_m,
    dtype_id,
    mma_m,
    mma_n,
    mma_k,
    b_k_contiguous=True,
    split_k=1,
):
    values = (
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        mma_m,
        mma_n,
        mma_k,
    )
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError(
            "dimensions, tiles, stages, wave counts and WMMA sizes must be positive integers"
        )
    supported_mma = SUPPORTED_MMA.get(dtype_id)
    mma = (mma_m, mma_n, mma_k)
    if supported_mma is None or mma not in supported_mma:
        dtype_name = GEMM_DTYPE_NAMES.get(dtype_id, "unknown")
        raise ValueError(f"unsupported WMMA {mma_m}x{mma_n}x{mma_k} for {dtype_name}")
    if type(group_m) is not int or group_m < 0:
        raise ValueError("group_m must be a nonnegative integer")
    if not 2 <= stages <= 33:
        raise ValueError("stages must be in [2, 33] (two TDM loads per stage)")
    if block_k not in (32, 64, 128, 256, 512):
        raise ValueError("block_k must be a power of two in [32, 512]")
    waves = m_waves * n_waves
    if waves > 32:
        raise ValueError("a workgroup cannot exceed 1024 threads")
    if block_m % (m_waves * mma_m) or block_n % (n_waves * mma_n) or block_k % mma_k:
        raise ValueError("tile dimensions must be divisible by the WMMA atom")
    if not (
        1 <= block_m // (m_waves * mma_m) <= 8
        and 1 <= block_n // (n_waves * mma_n) <= 8
    ):
        raise ValueError("each wave must compute 1..8 WMMA tiles along M and N")
    if block_m % waves or block_n % waves:
        raise ValueError("each wave must own a whole number of A and B rows")
    elem_bits = GEMM_DTYPE_BITS.get(dtype_id)
    if elem_bits is None or STORE_VEC_BITS % elem_bits:
        dtype_name = GEMM_DTYPE_NAMES.get(dtype_id, "unknown")
        raise ValueError(f"unsupported element size for {dtype_name}")
    vec_elems = STORE_VEC_BITS // elem_bits
    threads = waves * WAVE_SIZE
    if (
        block_n % vec_elems
        or threads % (block_n // vec_elems)
        or block_m * block_n % (threads * vec_elems)
    ):
        raise ValueError("the output tile must support whole-workgroup 16-byte stores")
    a_stage_size = block_m * (block_k + LDS_PAD)
    b_stage_size = (
        block_n * (block_k + LDS_PAD)
        if b_k_contiguous
        else block_k * (block_n + LDS_PAD)
    )
    shared_elements = max(stages * (a_stage_size + b_stage_size), block_m * block_n)
    shared_bytes = (shared_elements * elem_bits + 7) // 8
    if shared_bytes > 320 * 1024:
        raise ValueError("pipeline exceeds gfx1250's 320 KiB LDS capacity")
    if split_k not in range(1, 10):
        raise ValueError("cluster split_k must be in 1..9")
    if k % (split_k * block_k):
        raise ValueError("K must be divisible by split_k * block_k")
    if k // split_k < (stages - 1) * block_k:
        raise ValueError("prefetch needs at least stages - 1 tiles in each K slice")
    n_blocks = (n + block_n - 1) // block_n
    if group_m * n_blocks > (2**32 - 1) // 2:
        raise ValueError("block swizzle offsets exceed the supported 32-bit range")


def make_gemm_a16w16_gfx1250_kernel_name(
    dtype_id,
    block_m,
    block_n,
    block_k,
    stages,
    m_waves,
    n_waves,
    group_m,
    has_bias,
    b_k_contiguous,
    mma_m,
    mma_n,
    mma_k,
    split_k,
):
    dtype_name = GEMM_DTYPE_NAMES[dtype_id]
    layout = "nt" if b_k_contiguous else "nn"
    return (
        f"flydsl_hgemm_tdm_{dtype_name}_t{block_m}x{block_n}x{block_k}x{stages}"
        f"_w{m_waves}x{n_waves}_gm{group_m}_bias{int(has_bias)}_l{layout}"
        f"_sk{split_k}_mma{mma_m}x{mma_n}x{mma_k}"
    )


@functools.lru_cache(maxsize=128)
def build_gemm_gfx1250(
    dtype_id: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
    b_k_contiguous: bool,
    mma_m: int,
    mma_n: int,
    mma_k: int,
    split_k: int = 1,
):
    elem_dtype = {GEMM_DTYPE_FP16: fx.Float16, GEMM_DTYPE_BF16: fx.BFloat16}.get(
        dtype_id
    )
    if elem_dtype is None:
        name = GEMM_DTYPE_NAMES.get(dtype_id, "unknown")
        raise ValueError(f"unsupported dtype_id={dtype_id} ({name})")
    elem_bits = GEMM_DTYPE_BITS[dtype_id]
    vec_elems = STORE_VEC_BITS // elem_bits
    acc_elems = mma_m * mma_n // WAVE_SIZE
    waves = m_waves * n_waves
    threads = waves * WAVE_SIZE
    # Views are A[M, K] and B[K, N]. Pad whichever dimension is contiguous.
    a_pitch = block_k + LDS_PAD
    a_stage_size = block_m * a_pitch
    a_tile_stride = (a_pitch, 1)
    if b_k_contiguous:
        b_pitch = block_k + LDS_PAD
        b_stage_size = block_n * b_pitch
        b_tile_stride = (1, b_pitch)
    else:
        b_pitch = block_n + LDS_PAD
        b_stage_size = block_k * b_pitch
        b_tile_stride = (b_pitch, 1)
    shared_elements = max(stages * (a_stage_size + b_stage_size), block_m * block_n)
    m_steps, n_steps = block_m // (m_waves * mma_m), block_n // (n_waves * mma_n)

    @fx.struct
    class SharedStorage:
        data: fx.Array[elem_dtype, shared_elements, 16]

    @flyc.kernel
    def kernel(
        a: fx.Pointer,
        b: fx.Pointer,
        out: fx.Pointer,
        bias: fx.Pointer,
        workspace: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        a_leading_stride: fx.Int32,
        b_leading_stride: fx.Int32,
        c_leading_stride: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()
        tid = fx.thread_idx.x
        wave = rocdl.wave_id()
        lane = tid % WAVE_SIZE
        block_id = fx.block_idx.x
        m_blocks = (m + block_m - 1) // block_m
        n_blocks = (n + block_n - 1) // block_n
        block_swizzle = BlockSwizzle(
            NUM_XCDS=NUM_XCDS,
            NUM_PIDS_THRESHOLD=NUM_PIDS_THRESHOLD,
            GROUP_M=group_m,
        )
        block_row, block_col = block_swizzle.swizzle(m_blocks, n_blocks, block_id)
        row_start, col_start = block_row * block_m, block_col * block_n
        wave_row = wave // n_waves * (block_m // m_waves)
        wave_col = wave % n_waves * (block_n // n_waves)

        storage = fx.SharedAllocator(static=True).allocate(SharedStorage).peek()
        shared_a = storage.data.ptr
        shared_b = shared_a + stages * a_stage_size

        # Full tensor, not the tile. A tile that hangs past M or N stays
        # block-sized in LDS; TDM zero-fills the out-of-range elements.
        def tdm_operand(
            global_pointer, global_shape, global_stride, tile_shape, tile_stride
        ):
            global_view = fx.make_view(
                global_pointer,
                fx.make_layout(global_shape, global_stride),
            )
            atom, coords = rocdl.cdna5.make_tiled_tdm_atom(
                rocdl.TensorLoad(),
                global_view,
                fx.make_layout(tile_shape, tile_stride),
                tile_shape,
                num_warps=waves,
            )
            return atom, fx.zipped_divide(coords, tile_shape)

        a_global_stride = (a_leading_stride, 1)
        if const_expr(b_k_contiguous):
            b_global_stride = (1, b_leading_stride)
        else:
            b_global_stride = (b_leading_stride, 1)
        atom_a, coords_a = tdm_operand(
            a,
            (m, k),
            a_global_stride,
            (block_m, block_k),
            a_tile_stride,
        )
        atom_b, coords_b = tdm_operand(
            b,
            (k, n),
            b_global_stride,
            (block_k, block_n),
            b_tile_stride,
        )

        def copy_operand(
            atom,
            shared_pointer,
            tile_shape,
            tile_stride,
            stage_size,
            dst_stage,
            src_stage,
            coord,
        ):
            shared_view = fx.make_view(
                shared_pointer,
                fx.make_layout((tile_shape, stages), (tile_stride, stage_size)),
            )
            dst, src = rocdl.cdna5.tdm_partition(
                atom,
                wave,
                fx.make_layout(waves, 1),
                shared_view,
                coord,
            )
            fx.copy(atom, src[None, src_stage], dst[None, dst_stage])

        def copy_tile(dst_stage, src_stage):
            # Rest modes follow the view: A is (M tile, K tile), B is (K tile, N tile).
            copy_operand(
                atom_b,
                shared_b,
                (block_k, block_n),
                b_tile_stride,
                b_stage_size,
                dst_stage,
                src_stage,
                coords_b[None, (None, block_col)],
            )
            copy_operand(
                atom_a,
                shared_a,
                (block_m, block_k),
                a_tile_stride,
                a_stage_size,
                dst_stage,
                src_stage,
                coords_a[None, (block_row, None)],
            )
            rocdl.sched_barrier(0)

        def wait_tile(pending_stages):
            tdm_ops.tensor_wait(pending_stages * 2)
            # Per-wave completion is not enough: all waves share the tile.
            # This also protects the previous tile's LDS reads before recycling.
            fx.gpu.barrier()

        mma = fx.make_mma_atom(rocdl.WMMA(mma_m, mma_n, mma_k, elem_dtype, fx.Float32))
        accumulators = [
            fx.make_rmem_tensor(acc_elems, fx.Float32)
            for _ in range_constexpr(m_steps * n_steps)
        ]
        for acc in accumulators:
            acc.store(fx.full(acc_elems, 0.0, fx.Float32))

        def load_fragment(
            shared_pointer, stage_size, stage, row, column, k_contiguous=True
        ):
            if const_expr(k_contiguous):
                # A is always k_contiguous
                offset = (
                    stage * stage_size
                    + (row + lane % 16) * (block_k + LDS_PAD)
                    + column
                    + lane // 16 * 8
                )
                low = fx.ptr_load(
                    shared_pointer + offset, result_type=T.vec(8, elem_dtype.ir_type)
                )
                high = fx.ptr_load(
                    shared_pointer + offset + 16,
                    result_type=T.vec(8, elem_dtype.ir_type),
                )
                values = low.shuffle(high, list(range(16)))
            else:
                offset = (
                    stage * stage_size
                    + row
                    + lane % 16
                    + (column + lane // 16 * 8) * b_pitch
                )
                values = fx.Vector.from_elements(
                    [
                        fx.ptr_load(
                            shared_pointer + offset + (i % 8 + i // 8 * 16) * b_pitch
                        )
                        for i in range_constexpr(16)
                    ]
                )
            fragment = fx.make_rmem_tensor(16, elem_dtype)
            fragment.store(values)
            return fragment

        def compute_tile(stage):
            for ki in range_constexpr(block_k // mma_k):
                b_fragments = [
                    load_fragment(
                        shared_b,
                        b_stage_size,
                        stage,
                        wave_col + ni * mma_n,
                        ki * mma_k,
                        k_contiguous=b_k_contiguous,
                    )
                    for ni in range_constexpr(n_steps)
                ]
                rocdl.sched_barrier(0)
                for mi in range_constexpr(m_steps):
                    a_fragment = load_fragment(
                        shared_a, a_stage_size, stage, wave_row + mi * mma_m, ki * mma_k
                    )
                    rocdl.sched_barrier(0)
                    for ni in range_constexpr(n_steps):
                        acc = accumulators[mi * n_steps + ni]
                        fx.gemm(mma, acc, a_fragment, b_fragments[ni], acc)

        # Each cluster rank owns an equal K slice. Tile indices stay in the
        # full-tensor coordinate space.
        k_rank = fx.block_idx.y
        k_tiles = (k // split_k) // block_k
        k_tile_base = k_rank * k_tiles
        for stage in range_constexpr(stages - 1):
            copy_tile(stage, k_tile_base + fx.Int32(stage))

        # Two tiles per iteration. Double buffering retains literal LDS offsets;
        # deeper pipelines advance the ring modulo stages.
        main_tiles = k_tiles - (stages - 1)
        current_stage = fx.Int32(0)
        for pair in range(main_tiles // 2):
            for step in range_constexpr(2):
                read_stage = step if stages == 2 else current_stage
                wait_tile(stages - 2)
                copy_tile(
                    (read_stage + stages - 1) % stages,
                    k_tile_base + pair * 2 + step + stages - 1,
                )
                compute_tile(read_stage)
                current_stage = (current_stage + 1) % stages

        # A one-trip loop keeps the accumulators loop-carried instead of
        # materializing both sides of a predicated remainder branch.
        for remaining in range(main_tiles % 2):
            wait_tile(stages - 2)
            copy_tile(
                (current_stage + stages - 1) % stages,
                k_tile_base + main_tiles + stages - 2,
            )
            compute_tile(current_stage)
            current_stage = (current_stage + 1) % stages

        # Drain: retire the remaining loads in order without issuing more.
        for tail in range_constexpr(stages - 1):
            wait_tile(stages - 2 - tail)
            compute_tile(current_stage)
            current_stage = (current_stage + 1) % stages

        # Reuse the A/B LDS arena for coalesced output stores.
        # Native WMMA output: column=lane%16, row=lane//16*acc_elems+register_index.
        fx.gpu.barrier()
        for mi in range_constexpr(m_steps):
            for ni in range_constexpr(n_steps):
                values = accumulators[mi * n_steps + ni].load().to(elem_dtype)
                row = wave_row + mi * mma_m + lane // 16 * acc_elems
                col = wave_col + ni * mma_n + lane % 16
                for i in range_constexpr(acc_elems):
                    fx.ptr_store(values[i], shared_a + (row + i) * block_n + col)
        fx.gpu.barrier()

        def commit_output(values, row, col):
            destination = (
                out
                + fx.Int64(row_start + row) * fx.Int64(c_leading_stride)
                + col_start
                + col
            )
            # Interior tiles store one 16-byte vector along N. An edge tile may
            # cover rows or columns past M or N; those elements stay unstored.
            if (row_start + block_m <= m) and (col_start + block_n <= n):
                if const_expr(has_bias):
                    bias_values = fx.ptr_load(
                        bias + col_start + col,
                        result_type=T.vec(vec_elems, elem_dtype.ir_type),
                    )
                    # Round C to input dtype BEFORE bias.
                    values = (values.to(fx.Float32) + bias_values.to(fx.Float32)).to(
                        elem_dtype
                    )
                fx.make_view(destination, fx.make_layout(vec_elems, 1)).store(values)
            else:
                for element in range_constexpr(vec_elems):
                    if (row_start + row < m) and (col_start + col + element < n):
                        value = values[element]
                        if const_expr(has_bias):
                            bias_value = fx.ptr_load(
                                bias + col_start + col + element,
                                result_type=elem_dtype.ir_type,
                            )
                            value = (
                                value.to(fx.Float32) + bias_value.to(fx.Float32)
                            ).to(elem_dtype)
                        fx.ptr_store(value, destination + element)

        tile_elems = block_m * block_n
        if const_expr(split_k == 1):
            for i in range_constexpr(tile_elems // (threads * vec_elems)):
                offset = (i * threads + tid) * vec_elems
                row, col = offset // block_n, offset % block_n
                values = fx.ptr_load(
                    shared_a + offset, result_type=T.vec(vec_elems, elem_dtype.ir_type)
                )
                commit_output(values, row, col)
        else:
            for i in range_constexpr(tile_elems // (threads * vec_elems)):
                offset = (i * threads + tid) * vec_elems
                row, col = offset // block_n, offset % block_n
                values = fx.ptr_load(
                    shared_a + offset, result_type=T.vec(vec_elems, elem_dtype.ir_type)
                )
                partial = (
                    workspace
                    + fx.Int64(block_id) * (split_k * tile_elems)
                    + fx.Int64(k_rank) * tile_elems
                    + row * block_n
                    + col
                )
                fx.make_view(partial, fx.make_layout(vec_elems, 1)).store(values)
            cluster_barrier()
            if k_rank == 0:
                for i in range_constexpr(tile_elems // (threads * vec_elems)):
                    offset = (i * threads + tid) * vec_elems
                    row, col = offset // block_n, offset % block_n
                    total = fx.ptr_load(
                        workspace
                        + fx.Int64(block_id) * (split_k * tile_elems)
                        + offset,
                        result_type=T.vec(vec_elems, elem_dtype.ir_type),
                    ).to(fx.Float32)
                    for part in range_constexpr(split_k - 1):
                        partial = fx.ptr_load(
                            workspace
                            + fx.Int64(block_id) * (split_k * tile_elems)
                            + (part + 1) * tile_elems
                            + offset,
                            result_type=T.vec(vec_elems, elem_dtype.ir_type),
                        )
                        total = total + partial.to(fx.Float32)
                    commit_output(total.to(elem_dtype), row, col)

    @flyc.jit
    def launch(
        a: fx.Pointer,
        b: fx.Pointer,
        out: fx.Pointer,
        bias: fx.Pointer,
        workspace: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        a_leading_stride: fx.Int32,
        b_leading_stride: fx.Int32,
        c_leading_stride: fx.Int32,
        stream: fx.Stream,
    ):
        mn_tiles = (m + block_m - 1) // block_m * ((n + block_n - 1) // block_n)
        if split_k == 1:
            kernel(
                a,
                b,
                out,
                bias,
                workspace,
                m,
                n,
                k,
                a_leading_stride,
                b_leading_stride,
                c_leading_stride,
            ).launch(
                grid=(mn_tiles, 1, 1),
                block=(threads, 1, 1),
                stream=stream,
            )
        else:
            kernel(
                a,
                b,
                out,
                bias,
                workspace,
                m,
                n,
                k,
                a_leading_stride,
                b_leading_stride,
                c_leading_stride,
                value_attrs={"rocdl.cluster_dims": f"1,{split_k},1"},
            ).launch(
                grid=(mn_tiles, split_k, 1),
                block=(threads, 1, 1),
                stream=stream,
                cluster=(1, split_k, 1),
            )

    kernel._func.__name__ = make_gemm_a16w16_gfx1250_kernel_name(
        dtype_id,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        group_m,
        has_bias,
        b_k_contiguous,
        mma_m,
        mma_n,
        mma_k,
        split_k,
    )
    return launch


def gemm_a16w16_tdm(
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 128,
    stages: int = 2,
    m_waves: int = 4,
    n_waves: int = 4,
    group_m: int = 0,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
    split_k: int = 1,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Compute C = A[M,K] @ W[N,K].T, optionally adding bias.

    A must be K-contiguous; W may be K- or N-contiguous (NT/NN source layout).
    The leading strides may include aligned row padding. K must be divisible
    by split_k * block_k. Cluster split-K rounds each partial to input dtype;
    the sum is rounded before bias. Non-split also rounds C before bias.
    """
    if a.ndim != 2 or w.ndim != 2:
        raise ValueError("expected A[M,K] and W[N,K]")
    b = w.t()
    layout = "nt" if w.stride(1) == 1 else "nn"
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise ValueError("A and B must have the same FP16 or BF16 dtype")
    dtype_id = GEMM_DTYPE_FP16 if a.dtype is torch.float16 else GEMM_DTYPE_BF16
    if a.device != b.device or a.device.type != "cuda":
        raise ValueError("A and B must be on the same ROCm device")
    layout = layout.lower()
    if layout not in ("nn", "nt"):
        raise ValueError("layout must be 'nn' or 'nt'")
    m, k = a.shape
    n = b.shape[1]
    validate_gemm_gfx1250_config(
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        group_m,
        dtype_id,
        mma_m,
        mma_n,
        mma_k,
        layout == "nt",
        split_k,
    )
    a_leading_stride = _leading_stride(
        a, 1, 0, k, f"A does not satisfy the {layout.upper()} layout"
    )
    if layout == "nt":
        b_leading_stride = _leading_stride(
            b, 0, 1, k, "B does not satisfy the NT layout"
        )
    else:
        b_leading_stride = _leading_stride(
            b, 1, 0, n, "B does not satisfy the NN layout"
        )
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if out.shape != (m, n) or out.dtype != a.dtype or out.device != a.device:
        raise ValueError("out must be [M,N] with the input dtype and device")
    c_leading_stride = _leading_stride(out, 1, 0, n, "out must be row-major [M,N]")
    if bias is not None and (
        bias.shape != (n,)
        or bias.dtype != a.dtype
        or bias.device != a.device
        or not bias.is_contiguous()
    ):
        raise ValueError("bias must be contiguous [N] with the input dtype and device")
    tensors = [a, b, out] + ([] if bias is None else [bias])
    if any(t.data_ptr() % 16 for t in tensors):
        raise ValueError("all pointers must be 16-byte aligned")
    if (
        max(
            m * a_leading_stride,
            k * b_leading_stride,
            n * b_leading_stride,
            m * c_leading_stride,
        )
        > (2**32 - 1) // 2
    ):
        raise ValueError("tensor offsets exceed the supported 32-bit range")
    out_lo, out_hi = _storage_span(out)
    if any(
        _overlaps(out_lo, out_hi, t) for t in [a, b] + ([] if bias is None else [bias])
    ):
        raise ValueError("out must not overlap the inputs")
    with torch.cuda.device(a.device):
        arch = torch.cuda.get_device_properties(a.device).gcnArchName.split(":")[0]
        if arch != "gfx1250":
            raise ValueError(f"TDM GEMM requires gfx1250, got {arch}")
        if stream is None:
            stream = torch.cuda.current_stream()
        if stream.device != a.device:
            raise ValueError("stream must belong to the input device")
        dtype = fx.Float16 if dtype_id == GEMM_DTYPE_FP16 else fx.BFloat16
        mn_tiles = ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n)
        workspace = (
            out
            if split_k == 1
            else torch.empty(
                (mn_tiles * split_k * block_m * block_n,),
                device=a.device,
                dtype=a.dtype,
            )
        )
        pointers = [
            flyc.from_c_void_p(dtype, t.data_ptr())
            for t in (a, b, out, a if bias is None else bias, workspace)
        ]
        has_bias = bias is not None
        b_k_contiguous = layout == "nt"
        launch = build_gemm_gfx1250(
            dtype_id,
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            group_m,
            has_bias,
            b_k_contiguous,
            mma_m,
            mma_n,
            mma_k,
            split_k,
        )
        dispatch_args = (
            *pointers,
            m,
            n,
            k,
            a_leading_stride,
            b_leading_stride,
            c_leading_stride,
            stream,
        )
        _run_compiled(launch, *dispatch_args)
        # The temporary belongs to this wrapper, not the caller. Keep it alive
        # until a launch on a non-current stream has finished using it.
        if split_k > 1 and stream != torch.cuda.current_stream(a.device):
            workspace.record_stream(stream)

    return out


def _leading_stride(tensor, unit_dim, leading_dim, min_leading, message):
    unit, leading = tensor.stride(unit_dim), tensor.stride(leading_dim)
    if unit != 1 or leading < min_leading or leading * tensor.element_size() % 16 != 0:
        raise ValueError(
            f"{message}: expected stride({unit_dim}) == 1 and a 16-byte-aligned "
            f"leading stride({leading_dim}) >= {min_leading}; got stride={tuple(tensor.stride())}"
        )
    return int(leading)


def _storage_span(tensor):
    last = (tensor.shape[0] - 1) * tensor.stride(0) + (
        tensor.shape[1] - 1
    ) * tensor.stride(1)
    start = tensor.data_ptr()
    return start, start + (last + 1) * tensor.element_size()


def _overlaps(out_lo, out_hi, tensor):
    lo, hi = (
        _storage_span(tensor)
        if tensor.ndim == 2
        else (
            tensor.data_ptr(),
            tensor.data_ptr() + tensor.numel() * tensor.element_size(),
        )
    )
    return out_lo < hi and lo < out_hi
