# Gluon AMD GPU Examples Reference

## Source Files

- GFX1250 GEMM: [`third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py)
- GFX1250 common: [`third_party/amd/python/examples/gluon/f16_gemm_common_gfx1250.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/python/examples/gluon/f16_gemm_common_gfx1250.py)
- GFX1250 FA: [`third_party/amd/python/examples/gluon/f16_fa_gfx1250.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/python/examples/gluon/f16_fa_gfx1250.py)
- GFX1250 StreamK: [`third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/python/examples/gluon/f16_gemm_streamk_gfx1250.py)
- GFX1250 warp pipeline: [`third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/python/examples/gluon/f16_gemm_warp_pipeline_gfx1250.py)
- MXFP GEMM: [`third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py)
- MXFP FA: [`third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py)
- GFX1250 test: [`third_party/amd/python/test/test_gluon_gfx1250.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/python/test/test_gluon_gfx1250.py)
- Tutorials: [`python/tutorials/gluon/`](https://github.com/triton-lang/triton/tree/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/tutorials/gluon) (`01-intro.py` through `12-cluster-launch-control.py`)
- Attention example: [`python/examples/gluon/01-attention-forward.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/examples/gluon/01-attention-forward.py)

## Complete GFX1250 GEMM with TDM Pipelining

```python
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
import math

def build_layouts(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, TRANSPOSE_B):
    """Build all layouts for a GFX1250 GEMM kernel."""
    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)

    WMMA_LAYOUT = ttgl.amd.AMDWMMALayout(3, True, warp_bases, [], [16, 16, 32])

    SHARED_A = ttgl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0])
    if not TRANSPOSE_B:
        SHARED_B = ttgl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0])
    else:
        SHARED_B = ttgl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0])

    OPERAND_A = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_B = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    return SHARED_A, SHARED_B, WMMA_LAYOUT, OPERAND_A, OPERAND_B


@gluon.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr,
                SHARED_LAYOUT_A: ttgl.constexpr, SHARED_LAYOUT_B: ttgl.constexpr,
                ACC_LAYOUT: ttgl.constexpr,
                OPERAND_A: ttgl.constexpr, OPERAND_B: ttgl.constexpr):

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # Create tensor descriptors for TDM
    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K), layout=SHARED_LAYOUT_A)

    b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr,
        shape=(N, K) if TRANSPOSE_B else (K, N),
        strides=(stride_bn, stride_bk) if TRANSPOSE_B else (stride_bk, stride_bn),
        block_shape=(BLOCK_N, BLOCK_K) if TRANSPOSE_B else (BLOCK_K, BLOCK_N),
        layout=SHARED_LAYOUT_B)

    # Allocate multi-buffered shared memory
    a_buf = ttgl.allocate_shared_memory(
        a_desc.dtype, [NUM_BUFFERS] + a_desc.block_shape, a_desc.layout)
    b_buf = ttgl.allocate_shared_memory(
        b_desc.dtype, [NUM_BUFFERS] + b_desc.block_shape, b_desc.layout)

    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N
    accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=ACC_LAYOUT)

    producer = 0
    consumer = 0

    # Prologue: fill pipeline
    for _ in ttgl.static_range(NUM_BUFFERS - 1):
        buf_idx = producer % NUM_BUFFERS
        k_off = producer * BLOCK_K
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, k_off], a_buf.index(buf_idx))
        if TRANSPOSE_B:
            ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, k_off], b_buf.index(buf_idx))
        else:
            ttgl.amd.gfx1250.tdm.async_load(b_desc, [k_off, off_bn], b_buf.index(buf_idx))
        producer += 1

    # Main loop
    for _ in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        # Issue next load
        buf_idx = producer % NUM_BUFFERS
        k_off = producer * BLOCK_K
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, k_off], a_buf.index(buf_idx))
        if TRANSPOSE_B:
            ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, k_off], b_buf.index(buf_idx))
        else:
            ttgl.amd.gfx1250.tdm.async_load(b_desc, [k_off, off_bn], b_buf.index(buf_idx))
        producer += 1

        # Wait and compute
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        c_buf_idx = consumer % NUM_BUFFERS
        a_tile = a_buf.index(c_buf_idx).load(layout=OPERAND_A)
        if TRANSPOSE_B:
            b_tile = b_buf.index(c_buf_idx).permute([1, 0]).load(layout=OPERAND_B)
        else:
            b_tile = b_buf.index(c_buf_idx).load(layout=OPERAND_B)
        accumulator = ttgl.amd.gfx1250.wmma(a_tile, b_tile, accumulator)
        consumer += 1

    # Epilogue: drain pipeline
    for i in ttgl.static_range(NUM_BUFFERS - 1):
        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        c_buf_idx = consumer % NUM_BUFFERS
        a_tile = a_buf.index(c_buf_idx).load(layout=OPERAND_A)
        if TRANSPOSE_B:
            b_tile = b_buf.index(c_buf_idx).permute([1, 0]).load(layout=OPERAND_B)
        else:
            b_tile = b_buf.index(c_buf_idx).load(layout=OPERAND_B)
        accumulator = ttgl.amd.gfx1250.wmma(a_tile, b_tile, accumulator)
        consumer += 1

    # Store result
    offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, ACC_LAYOUT))
    offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, ACC_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)
```

## GFX1250 Warp-Specialized GEMM Pattern

```python
@gluon.jit
def producer_fn(args):
    """TDM loads with mbarrier signaling."""
    K = args.a_desc.shape[1]
    num_k_tiles = ttgl.cdiv(K, args.BLOCK_K)

    empty_phase = PhaseCounter.create(args.NUM_BUFFERS, args.NUM_BUFFERS)

    for k in range(num_k_tiles):
        buf_idx = k % args.NUM_BUFFERS
        ttgl.amd.gfx1250.mbarrier.wait(args.empty_bars.index(buf_idx), empty_phase.phase())
        ttgl.amd.gfx1250.tdm.async_load(args.a_desc, [0, k * args.BLOCK_K],
                                         args.a_buf.index(buf_idx))
        ttgl.amd.gfx1250.tdm.async_load(args.b_desc, [0, k * args.BLOCK_K],
                                         args.b_buf.index(buf_idx),
                                         mbarrier=args.ready_bars.index(buf_idx))
        empty_phase = empty_phase.next()


@gluon.jit
def consumer_fn(args, c_ptr, M, N, stride_cm, stride_cn, pid_m, pid_n):
    """WMMA compute with mbarrier synchronization."""
    K = args.a_desc.shape[1]
    OPERAND_A: ttgl.constexpr = ttgl.DotOperandLayout(0, args.WMMA_LAYOUT, 8)
    OPERAND_B: ttgl.constexpr = ttgl.DotOperandLayout(1, args.WMMA_LAYOUT, 8)
    BLOCK_M: ttgl.constexpr = args.a_desc.block_shape[0]
    BLOCK_N: ttgl.constexpr = args.b_desc.block_shape[1]

    acc = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=args.c_dtype, layout=args.WMMA_LAYOUT)
    ready_phase = PhaseCounter.create(0, args.NUM_BUFFERS)

    for k in range(ttgl.cdiv(K, args.BLOCK_K)):
        buf_idx = k % args.NUM_BUFFERS
        ttgl.amd.gfx1250.mbarrier.wait(args.ready_bars.index(buf_idx), ready_phase.phase())
        a = args.a_buf.index(buf_idx).load(layout=OPERAND_A)
        b = args.b_buf.index(buf_idx).load(layout=OPERAND_B)
        acc = ttgl.amd.gfx1250.wmma(a, b, acc)
        ttgl.amd.gfx1250.mbarrier.arrive(args.empty_bars.index(buf_idx), count=1)
        ready_phase = ready_phase.next()

    # Store output...


@gluon.jit
def ws_gemm_kernel(...):
    # Setup descriptors, buffers, mbarriers...
    WARP_SIZE = 32
    PRODUCER_WARPS = NUM_WARPS // 2
    CONSUMER_WARPS = NUM_WARPS // 2

    # Init barriers
    for i in ttgl.static_range(NUM_BUFFERS):
        ttgl.amd.gfx1250.mbarrier.init(empty_bars.index(i), count=CONSUMER_WARPS * WARP_SIZE)
        ttgl.amd.gfx1250.mbarrier.init(ready_bars.index(i), count=PRODUCER_WARPS)

    # Launch specialized warps
    ttgl.warp_specialize([
        (consumer_fn, (args, c_ptr, M, N, stride_cm, stride_cn, pid_m, pid_n)),
        (producer_fn, (args,)),
    ], [PRODUCER_WARPS])
```

## CDNA3/4 MFMA GEMM with Warp Pipeline

```python
@gluon.jit
def mfma_gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                     stride_am, stride_ak, stride_bk, stride_bn,
                     stride_cm, stride_cn,
                     BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                     MFMA_LAYOUT: ttgl.constexpr,
                     OPERAND_A: ttgl.constexpr, OPERAND_B: ttgl.constexpr,
                     BLOCKED_LAYOUT: ttgl.constexpr):
    from triton.experimental.gluon.language.amd.warp_pipeline import warp_pipeline_stage as wps

    pid = ttgl.program_id(axis=0)
    pid_m = pid % ttgl.cdiv(M, BLOCK_M)
    pid_n = pid // ttgl.cdiv(M, BLOCK_M)

    # Create offset tensors for buffer_load
    offs_am = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_bn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))

    acc = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=ttgl.float32, layout=MFMA_LAYOUT)

    for k in range(0, ttgl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K

        with wps("load", priority=3):
            a_offs = (offs_am[:, None] * stride_am + (k_off + ttgl.arange(0, BLOCK_K)) * stride_ak)
            b_offs = ((k_off + ttgl.arange(0, BLOCK_K))[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a = ttgl.amd.cdna3.buffer_load(a_ptr, a_offs.to(ttgl.int32))
            b = ttgl.amd.cdna3.buffer_load(b_ptr, b_offs.to(ttgl.int32))

        with wps("compute", priority=0):
            # Need layout conversion for MFMA dot operands
            acc = ttgl.amd.cdna3.mfma(a, b, acc)

    # Store result...
```

## CDNA4 Async Copy + MFMA Pattern

```python
@gluon.jit
def cdna4_gemm_kernel(...):
    # Allocate shared memory
    smem_a = ttgl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_K], SHARED_A)
    smem_b = ttgl.allocate_shared_memory(dtype, [BLOCK_K, BLOCK_N], SHARED_B)

    for k in range(num_k_tiles):
        # Async copy global -> shared (prefer buffer_load_to_shared)
        ttgl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a, a_ptr, a_offsets)
        ttgl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b, b_ptr, b_offsets)
        ttgl.amd.cdna4.async_copy.commit_group()
        ttgl.amd.cdna4.async_copy.wait_group(0)

        # Load from shared with relaxed semantics
        a = ttgl.amd.cdna4.async_copy.load_shared_relaxed(smem_a, OPERAND_A)
        b = ttgl.amd.cdna4.async_copy.load_shared_relaxed(smem_b, OPERAND_B)

        acc = ttgl.amd.cdna4.mfma(a, b, acc)
```

## Kernel Launch

```python
import triton

grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
kernel = gemm_kernel[grid](
    a_device, b_device, c_device,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B,
    SHARED_LAYOUT_A=SHARED_A, SHARED_LAYOUT_B=SHARED_B,
    ACC_LAYOUT=WMMA_LAYOUT,
    OPERAND_A=OPERAND_A, OPERAND_B=OPERAND_B,
    num_warps=num_warps,
    waves_per_eu=num_warps // 4,
    num_ctas=1,
)
```

## Static Profiling

```python
import re

amdgcn = kernel.asm['amdgcn']
sgpr_count = int(re.search(r'\.sgpr_count:\s+(\d+)', amdgcn).group(1))
vgpr_count = int(re.search(r'\.vgpr_count:\s+(\d+)', amdgcn).group(1))
vgpr_spill_count = int(re.search(r'\.vgpr_spill_count:\s+(\d+)', amdgcn).group(1))
occupancy = int(re.search(r';\s+Occupancy:\s+(\d+)', amdgcn).group(1))
```
