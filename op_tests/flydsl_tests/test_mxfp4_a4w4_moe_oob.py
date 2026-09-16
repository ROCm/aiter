# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone repro for the MXFP4 A4W4 MoE gemm2 illegal-memory-access (OOB read).

WHAT BREAKS
-----------
The FlyDSL MXFP4 A4W4 atomic MoE gemm2 kernel
(`aiter/ops/flydsl/kernels/mxfp4_gemm2.py`) hard-faults with a HIP "illegal
memory access" under load. Root cause is a broken *block-alignment invariant*
between the sort stage and the gemm kernel -- NOT a bug in the arithmetic:

* gemm2's atomic epilog runs the NON-persistent grid
  (`_persistent = epilog in ("nonatomic","nonatomic_mxfp4")` -> atomic is not
  persistent). That branch issues `_issue_all_a_loads(m_row0)` UNCONDITIONALLY
  for every grid block -- including the trailing padding block -- BEFORE the
  `if bx_i32 < bound` guard, relying on the A buffer descriptor to clamp.
* The A descriptor is sized `_aq_num = max_m_blocks * (BM * K_HALF)`, i.e. to
  `max_m_blocks * BM` rows, where `max_m_blocks = ceil(max_sorted / BM)`
  (see `aiter/ops/flydsl/mxfp4_gemm2_kernels.py::flydsl_mxfp4_gemm2`, which
  passes `max_m_blocks` and the raw A `data_ptr()` to the compiled kernel).
* The A buffer (`inter_sorted_quant`) is allocated with exactly `max_sorted`
  rows (`aiter/fused_moe.py::_mxfp4_a4w4_stage1`).

So the descriptor addresses `ceil(max_sorted/BM)*BM` rows while the allocation
has only `max_sorted`. When `max_sorted` is NOT a multiple of BM the descriptor
over-reads `ceil(max_sorted/BM)*BM - max_sorted` rows past the buffer. AMD
`buffer_load` bounds-checks against the descriptor's num_records (the oversized
figure), NOT the real allocation, so those lanes issue real memory reads and
fault **iff** the over-read rows land on an unmapped page. That is a small
over-read (~5 KB at production shapes), far below the ~2 MB allocator rounding,
so in a normal allocator it usually hides in mapped slack -- which is exactly
why the production crash was an intermittent, layout-dependent heisenbug.

`max_sorted` comes from the sort. `_adaptive_moe_sort` always rounds it up to a
multiple of BM (invariant HELD). Guilty commit fe4814c64 (PR #4526) added
`_aux_uses_opus()`, which for `block_size != 16` diverts the A4W4 atomic path to
the Opus sort, whose

    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk

is NOT a multiple of block_size (invariant BROKEN).

WHAT THESE TESTS DO
-------------------
Three complementary tests, cheapest first:

1. `test_moe_sorting_opus_row_extent_block_aligned` drives the exact production
   sort entry (`moe_sorting(..., output_aux=AUX_SORT_OPUS)` -- the call the real
   MXFP4 A4W4 path makes at fused_moe.py) and asserts the sole condition the gemm
   kernels require: the sorted/intermediate row extent is a multiple of
   block_size. Deterministic, needs only a GPU (no model). PRE-FIX it FAILS
   whenever `(M*topk - topk)` is not a multiple of block_size; POST-FIX it PASSES
   for all block_size. Production uses block_size (BM) == 32.

2. `test_mxfp4_gemm2_descriptor_fits_allocation` encodes the same invariant as
   pure arithmetic (no GPU) so the failure message names the exact rows/bytes
   over-read for the K2.7 fault shape.

3. `test_mxfp4_a4w4_gemm2_illegal_access` actually runs the real production
   kernel `flydsl_mxfp4_gemm2(atomic=True)` at the fault shape and reproduces the
   HIP illegal memory access itself (not just the invariant). It sizes the A
   buffer from the real sort output EXACTLY as `_mxfp4_a4w4_stage1` does, but
   places it against an unmapped VMM guard page so the tiny over-read
   deterministically crosses into unmapped VA and faults (the
   compute-sanitizer-style technique). Because a HIP fault is unrecoverable and
   poisons the process, the kernel runs in a SUBPROCESS and the test asserts the
   subprocess exits cleanly:
     * PRE-FIX  -> subprocess aborts (SIGABRT/nonzero) with "Memory access fault
       by GPU" -> test FAILS, having reproduced the real illegal access.
     * POST-FIX -> max_sorted is block-aligned, descriptor fits, subprocess
       prints CHILD_CLEAN and exits 0 -> test PASSES.

Run:
    pytest -q test_mxfp4_a4w4_moe_oob.py
    # or just the real-crash repro:
    pytest -q test_mxfp4_a4w4_moe_oob.py -k illegal_access -s
"""
import ctypes
import glob
import math
import os
import subprocess
import sys

import pytest
import torch

from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting


def _build_topk(M, num_experts, topk, device, seed=0):
    """A valid (topk_ids, topk_weights) pair; sort only needs legal expert ids."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Distinct experts per row (like real routing) so the sort does real work.
    topk_ids = torch.stack(
        [torch.randperm(num_experts, generator=g)[:topk] for _ in range(M)]
    ).to(device=device, dtype=torch.int32)
    topk_weights = torch.rand(M, topk, generator=g).to(
        device=device, dtype=torch.float32
    )
    return topk_ids, topk_weights


def _over_read_rows(n_rows, block_size):
    return ((n_rows + block_size - 1) // block_size) * block_size - n_rows


# (M, num_experts, topk) chosen so (M*topk - topk) % block_size != 0 for BM in
# {32,64,128}. The tiny case keeps the test fast; the K2.7 case is the real
# fault geometry (M=1172 -> max_sorted=22859, 21 rows over at BM=32).
_CONFIGS = [
    pytest.param(96, 8, 3, 2048, id="tiny"),          # 96*3-3 = 285
    pytest.param(717, 385, 9, 7168, id="k2.7-fault-M717"),
    pytest.param(1172, 385, 9, 7168, id="k2.7-fault-M1172"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("M,num_experts,topk,model_dim", _CONFIGS)
def test_moe_sorting_opus_row_extent_block_aligned(
    M, num_experts, topk, model_dim, block_size
):
    """The sorted row extent handed to the gemm kernels MUST be a block multiple.

    gemm2's non-persistent atomic path over-reads `inter_sorted_quant` by
    exactly `ceil(max_sorted/BM)*BM - max_sorted` rows otherwise.
    """
    device = "cuda"
    topk_ids, topk_weights = _build_topk(M, num_experts, topk, device)

    ret = moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype=torch.bfloat16,
        block_size=block_size,
        output_aux=AUX_SORT_OPUS,
        accumulate=True,
    )
    sorted_token_ids = ret[0]
    n_rows = int(sorted_token_ids.shape[0])  # == max_sorted -> inter_* row count

    over = _over_read_rows(n_rows, block_size)
    max_m_blocks = (n_rows + block_size - 1) // block_size
    assert over == 0, (
        f"BLOCK-ALIGNMENT INVARIANT BROKEN (M={M}, NE={num_experts}, topk={topk}, "
        f"block_size={block_size}): sorted row extent max_sorted={n_rows} is not a "
        f"multiple of block_size. gemm2 sizes its A descriptor to "
        f"max_m_blocks*block_size = {max_m_blocks * block_size} rows but "
        f"inter_sorted_quant has only {n_rows} rows -> non-persistent atomic path "
        f"over-reads {over} rows ({over * (model_dim // 2)} bytes at "
        f"inter_cols=model_dim//2) past the buffer -> HIP illegal memory access."
    )


def test_mxfp4_gemm2_descriptor_fits_allocation():
    """Pure-arithmetic form of the invariant for the K2.7 fault shape (no GPU).

    Mirrors what gemm2 computes: max_m_blocks = ceil(max_sorted/BM); the A
    descriptor spans max_m_blocks*BM rows while inter_sorted_quant has max_sorted
    rows. Reproduces the Opus sizing that PR #4526 routed the A4W4 path to.
    """
    M, num_experts, topk, block_size = 1172, 385, 9, 32
    # Opus path sizing (aiter/fused_moe.py::_moe_sorting_impl, pre-fix):
    max_sorted = M * topk + num_experts * block_size - topk
    assert max_sorted == 22859
    max_m_blocks = math.ceil(max_sorted / block_size)
    descriptor_rows = max_m_blocks * block_size  # what aq_rsrc spans
    over = descriptor_rows - max_sorted
    assert over == 21, f"expected 21 over-read rows for M=1172, got {over}"
    # The invariant the fix restores: descriptor must fit the allocation.
    aligned = max_m_blocks * block_size  # fixed max_num_tokens_padded
    assert descriptor_rows <= aligned, (
        f"gemm2 A descriptor spans {descriptor_rows} rows but inter_sorted_quant "
        f"has only {max_sorted}; fix pads max_num_tokens_padded to {aligned}."
    )


# ---------------------------------------------------------------------------
# Real illegal-access repro: run the production kernel in a subprocess.
# ---------------------------------------------------------------------------
# Fault shape for the subprocess (K2.7 geometry; D_HIDDEN shrunk to 512 to keep
# the single kernel launch fast -- the OOB is in the A/K dimension, independent
# of the N/output dimension D_HIDDEN).
_CRASH_ARGS = dict(M=1172, NE=385, topk=9, BM=32, D_HIDDEN=512, D_INTER=512)
_CHILD_FLAG = "--oob-child"


def _have_gpu_and_vmm():
    if not torch.cuda.is_available():
        return False
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    libs = sorted(glob.glob("/opt/rocm*/lib/libamdhip64.so"))
    if not libs:
        return False
    lib = ctypes.CDLL(libs[0])
    return all(
        hasattr(lib, s)
        for s in ("hipMemAddressReserve", "hipMemCreate", "hipMemMap", "hipMemSetAccess")
    )


@pytest.mark.skipif(
    not _have_gpu_and_vmm(), reason="needs a ROCm GPU + cupy + HIP VMM for the guard page"
)
def test_mxfp4_a4w4_gemm2_illegal_access():
    """Reproduce the ACTUAL HIP illegal memory access, deterministically.

    Runs `flydsl_mxfp4_gemm2(atomic=True)` (the exact production entry) at the
    K2.7 fault shape, with the A buffer placed against an unmapped guard page so
    the over-read is guaranteed to fault when it happens. The kernel runs in a
    subprocess (a HIP fault is unrecoverable); we assert it exits cleanly.
    """
    env = dict(os.environ)
    env["OOB_GUARD"] = "1"
    env["AMD_SERIALIZE_KERNEL"] = "1"
    env["HIP_LAUNCH_BLOCKING"] = "1"
    a = _CRASH_ARGS
    cmd = [
        sys.executable, os.path.abspath(__file__), _CHILD_FLAG,
        str(a["M"]), str(a["NE"]), str(a["topk"]),
        str(a["BM"]), str(a["D_HIDDEN"]), str(a["D_INTER"]),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    log = proc.stdout + proc.stderr
    faulted = (
        proc.returncode != 0
        or "Memory access fault" in log
        or "illegal memory access" in log
    )
    assert not faulted, (
        "MXFP4 A4W4 gemm2 raised a HIP illegal memory access at the K2.7 fault "
        f"shape {a} (subprocess rc={proc.returncode}). The Opus sort returned an "
        "unaligned max_sorted, so gemm2's oversized A buffer-descriptor over-read "
        "the intermediate buffer into the guard page. Apply the block-alignment "
        "fix (pad max_num_tokens_padded to a block multiple in _moe_sorting_impl / "
        "_flydsl_moe_sorting).\n--- subprocess output (tail) ---\n"
        + "\n".join(log.splitlines()[-15:])
    )


# ---------------------------------------------------------------------------
# Subprocess child (guard-page allocator + real kernel launch). Not collected by
# pytest; entered only via `python test_mxfp4_a4w4_moe_oob.py --oob-child ...`.
# ---------------------------------------------------------------------------
def _alloc_guarded(nbytes, device=0):
    """Device buffer whose LAST byte sits on a VMM mapped/unmapped boundary.

    Reserves `mapped + granule` of VA, maps physical memory to only the first
    `mapped` bytes, and returns a tensor at offset `mapped - nbytes` so the tensor
    END coincides with the mapped end -- the next byte is an unmapped guard. Any
    read past the buffer crosses into it and faults, no matter how small.
    """
    import cupy

    hip = ctypes.CDLL(sorted(glob.glob("/opt/rocm*/lib/libamdhip64.so"))[0])
    PINNED, DEVICE, RW, GRAN_MIN = 0x1, 1, 3, 0

    class _Loc(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

    class _Flags(ctypes.Structure):
        _fields_ = [
            ("compressionType", ctypes.c_ubyte),
            ("gpuDirectRDMACapable", ctypes.c_ubyte),
            ("usage", ctypes.c_ushort),
        ]

    class _Prop(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("requestedHandleType", ctypes.c_int),
            ("location", _Loc),
            ("win32HandleMetaData", ctypes.c_void_p),
            ("allocFlags", _Flags),
        ]

    class _Acc(ctypes.Structure):
        _fields_ = [("location", _Loc), ("flags", ctypes.c_int)]

    def ck(err, what):
        if err != 0:
            raise RuntimeError(f"{what} failed: hipError={err}")

    prop = _Prop()
    prop.type = PINNED
    prop.location.type = DEVICE
    prop.location.id = device

    gran = ctypes.c_size_t(0)
    ck(hip.hipMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), GRAN_MIN),
       "hipMemGetAllocationGranularity")
    g = int(gran.value)
    mapped = ((nbytes + g - 1) // g) * g

    ptr = ctypes.c_void_p(0)
    ck(hip.hipMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(mapped + g),
                                ctypes.c_size_t(0), ctypes.c_void_p(0), ctypes.c_ulonglong(0)),
       "hipMemAddressReserve")
    handle = ctypes.c_void_p(0)
    ck(hip.hipMemCreate(ctypes.byref(handle), ctypes.c_size_t(mapped), ctypes.byref(prop),
                        ctypes.c_ulonglong(0)), "hipMemCreate")
    ck(hip.hipMemMap(ptr, ctypes.c_size_t(mapped), ctypes.c_size_t(0), handle,
                     ctypes.c_ulonglong(0)), "hipMemMap")
    acc = _Acc()
    acc.location.type = DEVICE
    acc.location.id = device
    acc.flags = RW
    ck(hip.hipMemSetAccess(ptr, ctypes.c_size_t(mapped), ctypes.byref(acc), ctypes.c_size_t(1)),
       "hipMemSetAccess")

    base = int(ptr.value)
    tensor_ptr = base + (mapped - nbytes)  # tensor END == base + mapped == guard edge
    keep = (ptr, handle)  # keep the mapping alive for process lifetime
    _alloc_guarded._keep = getattr(_alloc_guarded, "_keep", [])
    _alloc_guarded._keep.append(keep)

    mem = cupy.cuda.UnownedMemory(tensor_ptr, nbytes, owner=keep)
    arr = cupy.ndarray((nbytes,), dtype=cupy.uint8, memptr=cupy.cuda.MemoryPointer(mem, 0))
    t = torch.from_dlpack(arr)
    t.zero_()
    return t


def _child_main(argv):
    from aiter import dtypes
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2
    from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4
    from aiter.ops.quant import per_1x32_f4_quant

    M, NE, topk, BM, D_HIDDEN, D_INTER = (int(x) for x in argv[:6])
    dev = "cuda"
    torch.manual_seed(0)

    topk_ids, topk_weights = _build_topk(M, NE, topk, dev)
    ret = moe_sorting(
        topk_ids, topk_weights, NE, D_HIDDEN,
        moebuf_dtype=torch.bfloat16, block_size=BM,
        output_aux=AUX_SORT_OPUS, accumulate=True,
    )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = ret[:4]
    max_sorted = int(sorted_ids.shape[0])
    over = _over_read_rows(max_sorted, BM)
    sys.stderr.write(
        f"CHILD max_sorted={max_sorted} BM={BM} over_read_rows={over} "
        f"aligned={over == 0}\n"
    )
    sys.stderr.flush()

    # A input sized EXACTLY as _mxfp4_a4w4_stage1 allocates it, but against a
    # guard page so the gemm2 over-read faults deterministically when present.
    nbytes = max_sorted * (D_INTER // 2)
    if os.environ.get("OOB_GUARD", "0") == "1":
        inter_sorted_quant = _alloc_guarded(nbytes, device=0).view(max_sorted, D_INTER // 2)
    else:
        inter_sorted_quant = torch.empty(
            (max_sorted, D_INTER // 2), device=dev, dtype=torch.uint8
        )
    inter_scale_cols = D_INTER // 32
    inter_scale_rows = ((max_sorted + 31) // 32) * 32
    inter_sorted_shuffled_scale = torch.empty(
        (inter_scale_rows, inter_scale_cols), device=dev, dtype=torch.uint8
    )

    w2 = torch.randn((NE, D_HIDDEN, D_INTER), dtype=torch.bfloat16, device=dev) / 4
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_q = w2_q.view(NE, D_HIDDEN, D_INTER // 2)
    w2_shuffled = shuffle_weight_a16w4(w2_q, 16, False)
    w2_scale_shuffled = shuffle_scale_a16w4(w2_scale, NE, False).view(torch.uint8)
    out = torch.zeros((M, D_HIDDEN), dtype=torch.bfloat16, device=dev)

    # Production entry (aiter/fused_moe.py::_mxfp4_a4w4_stage2 calls this exact
    # function). It hands the compiled kernel max_m_blocks = ceil(max_sorted/BM)
    # and the raw A data_ptr(); the kernel sizes aq_rsrc num_records to
    # max_m_blocks*BM*(D_INTER//2) -> over-reads when max_sorted % BM != 0.
    flydsl_mxfp4_gemm2(
        inter_sorted_quant=inter_sorted_quant,
        inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        w2_u8=w2_shuffled,
        w2_scale_u8=w2_scale_shuffled,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        sorted_token_ids=sorted_ids,
        sorted_weights=sorted_weights,
        flat_out=out,
        M_logical=M,
        max_sorted=max_sorted,
        BM=BM,
        use_nt=False,
        atomic=True,
        mxfp4out=False,
        NE=NE,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        topk=topk,
        BN=256,
        BK=256,
    )
    torch.cuda.synchronize()
    print("CHILD_CLEAN")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == _CHILD_FLAG:
        sys.exit(_child_main(sys.argv[2:]))
    sys.exit(pytest.main([__file__, "-v"]))
