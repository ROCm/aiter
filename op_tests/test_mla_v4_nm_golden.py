# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical correctness test for mla_decode_fwd_v4_nm.

Loads byte-exact device buffers from poc_kl's `dump_result=1` artifacts plus
extra `devbins/*.bin` produced by an instrumented poc_kl run, copies them
straight into PyTorch CUDA tensors backing the aiter dispatcher's inputs,
and asserts byte equality against poc_kl's GPU output (gpu_SPLIT_DATA.hex).

Required setup (one-time, host side, inside the gfx950 container):
  cd /dockerx/poc_kl_dumps/poc_kl_tree/mi350/mla_asm     # see plan §3.x
  # 1. Build poc_kl mla.exe with the devbin-dump patch in
  #    mla_execute_v4_hip.inl (see commit history).
  /opt/rocm/bin/hipcc --offload-arch=gfx950 -g mla.cpp -o mla.exe
  # 2. Run with dump_result=1 (writes the .hex files including the golden
  #    gpu_SPLIT_DATA.hex) and POC_KL_V4_DUMP_DEVBINS=1 (writes the .bin
  #    files used to feed aiter byte-exact).
  POC_KL_V4_DUMP_DEVBINS=1 POC_KL_V4_BIN_OUTDIR=/dockerx/poc_kl_dumps/devbins \
    ./mla.exe model_version=4 sub_Q=64 batch=2 num_kv_heads=1 \
              gqa_ratio=16 kv_seq_lens=64 dim=512 block_size=1 \
              wv_tg=4 atm_f32=0 seed=0 data_type=2 passes=1 \
              pass_size=16 out_16_nosplit=0 q_seq_lens=4 mask=0 \
              pattern_q=0 dump_result=1
  cp gpu_SPLIT_DATA.hex /dockerx/poc_kl_dumps/

Then point the test at the dump dir (must be visible inside the test runner):
  POC_KL_DUMP_DIR=/dockerx/poc_kl_dumps \
      pytest -xvs op_tests/test_mla_v4_nm_golden.py

If POC_KL_DUMP_DIR is unset or files are missing, the test SKIPs (CI-safe).

Why we load .bin (raw device dumps) instead of reconstructing inputs from
the human-readable .hex files:
  The kernel reads 576 bytes per token from Q/KV. Only the first 512 bytes
  per token are written by poc_kl `init_host_buffers` (NOPE 448 + e8m0 scale
  16 + zero pad 48). The trailing 64 bytes of each token come from a
  poc_kl-internal `hipMemcpy` over-copy (buf_size_Q = sz_Q*sizeof(cl_half))
  that pulls in heap-adjacent buffer bytes. These bytes are non-trivial to
  reconstruct from the .hex dumps alone. Worse, the kernel was observed to
  produce 100% NaN output when given *uniform* e8m0 scale bytes (e.g. all
  0x7f = e8m0=1.0) at offset [448:464] of every token — a degenerate code
  path that doesn't show up with poc_kl's varying init data. To sidestep
  both issues, we just copy the raw device bytes byte-for-byte and let
  poc_kl produce a matching golden, getting byte-exact equality.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import aiter
import aiter.mla  # main no longer auto-imports submodules; need explicit
from aiter.jit.utils.chip_info import get_gfx


# ---------------------------------------------------------------------------
# Test config — must match the poc_kl `mla.exe` cmdline above.
# ---------------------------------------------------------------------------
BATCH               = 2
KV_SEQ_LENS         = 64                    # per-seq KV tokens
Q_SEQ_LOGICAL       = 4                     # per-seq Q tokens (q_seq_lens cli)
GQA_RATIO           = 16
NUM_KV_HEADS        = 1
NUM_HEADS           = GQA_RATIO * NUM_KV_HEADS   # 16
PAGE_SIZE           = 1
SUB_Q               = 64
NUM_KV_SPLITS       = 1                     # passes=1
DIM_NOPE_BYTES      = 448
DIM_ROPE_ELEM       = 64                    # BF16 elements (= 128 bytes)
DIM_QK_PACKED       = 576                   # = args.dim(512) + args.k_rotary(64)
V_HEAD_DIM          = 512                   # kv_lora_rank


def _parse_4byte_hex(path: str) -> np.ndarray:
    """Parse `0x%08x ` 4-byte (FP32/INT32) dump → flat uint32 LE bytes."""
    pat = re.compile(rb"\b0x([0-9a-fA-F]{8})\b")
    with open(path, "rb") as f:
        data = f.read()
    vals = [int(m.group(1), 16) for m in pat.finditer(data)]
    return np.array(vals, dtype=np.uint32)


def _on_gfx950():
    try:
        return get_gfx() == "gfx950"
    except Exception:
        return False


DUMP_DIR = os.environ.get(
    "POC_KL_DUMP_DIR",
    "/dockerx/poc_kl_dumps",
)
DEVBIN_DIR = Path(DUMP_DIR) / "devbins"
GOLDEN_PATH = Path(DUMP_DIR) / "gpu_SPLIT_DATA.hex"

REQUIRED_BINS = [
    "d_q.bin", "d_kv.bin", "d_qrope.bin", "d_kvrope.bin",
    "d_ltp.bin", "d_ltd.bin", "d_ltl.bin", "d_qtp.bin", "d_stp.bin",
]
_missing = [f for f in REQUIRED_BINS if not (DEVBIN_DIR / f).exists()]
if not GOLDEN_PATH.exists():
    _missing.append(str(GOLDEN_PATH))

needs_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available() or not _on_gfx950(),
    reason="v4 nm shader needs gfx950 GPU",
)
needs_dumps = pytest.mark.skipif(
    bool(_missing),
    reason=(
        f"poc_kl dump/devbin files missing in {DUMP_DIR} ({_missing}); "
        "see file docstring for how to regenerate"
    ),
)


# ---------------------------------------------------------------------------
# Helper: copy raw bytes from a host file into a CUDA tensor's backing memory.
#
# libamdhip64.so is *lazy-loaded* inside _ensure_hip() so that CI machines
# without ROCm don't fail at module-import time (pytest collection would
# otherwise hard-error before the needs_gfx950 / needs_dumps skipif decorators
# get a chance to run).
# ---------------------------------------------------------------------------
import ctypes

_HIP_H2D = 1
_hip = None


def _ensure_hip():
    """Late-bind libamdhip64.so. Only invoked from inside the test body, so
    the import-time path stays ROCm-free and CI machines without HIP get a
    clean SKIPPED instead of a collection error."""
    global _hip
    if _hip is None:
        _hip = ctypes.CDLL("libamdhip64.so")
        _hip.hipMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _hip.hipMemcpy.restype = ctypes.c_int
    return _hip


def _fill_tensor_from_bin(tensor: torch.Tensor, bin_path: Path, max_bytes: int = None):
    """Byte-for-byte H2D memcpy from `bin_path` into `tensor.data_ptr()`."""
    hip = _ensure_hip()
    nbytes_tensor = tensor.numel() * tensor.element_size()
    with open(bin_path, "rb") as f:
        host_bytes = f.read()
    n = min(nbytes_tensor, len(host_bytes), max_bytes) if max_bytes else \
        min(nbytes_tensor, len(host_bytes))
    buf = (ctypes.c_uint8 * n).from_buffer_copy(host_bytes[:n])
    err = hip.hipMemcpy(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.cast(buf, ctypes.c_void_p),
        n, _HIP_H2D,
    )
    if err != 0:
        raise RuntimeError(f"hipMemcpy ({bin_path.name}) failed: hipError {err}")
    return n


# ---------------------------------------------------------------------------
# Loader: returns the `args` dict for mla_decode_fwd_v4_nm.
# ---------------------------------------------------------------------------
def _load_inputs_from_devbins() -> dict:
    """Allocate CUDA tensors of the shapes aiter expects, then memcpy
    poc_kl's exact device bytes into them.
    """
    total_q  = BATCH * Q_SEQ_LOGICAL
    num_page = BATCH * (KV_SEQ_LENS // PAGE_SIZE)
    device = "cuda"
    # aiter.dtypes.fp8 auto-resolves per arch (gfx942 = e4m3fnuz, gfx950 = e4m3fn).
    # The v4 nm kernel reads raw FP8 bytes so the variant label doesn't affect
    # numerics; using aiter's canonical alias is what passes the strict dtype
    # check in aiter/utility/dtypes.py::torch_to_aiter() on either arch.
    fp8_dt = aiter.dtypes.fp8

    # Allocate
    q          = torch.zeros((total_q, NUM_HEADS, DIM_QK_PACKED),
                              dtype=torch.uint8, device=device).view(fp8_dt)
    kv_buffer  = torch.zeros((num_page, PAGE_SIZE, NUM_KV_HEADS, DIM_QK_PACKED),
                              dtype=torch.uint8, device=device).view(fp8_dt)
    qrope      = torch.zeros((total_q, NUM_HEADS, DIM_ROPE_ELEM),
                              dtype=torch.bfloat16, device=device)
    kvrope     = torch.zeros((num_page, PAGE_SIZE, NUM_KV_HEADS, DIM_ROPE_ELEM),
                              dtype=torch.bfloat16, device=device)
    qo_indptr        = torch.zeros(BATCH + 1, dtype=torch.int32, device=device)
    kv_indptr        = torch.zeros(BATCH + 1, dtype=torch.int32, device=device)
    kv_page_indices  = torch.zeros(num_page, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.zeros(BATCH, dtype=torch.int32, device=device)
    split_indptr     = torch.zeros(BATCH + 1, dtype=torch.int32, device=device)
    output           = torch.zeros((total_q, NUM_HEADS, V_HEAD_DIM),
                                   dtype=torch.bfloat16, device=device)

    # Fill from poc_kl's device-side dumps. We use the FIRST nbytes_tensor
    # bytes of each .bin (poc_kl over-allocates some of them).
    _fill_tensor_from_bin(q,                 DEVBIN_DIR / "d_q.bin")
    _fill_tensor_from_bin(kv_buffer,         DEVBIN_DIR / "d_kv.bin")
    _fill_tensor_from_bin(qrope,             DEVBIN_DIR / "d_qrope.bin")
    _fill_tensor_from_bin(kvrope,            DEVBIN_DIR / "d_kvrope.bin")
    _fill_tensor_from_bin(qo_indptr,         DEVBIN_DIR / "d_qtp.bin")
    _fill_tensor_from_bin(kv_indptr,         DEVBIN_DIR / "d_ltp.bin")
    _fill_tensor_from_bin(kv_page_indices,   DEVBIN_DIR / "d_ltd.bin",
                          max_bytes=num_page * 4)
    _fill_tensor_from_bin(kv_last_page_lens, DEVBIN_DIR / "d_ltl.bin")
    _fill_tensor_from_bin(split_indptr,      DEVBIN_DIR / "d_stp.bin")
    torch.cuda.synchronize()

    return dict(
        q=q, qrope=qrope,
        kv_buffer=kv_buffer, kvrope=kvrope,
        output=output,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
        split_indptr=split_indptr,
        max_seqlen_q=Q_SEQ_LOGICAL,
        num_kv_splits=NUM_KV_SPLITS,
        sub_Q=SUB_Q,
        out_16_nosplit=0,
    )


# ---------------------------------------------------------------------------
# The actual golden test
# ---------------------------------------------------------------------------
@needs_gfx950
@needs_dumps
def test_v4_nm_byte_exact_against_poc_kl():
    """Byte-exact correctness: feed poc_kl's device-side input bytes →
    aiter logits == poc_kl gpu_SPLIT_DATA.hex bit-for-bit.

    Failure modes:
      - Output bytes differ → ABI mismatch or kernarg packing changed
        relative to poc_kl's mla_execute_v4_hip.inl::execute_v4_kernel.
      - Output is NaN → either input bytes mismatched (regenerate devbins)
        or the kernel hit a degenerate code path (e.g. uniform-scale).
    """
    args = _load_inputs_from_devbins()

    # SENTINEL pre-fill so we can verify the kernel actually wrote everything.
    SENTINEL = -7.7e30
    args["logits"] = torch.full(
        (BATCH, NUM_KV_SPLITS, NUM_KV_HEADS, GQA_RATIO * Q_SEQ_LOGICAL, V_HEAD_DIM),
        SENTINEL, dtype=torch.float32, device="cuda",
    )
    args["attn_lse"] = torch.full(
        (BATCH, NUM_KV_SPLITS, NUM_KV_HEADS, GQA_RATIO * Q_SEQ_LOGICAL, 1),
        SENTINEL, dtype=torch.float32, device="cuda",
    )

    logits, attn_lse = aiter.mla.mla_decode_fwd_v4_nm(**args)
    torch.cuda.synchronize()

    # Per-batch coverage check
    for b in range(BATCH):
        ut = (logits[b] == SENTINEL).float().mean().item()
        assert ut < 0.01, f"batch {b}: kernel skipped ({ut*100:.1f}% SENTINEL)"

    # Byte-exact compare against the golden gpu_SPLIT_DATA.hex
    golden_u32 = _parse_4byte_hex(str(GOLDEN_PATH))
    expected_n = BATCH * NUM_KV_SPLITS * NUM_KV_HEADS * (GQA_RATIO * Q_SEQ_LOGICAL) * V_HEAD_DIM
    assert golden_u32.size == expected_n, (
        f"gpu_SPLIT_DATA.hex has {golden_u32.size} u32 vals, want {expected_n}"
    )
    golden = torch.from_numpy(golden_u32).cuda()
    aiter_bits = logits.contiguous().view(torch.uint32).flatten()
    n_diff = (aiter_bits != golden).sum().item()
    n_total = aiter_bits.numel()
    assert n_diff == 0, (
        f"aiter output differs from poc_kl golden in {n_diff}/{n_total} elements "
        f"({100*n_diff/n_total:.2f}%); first 8 mismatches: "
        + ", ".join(
            f"[{i}] aiter=0x{aiter_bits[i].item():08x} golden=0x{golden[i].item():08x}"
            for i in (aiter_bits != golden).nonzero(as_tuple=True)[0][:8].tolist()
        )
    )


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
