#!/usr/bin/env python3
"""Generate the aiter Gluon kernel body from the Artemis gdn_decode_m16 source.

Done as a script rather than by hand: the source is ~600 lines of dense Gluon
with explicit register layouts, and a transcription slip would surface as a
numerical mismatch that is expensive to localize. Every edit below is targeted
and asserted, so a change in the upstream source fails loudly instead of
silently emitting something subtly different.

Provenance is recorded in the emitted header (source file + sha256), per the
"pin the exact source revision" requirement in sikl-port-generated-kernels.

Edits applied to the verbatim source:
  1. pad_slot_id guard after each `slot = gl.load(Indices + token)`
  2. HAS_FP8 constexpr threaded through _finish_pair so the quant epilogue is
     optional (an unquantized out_proj cannot consume an (fp8, scales) tuple)
  3. PAD_SLOT_ID / HAS_FP8 added to both kernel entry signatures
"""

import hashlib
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "artemis_kernels/gdn_decode_m16.py")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "fused_gdn_decode_qkvz_kernel.py")

text = SRC.read_text()
sha = hashlib.sha256(text.encode()).hexdigest()
lines = text.splitlines()

# Gluon function bodies only: from the first @gluon.jit to just before the host fn.
start = next(i for i, line in enumerate(lines) if line.startswith("@gluon.jit"))
end = next(i for i, line in enumerate(lines) if line.startswith("def gdn_fused_decode_group_fp8_quant"))
body = "\n".join(lines[start:end]).rstrip() + "\n"


def sub(old, new, count):
    """Replace and assert the expected number of occurrences."""
    global body
    found = body.count(old)
    assert found == count, f"expected {count} occurrences of {old!r}, found {found}"
    body = body.replace(old, new)


# --- 1. pad-slot guard -------------------------------------------------------
# One program owns one (token, k_head), so an early return skips the state load,
# the convolution, the recurrence, the history rollover and every store. This is
# the convention SGLang's replay_state_indices_validator documents and the one
# aiter's fused_conv_recurrent_norm already implements (`if state_idx < 0: return`).
sub(
    """    slot = gl.load(Indices + token)
    if INDEX64:
        slot = slot.to(gl.int64)
""",
    """    slot = gl.load(Indices + token)
    if slot == PAD_SLOT_ID:
        # Padded CUDA-graph row: read no state and write none. Outputs for this
        # row are left undefined, matching aiter's other indexed state kernels.
        return
    if INDEX64:
        slot = slot.to(gl.int64)
""",
    2,
)

# --- 2. optional fp8 epilogue ------------------------------------------------
sub(
    """                 EARLY_SIGMOID: gl.constexpr, LATE_STORES: gl.constexpr,
                 FAST_QUANT: gl.constexpr,
                 DPP_RMS: gl.constexpr):""",
    """                 EARLY_SIGMOID: gl.constexpr, LATE_STORES: gl.constexpr,
                 FAST_QUANT: gl.constexpr,
                 DPP_RMS: gl.constexpr,
                 HAS_FP8: gl.constexpr):""",
    1,
)

# Indent the whole quant epilogue under `if HAS_FP8:` and give the bf16-only
# path its own LATE_STORES handling for the normalized output.
quant_start = "    # Quantize the externally visible BF16 output, preserving its rounding point."
idx = body.index(quant_start)
head, tail = body[:idx], body[idx:]
tail_end = tail.index("\n\n\n@gluon.jit")
epilogue, rest = tail[:tail_end], tail[tail_end:]
# Guard with an early return and leave the epilogue at its original indentation.
# Do NOT also indent it: doing both puts the whole block inside the `if` after
# the `return`, which is valid Python, compiles cleanly, and silently never
# executes -- the quantized/scales buffers then keep their torch.empty garbage.
body = (
    head
    + "    if not HAS_FP8:\n"
    + "        # bf16-only caller: skip the group quantization entirely.\n"
    + "        if LATE_STORES:\n"
    + "            gl.store(Output + offset, normalized)\n"
    + "        return\n\n"
    + epilogue
    + rest
)
# The epilogue must stay reachable at function-body indentation.
assert "\n    values = normalized.to(gl.float32)\n" in body, "quant epilogue lost its indentation"
assert "\n        values = normalized.to(gl.float32)\n" not in body, "quant epilogue is dead code"

# --- 3. thread the new constexprs through call sites and entry points --------
sub(
    """        cached_z, cached_weight, cached_sigmoid, EARLY_SIGMOID, True,
        FAST_QUANT, DPP_RMS,
    )""",
    """        cached_z, cached_weight, cached_sigmoid, EARLY_SIGMOID, True,
        FAST_QUANT, DPP_RMS, HAS_FP8,
    )""",
    1,
)
sub(
    """        cached_z, cached_weight, cached_sigmoid, False, not B32,
        B32, False,
    )""",
    """        cached_z, cached_weight, cached_sigmoid, False, not B32,
        B32, False, HAS_FP8,
    )""",
    1,
)
sub(
    """                  BATCH: gl.constexpr, INDEX64: gl.constexpr):""",
    """                  BATCH: gl.constexpr, INDEX64: gl.constexpr,
                  PAD_SLOT_ID: gl.constexpr, HAS_FP8: gl.constexpr):""",
    1,
)
sub(
    """                        INDEX64: gl.constexpr, B32: gl.constexpr):""",
    """                        INDEX64: gl.constexpr, B32: gl.constexpr,
                        PAD_SLOT_ID: gl.constexpr, HAS_FP8: gl.constexpr):""",
    1,
)

header = f'''# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon kernels for the fused Qwen3-Next GDN decode step (gfx950/CDNA4).

One eight-wave workgroup owns a token/key-head group's convolution history and
both FP32 value-head states, fusing: the packed qkvz/ba split, a depthwise
causal conv1d with bias and SiLU, the delta-rule gating, the FP32 recurrent
state update, a gated RMSNorm with a SiLU gate, and an optional per-head
group-128 FP8 quantization epilogue.

Partial-column prefetch overlaps state reads with QKV preparation. Pool-size
bounds select safe 32-bit or full-width addressing. Wave-local quantization
preserves BF16 rounding and maximum-number semantics.

Rows whose state index equals ``PAD_SLOT_ID`` are skipped without reading or
writing state; their outputs are undefined.

Generated from Artemis kernel pack source, then modified to add the pad-slot
guard and the optional FP8 epilogue. Do not hand-edit; see build_aiter_kernel.py.

  source: {SRC.name}
  sha256: {sha}
"""

import torch  # noqa: F401  (kept for parity with the aiter op module layout)
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


'''

DST.write_text(header + body)
print(f"wrote {DST} ({len((header + body).splitlines())} lines) from {SRC.name} sha256={sha[:16]}")
