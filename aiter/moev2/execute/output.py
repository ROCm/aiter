# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The caller-provided `output=` buffer: validation and the final copy.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
This module moved here from aiter/fused_moe.py and is shared with the legacy MoE path:
aiter.fused_moe re-exports it and legacy MoE calls run it. Changes must keep legacy
behavior (`python tests/moe_dispatch/snapshot.py --check`) and should loop in the moev2
owners (@qli88). See aiter/moev2/README.md.
"""

from aiter.presorted_moe import bytes_overlap


# `output`: caller-provided [M, model_dim] destination.
#
#     path                                      in-place?  why?
#     opus / ck / flydsl sort + accumulate      yes        sort kernel zeroes what it is handed
#     ... + reduce mode (non-EP)                yes        buffer is born in fused_moe_2stages
#     FLAT 1stage (tuned flat=1/2)              no         kernel needs 64 spare bytes past the rows
#     adaptive-aux sort (a4w4, atomic)          yes        if 4-byte aligned (BM16 zeroes in 4B)
#     caller presort (register_presorted_moe)   yes        zeroed here unless caller did
#     grouped a4w4/a8w4 (gfx1250)               no         callee has no out param
#
# Not in-place => _return_output copies, so the contract holds either way;
# only the saving is lost.
def _validate_output_buffer_metadata(output, shape, dtype, device):
    if output is None:
        return
    # The buffer has to stand in for the tensor fused_moe would have allocated.
    if (
        output.shape != shape
        or output.dtype != dtype
        or output.device != device
        or not output.is_contiguous()
    ):
        raise RuntimeError(
            f"output mismatch: got shape={tuple(output.shape)} dtype={output.dtype} "
            f"device={output.device} contiguous={output.is_contiguous()}, expected a "
            f"contiguous {tuple(shape)} {dtype} tensor on {device}"
        )


def _validate_output_buffer_no_overlap(output, hidden_states):
    if output is None:
        return
    # Accumulate mode zeroes the output rows before stage1 reads hidden_states.
    if bytes_overlap(output, hidden_states):
        raise RuntimeError(
            "output overlaps hidden_states, which moe_sorting zeroes before stage1 "
            "reads it"
        )


def _return_output(ret, output):
    # Paths that already wrote output return it directly; others copy here.
    if output is None or ret is output:
        return ret
    return output.copy_(ret)
