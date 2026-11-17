# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Packed Attention A2A profiles: what ``AttentionA2AIntraNodeOp(return_packed=True)``
sends for each dense MHA-v4 backend, and how that backend reads it.

Frameworks pick a profile per attention backend and construct the op from it; producers of
the same packed buffers (the pre-attention monokernel) lay them out from it.
"""

from dataclasses import dataclass

from aiter.ops.mha_v4 import AttentionPack

_SCALE_KINDS = {
    "f32_per_tensor": "tensor",
    "f32_per_channel": "channel",
    "e8m0_per_1x32": "block",
}


@dataclass(frozen=True)
class AttentionA2AProfile:
    name: str
    # Transport codecs of Q, K and V, and the codecs the MHA reads them as.
    codecs: tuple
    consumer_codecs: tuple
    # Per role: "f32_per_tensor", "f32_per_channel" or "e8m0_per_1x32".
    scale_modes: tuple
    attention_backend: str
    v_pack: AttentionPack = AttentionPack.DEFAULT
    # Whether Q and K take the Hadamard before quantization by default.
    hadamard: bool = True
    block_num: int = 512
    # Whether K and V may be trimmed to the valid sequence (valid_kv_len).
    direct_kv_output: bool = False
    # The local sequence must be a multiple of this.
    pad_multiple: int = 32

    @property
    def quant(self):
        """The op's ``quant`` argument: the Q/K codec and the V codec."""
        return (self.codecs[0], self.codecs[2])

    @property
    def scale_kinds(self):
        """Per role: "tensor", "channel" or "block" (E8M0 per 32 values)."""
        return tuple(_SCALE_KINDS[mode] for mode in self.scale_modes)


_TENSOR, _CHANNEL, _BLOCK = "f32_per_tensor", "f32_per_channel", "e8m0_per_1x32"
_FP6_P = AttentionPack.V_FOR_FP6_P

ATTENTION_A2A_PROFILES = {
    profile.name: profile
    for profile in (
        AttentionA2AProfile(
            "e4m3-e4m3", ("e4m3",) * 3, ("e4m3",) * 3, (_TENSOR,) * 3, "aiter_fp8"
        ),
        AttentionA2AProfile(
            "int8-e4m3",
            ("int8", "int8", "e4m3"),
            ("int8", "int8", "e4m3"),
            (_TENSOR,) * 3,
            "aiter_i8fp8",
            hadamard=False,
        ),
        AttentionA2AProfile(
            "mxfp8-e4m3",
            ("mxfp8", "mxfp8", "e4m3"),
            ("e4m3",) * 3,
            (_BLOCK, _BLOCK, _TENSOR),
            "aiter_mxfp8",
        ),
        AttentionA2AProfile(
            "e4m3-mxfp6",
            ("e4m3", "e4m3", "mxfp6_p"),
            ("e4m3", "e4m3", "mxfp6"),
            (_TENSOR, _TENSOR, _BLOCK),
            "aiter_f8f6",
            v_pack=_FP6_P,
        ),
        AttentionA2AProfile(
            "mxfp6-e4m3",
            ("mxfp6", "mxfp6", "e4m3_pc"),
            ("mxfp6", "mxfp6", "e4m3"),
            (_BLOCK, _BLOCK, _CHANNEL),
            "aiter_mxfp6",
            block_num=256,
        ),
        AttentionA2AProfile(
            "mxfp6-mxfp6",
            ("mxfp6", "mxfp6", "mxfp6_p"),
            ("mxfp6",) * 3,
            (_BLOCK,) * 3,
            "aiter_f6f6",
            v_pack=_FP6_P,
            block_num=256,
            direct_kv_output=True,
        ),
        AttentionA2AProfile(
            "mxfp4-mxfp4",
            ("mxfp4",) * 3,
            ("mxfp4",) * 3,
            (_BLOCK,) * 3,
            "aiter_mxfp4",
            v_pack=_FP6_P,
            direct_kv_output=True,
        ),
        AttentionA2AProfile(
            "mxfp6-mxfp4",
            ("mxfp6", "mxfp6", "mxfp4"),
            ("mxfp6", "mxfp6", "mxfp4"),
            (_BLOCK,) * 3,
            "aiter_f6f4",
            v_pack=_FP6_P,
            block_num=256,
            direct_kv_output=True,
        ),
    )
}


def attention_a2a_profile(name):
    """The profile called ``name``, such as ``"e4m3-e4m3"``."""
    try:
        return ATTENTION_A2A_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown Attention A2A profile {name!r}; "
            f"expected one of {', '.join(ATTENTION_A2A_PROFILES)}"
        ) from None
