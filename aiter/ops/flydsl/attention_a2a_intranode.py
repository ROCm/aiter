# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the FlyDSL single-tensor intranode push all-to-all."""

from __future__ import annotations

import math
import operator
import os
from typing import NamedTuple

import flydsl.expr as fx
import mori.shmem as ms
import torch
from flydsl.expr.typing import Stream
from mori.shmem import mori_shmem_create_tensor

from aiter.ops.mha_v4 import AttentionPack
from aiter.ops.mha_v4_quant import (
    MHA_V4_MXFP6_V_SCALE_TILE_BYTES,
    mxfp6_v_raw_buffer_size,
    mxfp6_v_tiles,
)

from .communication_ops_host_utils import build_p2p_table
from .kernels.attention_a2a_intranode_kernel import (
    _transport_bytes,
    make_attention_a2a_dequant_jit,
    make_attention_a2a_jit,
    make_attention_a2a_reuse_jit,
    needs_split_v_exchange,
    supports_kv_seq_len,
    v_pc_partial_elems,
)
from .kernels.tensor_shim import _run_compiled

_DEFAULT_BLOCK_NUM = 128
_DEFAULT_WARP_NUM_PER_BLOCK = 8
_MAX_RAW_BUFFER_BYTES = 0xFFFFFFFF


def _payload_size(
    role,
    codec,
    return_packed,
    quant,
    role_numel,
    mxfp4_k_size,
    fp6_k_payload_size,
    mxfp4_v_size,
    fp6_v_payload_size,
):
    if return_packed:
        if role == 1 and codec == "mxfp4":
            return mxfp4_k_size
        if role == 1 and codec == "mxfp6":
            return fp6_k_payload_size
        if role == 2 and codec == "mxfp4":
            return mxfp4_v_size
        if role == 2 and codec == "mxfp6_p":
            return fp6_v_payload_size
    return _transport_bytes(role_numel, codec) if quant else role_numel


def _scale_size(
    role,
    codec,
    per_tensor,
    return_packed,
    role_numel,
    heads_local,
    world_size,
    block_num,
    valid_kv_len,
    fp6_k_scale_size,
):
    if per_tensor:
        if codec == "e4m3_pc":
            return heads_local * 128 * (1 + world_size * world_size)
        return 1 + world_size * block_num * 4
    if return_packed and role == 2:
        return (
            heads_local * mxfp6_v_tiles(valid_kv_len) * MHA_V4_MXFP6_V_SCALE_TILE_BYTES
        )
    if return_packed and role == 1 and codec == "mxfp6":
        return fp6_k_scale_size
    return role_numel // 32


def _scale_padding_size(role, codec, role_sequence, valid_kv_len, heads_local):
    """Consumer-owned backing slack for speculative scale gathers (packed return)."""
    from aiter.ops.mha_v4_quant import (
        MHA_V4_KV_SCALE_LOOKAHEAD_ROWS,
        MHA_V4_KV_TILE_ROWS,
        MHA_V4_MXFP4_K_SCALE_SLACK_BYTES,
        MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,
    )

    # Q scales are unpadded since #6841; the gather clamps via buffer descriptor num_records.
    if role == 1 and codec == "mxfp4":
        rounded = (
            (valid_kv_len + MHA_V4_KV_TILE_ROWS - 1) // MHA_V4_KV_TILE_ROWS
        ) * MHA_V4_KV_TILE_ROWS
        return (
            rounded + MHA_V4_KV_SCALE_LOOKAHEAD_ROWS - valid_kv_len
        ) * heads_local * 4 + MHA_V4_MXFP4_K_SCALE_SLACK_BYTES
    if role == 2 and codec == "mxfp4":
        return MHA_V4_MXFP4_V_SCALE_SLACK_BYTES
    return 0


def _role_format(return_packed, role):
    return "qkv"[role] if return_packed else ""


def _require_raw_buffer_extent(role, extent):
    if extent > _MAX_RAW_BUFFER_BYTES:
        raise ValueError(f"{role} extent {extent} exceeds raw-buffer limit")


class PackedRoleResult(NamedTuple):
    """Packed payload and scale for one role from `submit_role`.

    These alias the op's double-buffered storage; they are not copies and are
    reused two Q/K/V trios later. Consumers must be ordered after the submit on
    its stream (same stream, or an event recorded after submit_role).
    """

    payload: torch.Tensor
    scale: torch.Tensor


class AttentionA2AIntraNodeOp:
    """Own symmetric receive and handshake buffers for one tensor shape.

    Submit Q/K/V as ordered per-role launch sequences.
    Set quant to one codec for uniform Q/K/V quantization or a (Q/K, V) codec pair.
    Native codecs use one E8M0 scale per 32 adjacent values (see the table below
    for packed return). Raw MXFP4 payloads
    contain two E2M1 values per byte, low nibble first. Raw MXFP6 payloads contain
    contiguous E2M3 six-bit codes, least-significant bits first.
    Quantized calls return locally dequantized bf16 Q/K/V by default.
    By default, submit_role returns None for Q/K and a tuple of three BF16 outputs
    for V; quantized outputs are locally dequantized. With return_packed=True, each
    submit_role returns PackedRoleResult(payload, scale) for that role in MHA V4
    layout. Results alias double-buffered storage and become ready when the
    submitted stream work completes.
    e4m3 is the device's native FP8 (E4M3FNUZ on gfx942, OCP E4M3FN on gfx950).
    Packed MX formats are gfx950-only. Packed outputs use role-specific scale layouts:
    MXFP4 V uses token-axis scales in gather order, and E4M3 V uses one float32
    descale per destination tensor (F32_PER_TENSOR). e4m3_pc is V-only and
    packed-only: its ordinary FP8 BSHD bytes use contiguous float32 [1, Hlocal, 128]
    descales (F32_PER_CHANNEL), for example with dense MHA V4 f6f8. Zero channels
    clamp amax to 1e-12, giving a positive descale and zero payload, matching
    quantize_v_fp8.
    hadamard=True applies normalized Walsh-Hadamard
    to Q/K before quantization;
    V is unchanged. The default is enabled for packed output and disabled otherwise.
    mxfp6_p is V-only and packed-only, with FP6-P ordering and token-axis scales.
    v_pack=AttentionPack.V_FOR_FP6_P selects FP6-P ordering for MXFP4 or mxfp6_p V
    and requires seq_local divisible by 32; DEFAULT retains the standard ordering.
    Odd multiples of 32 publish partial maxima before the V packing launch.
    Packed INT8 does not apply Hadamard rotation. MX Q folds softmax_scale * log2(e).
    K is quantized without mha_v4's gated K-mean centering (_k_mean). Results match
    mha_v4() only while that gate stays off; K with a large shared component can
    incur more quantization error here than in mha_v4().
    Scale format per codec and mode (native = return_packed=False, packed =
    return_packed=True):

    codec    native                       packed
    e4m3     E8M0 per 32 channel values   FP32 per tensor (Q/K/V)
    int8     E8M0 per 32 channel values   FP32 per tensor (Q/K only)
    mxfp8    same bytes as native e4m3    E8M0 per 32 channel values (Q/K only)
    mxfp4    E8M0 per 32 channel values   Q/K: E8M0 per 32 channels;
                                          V: E8M0 per channel per 32 tokens
    mxfp6    E8M0 per 32 channel values   E8M0 per 32 channel values (Q/K only)
    e4m3_pc  not supported                FP32 per channel (V only)
    mxfp6_p  not supported                E8M0 per channel per 32 tokens, FP6-P
                                          order (V only)

    On gfx942 packed return is limited to (int8, int8, e4m3) and (e4m3, e4m3, e4m3).
    AITER_A2A_FP6_FORCE_SOFTWARE=1 forces the software FP6 encoder on gfx950 (the
    encoder gfx942 uses); it is part of the JIT cache key.
    valid_kv_len defaults to the full padded sequence. With packed output and
    token-local K/V scales, it keeps Q untrimmed and returns K/V for valid_kv_len
    tokens in ceil(valid_kv_len / 128) tiles. Tokens at or past that length are
    ignored; FP6-P V repeats the last valid token through its final tile.
    Padding must be confined to the last rank's input slice.
    All ranks must use the same mode and serialize calls on one stream.
    Consumers must finish on that stream before the next call reusing their
    parity (two calls later); cross-stream consumers require an explicit join.

    group is a required torch.distributed ProcessGroup containing exactly the
    MoRI participants (size == world_size, group rank == rank). Construction
    verifies that all ranks agree on the configuration and on the offsets of
    every symmetric buffer, raising ValueError on every rank otherwise.
    A role-0 submit on a different stream than the previous trio raises; use a
    new op or join the streams.

    Debug: AITER_A2A_SPIN_TIMEOUT_ITERS=N (default 0, off) makes each device
    wait loop trap after N polls instead of hanging; part of the cache key.
    """

    def __init__(
        self,
        *,
        rank,
        world_size,
        shape,
        block_num=_DEFAULT_BLOCK_NUM,
        warp_num_per_block=_DEFAULT_WARP_NUM_PER_BLOCK,
        quant: str | tuple[str, str] | None = None,
        return_packed=False,
        v_pack: AttentionPack = AttentionPack.DEFAULT,
        softmax_scale=None,
        hadamard=None,
        valid_kv_len=None,
        group,
    ):
        self._next_role = 0
        self._role_stream = None
        spin_timeout_iters = int(os.environ.get("AITER_A2A_SPIN_TIMEOUT_ITERS", "0"))
        if group is None:
            raise ValueError(
                "group is required: a torch.distributed ProcessGroup containing "
                "exactly the MoRI participants"
            )
        self.quant = quant is not None
        if self.quant:
            if isinstance(quant, str):
                self.codecs = (quant, quant, quant)
            elif isinstance(quant, tuple):
                if len(quant) != 2:
                    raise ValueError(
                        "quant codec pair must contain exactly two entries"
                    )
                self.codecs = (quant[0], quant[0], quant[1])
            else:
                raise ValueError("quant must be a codec string or a two-codec tuple")
            for codec in self.codecs:
                if codec not in (
                    "e4m3",
                    "e4m3_pc",
                    "int8",
                    "mxfp4",
                    "mxfp6",
                    "mxfp6_p",
                    "mxfp8",
                ):
                    raise ValueError(
                        f"expected codec 'e4m3', 'e4m3_pc', 'int8', 'mxfp4', 'mxfp6', 'mxfp6_p', or 'mxfp8', got {codec}"
                    )
        else:
            self.codecs = ("e4m3",) * 3
        if "e4m3_pc" in self.codecs[:2]:
            raise ValueError("e4m3_pc is V-only; Q/K must use another codec")
        if self.codecs[2] == "e4m3_pc" and not return_packed:
            raise ValueError(
                "e4m3_pc requires return_packed=True; BF16 return is unsupported"
            )
        if "mxfp6_p" in self.codecs[:2]:
            raise ValueError("mxfp6_p is V-only; Q/K must use another codec")
        if self.codecs[2] == "mxfp6_p" and not return_packed:
            raise ValueError(
                "mxfp6_p requires return_packed=True; BF16 return is unsupported"
            )
        self.return_packed = return_packed
        self.v_pack = AttentionPack(v_pack)
        if self.codecs[2] == "mxfp6_p" and self.v_pack != AttentionPack.V_FOR_FP6_P:
            raise ValueError("mxfp6_p requires V_FOR_FP6_P")
        if self.v_pack == AttentionPack.V_FOR_FP6_P and (
            not self.return_packed or self.codecs[2] not in ("mxfp4", "mxfp6_p")
        ):
            raise ValueError(
                "V_FOR_FP6_P requires packed output with MXFP4 or mxfp6_p V"
            )
        device_index = torch.cuda.current_device()
        self.device = torch.device("cuda", device_index)
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        if arch not in {"gfx942", "gfx950"}:
            raise RuntimeError(
                f"attention A2A intranode requires gfx942 or gfx950, got {arch}"
            )
        self.fp8_fnuz = arch == "gfx942"
        if self.return_packed and self.codecs[2] not in {
            "e4m3",
            "e4m3_pc",
            "mxfp4",
            "mxfp6_p",
        }:
            raise ValueError(
                f"packed V supports only e4m3, e4m3_pc, mxfp4, and mxfp6_p, got {self.codecs[2]!r}"
            )
        if (
            self.return_packed
            and self.fp8_fnuz
            and self.codecs
            not in {
                ("int8", "int8", "e4m3"),
                ("e4m3", "e4m3", "e4m3"),
            }
        ):
            raise ValueError(
                "gfx942 packed output supports only int8/e4m3 and e4m3/e4m3"
            )
        # Amax and payload quantization must apply identical Q/K preprocessing.
        if hadamard is None:
            hadamard = self.return_packed
        self.hadamard = (
            self.quant
            and hadamard
            and not (self.return_packed and self.codecs[0] == "int8")
        )
        if self.return_packed and not self.quant:
            raise ValueError(
                "return_packed=True requires quant to be set (a codec or codec pair); nothing to pack from a bf16 passthrough"
            )
        if self.return_packed and (
            softmax_scale is None
            or not math.isfinite(softmax_scale)
            or softmax_scale <= 0
        ):
            raise ValueError("V4 Q requires an explicit positive finite softmax_scale")
        q_multiplier = (
            softmax_scale * math.log2(math.e)
            if self.return_packed and self.codecs[0] not in ("int8", "e4m3")
            else 1.0
        )
        if world_size not in (2, 4, 8):
            raise ValueError(f"world_size must be one of 2, 4, 8; got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if len(shape) != 4 or shape[0] != 1 or shape[-1] != 128:
            raise ValueError(f"expected input shape [1, S, H, 128], got {tuple(shape)}")
        if any(dim <= 0 for dim in shape):
            raise ValueError("input dimensions must be nonzero")
        if self.v_pack == AttentionPack.V_FOR_FP6_P and shape[1] % 32 != 0:
            raise ValueError("V_FOR_FP6_P requires seq_local divisible by 32")
        if self.return_packed and self.codecs[2] != "e4m3_pc" and shape[1] % 32 != 0:
            raise ValueError("V4 V output requires seq_local divisible by 32")
        if shape[2] % world_size != 0:
            raise ValueError(
                f"head count {shape[2]} must be divisible by world_size {world_size}"
            )
        if block_num <= 0 or warp_num_per_block <= 0:
            raise ValueError("launch geometry must be positive")

        numel = 1
        for dim in shape:
            numel *= dim
        if numel % world_size != 0:
            raise ValueError("input numel must divide evenly across ranks")
        payload_dtype = torch.uint8 if self.quant else torch.bfloat16
        v4_amax_exchange = tuple(
            (self.return_packed and role == 2 and codec in ("e4m3", "e4m3_pc"))
            or (self.return_packed and role in (0, 1) and codec in ("int8", "e4m3"))
            for role, codec in enumerate(self.codecs)
        )
        split_v_exchange = needs_split_v_exchange(self.v_pack, shape[1])
        role_warp_counts = tuple(
            (
                4
                if (self.return_packed and role == 2) or v4_amax_exchange[role]
                else warp_num_per_block
            )
            for role in range(3)
        )
        for role, role_warp_count in enumerate(role_warp_counts):
            total_waves = block_num * role_warp_count
            peer_waves = total_waves // world_size
            if peer_waves < 1 or total_waves % world_size:
                raise ValueError(
                    f"role {'qkv'[role]} requires total_waves={total_waves} "
                    f"divisible by world_size={world_size}"
                )

        self.shape = tuple(shape)
        self.dtype = torch.bfloat16
        heads_local = shape[2] // world_size
        seq_full = shape[1] * world_size
        if valid_kv_len is None:
            valid_kv_len = seq_full
        try:
            if isinstance(valid_kv_len, bool):
                raise TypeError
            valid_kv_len = operator.index(valid_kv_len)
        except TypeError as exc:
            raise ValueError("valid_kv_len must be an integer, not bool") from exc
        lower_bound = seq_full - shape[1]
        if not lower_bound < valid_kv_len <= seq_full:
            raise ValueError(
                f"valid_kv_len must be in ({lower_bound}, {seq_full}], "
                f"with padding confined to the last rank; got {valid_kv_len}"
            )
        if valid_kv_len != seq_full:
            unsupported = [
                f"{'qkv'[role]}={self.codecs[role]}"
                for role in (1, 2)
                if not supports_kv_seq_len("qkv"[role], self.codecs[role])
            ]
            if not self.return_packed or unsupported:
                raise ValueError(
                    f"valid_kv_len < {seq_full} requires packed output with "
                    f"token-local K/V scales; codecs={self.codecs}, "
                    f"pack={self.v_pack.name}; unsupported: "
                    f"{unsupported or 'BF16 return'}"
                )
        self.valid_kv_len = valid_kv_len
        if rank != ms.shmem_mype() or world_size != ms.shmem_npes():
            raise ValueError(
                f"rank/world_size ({rank}/{world_size}) != MoRI "
                f"PE/npes ({ms.shmem_mype()}/{ms.shmem_npes()})"
            )
        # Must run before any symmetric allocation: mismatched configs
        # would otherwise hang in the shmem barrier instead of raising.
        if group.size() != world_size:
            raise ValueError(f"group size {group.size()} != world_size {world_size}")
        if torch.distributed.get_rank(group) != rank:
            raise ValueError(
                f"rank {rank} != group rank {torch.distributed.get_rank(group)}"
            )
        config = (
            tuple(shape),
            tuple(self.codecs),
            self.quant,
            bool(return_packed),
            int(self.v_pack),
            bool(self.hadamard),
            valid_kv_len,
            block_num,
            warp_num_per_block,
            q_multiplier,
            world_size,
            spin_timeout_iters,
        )
        configs = [None] * world_size
        torch.distributed.all_gather_object(configs, config, group=group)
        if any(c != configs[0] for c in configs):
            raise ValueError(
                f"ranks disagree on AttentionA2AIntraNodeOp config: {configs}"
            )
        role_sequences = (seq_full, valid_kv_len, valid_kv_len)
        role_numels = tuple(
            sequence * heads_local * shape[3] for sequence in role_sequences
        )
        kv_tiles = (valid_kv_len + 127) // 128
        mxfp4_k_size = heads_local * kv_tiles * 8192
        mxfp4_v_size = heads_local * kv_tiles * 128 * 64 + 64
        fp6_k_payload_size = heads_local * kv_tiles * 17408 + 256
        fp6_v_payload_size = mxfp6_v_raw_buffer_size(1, valid_kv_len, heads_local)
        fp6_k_scale_size = valid_kv_len * heads_local * 4 + 64
        if self.return_packed and any(codec == "mxfp4" for codec in self.codecs):
            from aiter.ops.mha_v4_quant import (
                mxfp4_k_raw_buffer_size,
                mxfp4_v_raw_buffer_size,
            )

            mxfp4_k_size = mxfp4_k_raw_buffer_size(1, valid_kv_len, heads_local)
            mxfp4_v_size = mxfp4_v_raw_buffer_size(1, valid_kv_len, heads_local)
        if self.return_packed and any(codec == "mxfp6" for codec in self.codecs):
            from aiter.ops.triton.quant.mxfp6_fmha_pack import fp6_k_raw_buffer_sizes

            fp6_k_payload_size, fp6_k_scale_size = fp6_k_raw_buffer_sizes(
                1, valid_kv_len, heads_local
            )
        payload_sizes = tuple(
            _payload_size(
                role,
                codec,
                self.return_packed,
                self.quant,
                role_numels[role],
                mxfp4_k_size,
                fp6_k_payload_size,
                mxfp4_v_size,
                fp6_v_payload_size,
            )
            for role, codec in enumerate(self.codecs)
        )
        scale_sizes = tuple(
            _scale_size(
                role,
                codec,
                per_tensor,
                self.return_packed,
                role_numels[role],
                heads_local,
                world_size,
                block_num,
                valid_kv_len,
                fp6_k_scale_size,
            )
            for role, (codec, per_tensor) in enumerate(
                zip(self.codecs, v4_amax_exchange)
            )
        )
        scale_storage_sizes = scale_sizes
        if self.return_packed:
            # Mirror mha_v4_quant's consumer-owned backing extent for speculative gathers.
            padding_sizes = tuple(
                _scale_padding_size(
                    role, codec, role_sequences[role], valid_kv_len, heads_local
                )
                for role, codec in enumerate(self.codecs)
            )
            scale_storage_sizes = tuple(
                size + padding for size, padding in zip(scale_sizes, padding_sizes)
            )
        for role, size in enumerate(payload_sizes):
            _require_raw_buffer_extent(
                f"{'qkv'[role]} output",
                size * torch.empty((), dtype=payload_dtype).element_size(),
            )
        for role, (size, per_tensor) in enumerate(
            zip(scale_storage_sizes, v4_amax_exchange)
        ):
            _require_raw_buffer_extent(
                f"{'qkv'[role]} scale",
                size * (4 if per_tensor else 1),
            )
        for role in range(3):
            _require_raw_buffer_extent(f"{'qkv'[role]} input", numel * 2)
        for name, extent in (
            ("p2p output table", world_size * 8),
            ("p2p scale table", world_size * 8),
            ("xdb memory", world_size * 8),
            ("reuse memory", world_size * 8),
            ("reuse flags", 2 * 8),
            ("xdb flag", 8),
        ):
            _require_raw_buffer_extent(name, extent)
        if split_v_exchange:
            _require_raw_buffer_extent("V partial amax", shape[2] * 1024)
        self.v_pc_partial = None
        if self.codecs[2] == "e4m3_pc":
            partial_size = v_pc_partial_elems(shape[1], shape[2], shape[3])
            _require_raw_buffer_extent("V per-channel partial amax", partial_size * 4)
            self.v_pc_partial = torch.empty(
                partial_size, dtype=torch.float32, device=self.device
            )
        self.v_partial_sets = (
            tuple(
                mori_shmem_create_tensor(
                    (world_size // 2, 2, heads_local, 256), torch.float32
                )
                for _ in range(2)
            )
            if split_v_exchange
            else ()
        )
        self.v_partial_ready_sets = (
            tuple(
                mori_shmem_create_tensor((world_size, heads_local), torch.int64)
                for _ in range(2)
            )
            if split_v_exchange
            else ()
        )
        for ready in self.v_partial_ready_sets:
            ready.zero_()
        self.outputs_sets = tuple(
            tuple(
                mori_shmem_create_tensor((size,), payload_dtype)
                for size in payload_sizes
            )
            for _ in range(2)
        )
        # FP32 descales precede the replicated partial-max exchange slots.
        self.scale_storage_sets = tuple(
            tuple(
                mori_shmem_create_tensor(
                    (size,),
                    torch.float32 if per_tensor else torch.uint8,
                )
                for size, per_tensor in zip(scale_storage_sizes, v4_amax_exchange)
            )
            for _ in range(2)
        )
        self.scales_sets = tuple(
            tuple(
                (
                    scale[: heads_local * 128].view(1, heads_local, 128)
                    if codec == "e4m3_pc"
                    else scale[:1] if per_tensor else scale[:size]
                )
                for scale, per_tensor, size, codec in zip(
                    scales, v4_amax_exchange, scale_sizes, self.codecs
                )
            )
            for scales in self.scale_storage_sets
        )
        device = self.device
        self.bf16_outputs_sets = ()
        self._dequant_launch = None
        if self.quant and not self.return_packed:
            # Local consumers retain the same two-parity lifetime as received payloads.
            self.bf16_outputs_sets = tuple(
                tuple(
                    torch.empty(numel, dtype=torch.bfloat16, device=device)
                    for _ in range(3)
                )
                for _ in range(2)
            )
            self._dequant_launch = make_attention_a2a_dequant_jit(
                numel=numel, codec=self.codecs, fp8_fnuz=self.fp8_fnuz
            )
        self.xdb_mem = mori_shmem_create_tensor((world_size,), torch.int64)
        # Consumer-drain readiness is separate from producer-complete flags.
        self.reuse_mem_sets = tuple(
            mori_shmem_create_tensor((world_size,), torch.int64) for _ in range(2)
        )
        for ready_mem in self.reuse_mem_sets:
            ready_mem.zero_()
        self.reuse_flags = torch.ones(2, dtype=torch.int64, device=device)
        for scales in self.scales_sets:
            for role, (scale, codec) in enumerate(zip(scales, self.codecs)):
                if self.return_packed and role == 1 and codec == "mxfp6":
                    scale.zero_()
        for storage, scales in zip(self.scale_storage_sets, self.scales_sets):
            for backing, scale in zip(storage, scales):
                backing[scale.numel() :].zero_()
        for outputs in self.outputs_sets:
            for output in outputs:
                output.zero_()
        self.xdb_mem.zero_()
        self.xdb_flag = torch.ones(1, dtype=torch.int64, device=device)

        # The MoRI barrier is host-only: peers must not publish into our flags
        # before our zeroing has completed on the device. The zeroing is enqueued
        # on the current stream, so syncing it avoids waiting on unrelated work.
        torch.cuda.current_stream(self.device).synchronize()
        ms.shmem_barrier_all()
        self.p2p_outputs_sets = tuple(
            tuple(
                build_p2p_table(output, rank, world_size, device) for output in outputs
            )
            for outputs in self.outputs_sets
        )
        self.p2p_scales_sets = tuple(
            tuple(build_p2p_table(scale, rank, world_size, device) for scale in scales)
            for scales in self.scale_storage_sets
        )
        self.p2p_v_partial_sets = tuple(
            build_p2p_table(partial, rank, world_size, device)
            for partial in self.v_partial_sets
        )
        self.p2p_v_partial_ready_sets = tuple(
            build_p2p_table(ready, rank, world_size, device)
            for ready in self.v_partial_ready_sets
        )
        self.p2p_xdb_mem = build_p2p_table(self.xdb_mem, rank, world_size, device)
        self.p2p_reuse_mem_sets = tuple(
            build_p2p_table(ready_mem, rank, world_size, device)
            for ready_mem in self.reuse_mem_sets
        )
        ms.shmem_barrier_all()
        # shmem_ptr_p2p translates by the offset from the local heap base, so
        # divergent allocation order would silently address the wrong buffer.
        sym = [("output", t) for outs in self.outputs_sets for t in outs]
        sym += [("scale", t) for sc in self.scale_storage_sets for t in sc]
        sym += [("v_partial", t) for t in self.v_partial_sets]
        sym += [("v_partial_ready", t) for t in self.v_partial_ready_sets]
        sym += [("xdb", self.xdb_mem)]
        sym += [("reuse", t) for t in self.reuse_mem_sets]
        base = sym[0][1].data_ptr()
        offsets = [(n, t.data_ptr() - base) for n, t in sym]
        all_offsets = [None] * world_size
        torch.distributed.all_gather_object(all_offsets, offsets, group=group)
        for i, mine in enumerate(offsets):
            if any(o[i] != all_offsets[0][i] for o in all_offsets):
                raise ValueError(
                    f"symmetric buffer {i} ({mine[0]}) offset differs across ranks: "
                    f"{[o[i][1] for o in all_offsets]}"
                )
        self._epoch = 0
        self._reuse_launch = make_attention_a2a_reuse_jit(
            rank=rank, npes=world_size, spin_timeout_iters=spin_timeout_iters
        )

        role_count = 3
        self._jit_kwargs = tuple(
            {
                "rank": rank,
                "npes": world_size,
                "heads": shape[2],
                "seq_len": shape[1],
                "head_dim": shape[3],
                "block_num": block_num,
                "warp_num_per_block": role_warp_counts[i],
                "quant": self.quant,
                "codec": self.codecs[i],
                "hadamard": self.hadamard and i < 2,
                "v4_output": _role_format(self.return_packed, i),
                "valid_kv_len": role_sequences[i],
                **({"v_pack": self.v_pack} if i == 2 else {}),
                "q_multiplier": q_multiplier,
                "fp8_fnuz": self.fp8_fnuz,
                "spin_timeout_iters": spin_timeout_iters,
            }
            for i in range(role_count)
        )
        self._launches = tuple(
            make_attention_a2a_jit(**kwargs) for kwargs in self._jit_kwargs
        )
        self._amax_launches = tuple(
            (
                make_attention_a2a_jit(
                    **(kwargs | {"warp_num_per_block": 4, "v4_amax": True})
                )
                if v4_amax_exchange[i]
                else None
            )
            for i, kwargs in enumerate(self._jit_kwargs)
        )

    def submit_role(self, role, input, stream=None):
        """Submit Q/K/V in order on one stream, retaining one parity for the trio.

        Producers and prior consumers must be ordered before their side-stream
        reads/writes. The caller retains inputs until V completes on that stream.

        Returns, per mode:
        - return_packed: PackedRoleResult(payload, scale) for every role.
        - quantized BF16 (dequant): None for Q/K; the three BF16 outputs after V.
        - lossless BF16: None for Q/K; the three outputs after V.
        """
        if role not in (0, 1, 2) or role != self._next_role:
            raise ValueError("submit Q, K, V in order without interleaving trios")
        if input.dtype != self.dtype or tuple(input.shape) != self.shape:
            raise ValueError(f"expected {self.dtype} input with shape {self.shape}")
        if not input.is_cuda or not input.is_contiguous():
            raise ValueError("input must be a contiguous CUDA tensor")
        torch_stream = torch.cuda.current_stream() if stream is None else stream
        if input.device != self.device or torch_stream.device != self.device:
            raise ValueError("input and stream must use the owning device")
        if (
            self._role_stream is not None
            and self._role_stream != torch_stream.cuda_stream
        ):
            raise ValueError(
                "all trios must use one stream; cross-stream use requires "
                "constructing a new op or joining the streams"
            )
        self._role_stream = torch_stream.cuda_stream
        stream = Stream(torch_stream)
        parity = self._epoch % 2
        if role == 0:
            reuse_args = (
                self.reuse_mem_sets[parity].data_ptr(),
                self.p2p_reuse_mem_sets[parity].data_ptr(),
                self.reuse_flags[parity : parity + 1].data_ptr(),
                stream,
            )
            _run_compiled(
                self._reuse_launch,
                *(fx.Int64(arg) for arg in reuse_args[:-1]),
                reuse_args[-1],
            )
        args = (
            input.data_ptr(),
            self.p2p_outputs_sets[parity][role].data_ptr(),
            self.p2p_scales_sets[parity][role].data_ptr(),
            self.xdb_mem.data_ptr(),
            self.p2p_xdb_mem.data_ptr(),
            self.xdb_flag.data_ptr(),
            (
                self.p2p_v_partial_sets[parity].data_ptr()
                if role == 2 and self.p2p_v_partial_sets
                else (
                    self.v_pc_partial.data_ptr()
                    if role == 2 and self.v_pc_partial is not None
                    else 0
                )
            ),
            (
                self.p2p_v_partial_ready_sets[parity].data_ptr()
                if role == 2 and self.p2p_v_partial_ready_sets
                else 0
            ),
            stream,
        )
        self._submit_role_launch(role, args)
        self._next_role = (role + 1) % 3
        packed_result = (
            PackedRoleResult(
                self.outputs_sets[parity][role], self.scales_sets[parity][role]
            )
            if self.return_packed
            else None
        )
        if role == 2:
            if self.return_packed:
                self._epoch += 1
                return packed_result
            if self.quant:
                bf16_outputs = self.bf16_outputs_sets[parity]
                args = (
                    *(output.data_ptr() for output in self.outputs_sets[parity]),
                    *(scale.data_ptr() for scale in self.scales_sets[parity]),
                    *(output.data_ptr() for output in bf16_outputs),
                    stream,
                )
                # V completes the trio's receive-acquire handshake on this stream.
                _run_compiled(
                    self._dequant_launch,
                    *(fx.Int64(arg) for arg in args[:-1]),
                    args[-1],
                )
                self._epoch += 1
                return bf16_outputs
            self._epoch += 1
            return self.outputs_sets[parity]
        return packed_result

    def _submit_role_launch(self, i, args):
        launch_args = (*(fx.Int64(arg) for arg in args[:-1]), args[-1])
        if self._amax_launches[i] is not None:
            _run_compiled(self._amax_launches[i], *launch_args)
        _run_compiled(self._launches[i], *launch_args)
