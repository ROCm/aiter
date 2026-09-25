# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the FlyDSL single-tensor intranode push all-to-all."""

from __future__ import annotations

import functools
import math

import flydsl.expr as fx
import mori.shmem as ms
import torch
from flydsl.expr.typing import Stream
from mori.shmem import mori_shmem_create_tensor

from aiter.ops.mha_v4 import AttentionPack

from .communication_ops_host_utils import build_p2p_table
from .kernels.attention_a2a_intranode_kernel import (
    _transport_bytes,
    make_attention_a2a_dequant_jit,
    make_attention_a2a_jit,
    make_attention_a2a_reuse_jit,
)
from .kernels.tensor_shim import _run_compiled

_DEFAULT_BLOCK_NUM = 128
_DEFAULT_WARP_NUM_PER_BLOCK = 8
_MAX_RAW_BUFFER_BYTES = 0xFFFFFFFF


def _role_format(return_packed, role):
    return "qkv"[role] if return_packed else ""


@functools.cache
def _device_cu_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _require_raw_buffer_extent(role, extent):
    if extent > _MAX_RAW_BUFFER_BYTES:
        raise ValueError(f"{role} extent {extent} exceeds raw-buffer limit")


class AttentionA2AIntraNodeOp:
    """Own symmetric receive and handshake buffers for one tensor shape.

    Submit Q/K/V as three ordered per-role launches.
    Set quant to one codec for uniform Q/K/V quantization or a (Q/K, V) codec pair.
    Native codecs use one E8M0 scale per 32 adjacent values. Raw MXFP4 payloads
    contain two E2M1 values per byte, low nibble first. Raw MXFP6 payloads contain
    contiguous E2M3 six-bit codes, least-significant bits first.
    Quantized calls return locally dequantized bf16 Q/K/V by default.
    Set return_packed=True for MHA V4 bytes in
    (outputs, (q_scales, k_scales, v_scales)); otherwise results are dequantized to bf16.
    e4m3 is the device's native FP8 (E4M3FNUZ on gfx942, OCP E4M3FN on gfx950).
    Packed MX formats are gfx950-only. Packed outputs use role-specific scale layouts:
    MXFP4 V uses token-axis scales in gather order, and E4M3 V uses one float32
    descale per destination tensor. hadamard=True applies normalized Walsh-Hadamard
    to Q/K before quantization;
    V is unchanged. The default is enabled for packed output and disabled otherwise.
    v_pack=AttentionPack.V_FOR_FP6_P selects FP6-P ordering for packed MXFP4 V
    and requires seq_local divisible by 64; DEFAULT retains the standard ordering.
    Packed INT8 does not apply Hadamard rotation. MX Q folds softmax_scale * log2(e).
    All ranks must use the same mode and serialize calls on one stream.
    Consumers must finish on that stream before the next call reusing their
    parity (two calls later); cross-stream consumers require an explicit join.
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
    ):
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
                if codec not in ("e4m3", "int8", "mxfp4", "mxfp6", "mxfp8"):
                    raise ValueError(
                        f"expected codec 'e4m3', 'int8', 'mxfp4', 'mxfp6', or 'mxfp8', got {codec}"
                    )
        else:
            self.codecs = ("e4m3",) * 3
        self.return_packed = return_packed
        self.v_pack = AttentionPack(v_pack)
        if self.v_pack == AttentionPack.V_FOR_FP6_P and (
            not self.return_packed or self.codecs[2] != "mxfp4"
        ):
            raise ValueError("V_FOR_FP6_P requires packed output with MXFP4 V")
        device_index = torch.cuda.current_device()
        self.device = torch.device("cuda", device_index)
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        if arch not in {"gfx942", "gfx950"}:
            raise RuntimeError(
                f"attention A2A intranode requires gfx942 or gfx950, got {arch}"
            )
        self.fp8_fnuz = arch == "gfx942"
        if self.return_packed and self.codecs[2] not in {"e4m3", "mxfp4"}:
            raise ValueError(
                f"packed V supports only e4m3 and mxfp4, got {self.codecs[2]!r}"
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
        self.hadamard = self.quant and hadamard
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
        if self.v_pack == AttentionPack.V_FOR_FP6_P and shape[1] % 64 != 0:
            raise ValueError("V_FOR_FP6_P requires seq_local divisible by 64")
        if self.return_packed and shape[1] % 32 != 0:
            raise ValueError("V4 V output requires seq_local divisible by 32")
        if shape[2] % world_size != 0:
            raise ValueError(
                f"head count {shape[2]} must be divisible by world_size {world_size}"
            )
        if block_num <= 0 or warp_num_per_block <= 0:
            raise ValueError("launch geometry must be positive")
        cu_count = _device_cu_count(device_index)
        if block_num > cu_count:
            raise ValueError(
                f"block_num={block_num} exceeds device CU count ({cu_count}); "
                "the grid-wide barrier requires all blocks to be resident"
            )

        numel = 1
        for dim in shape:
            numel *= dim
        if numel % world_size != 0:
            raise ValueError("input numel must divide evenly across ranks")
        payload_dtype = torch.uint8 if self.quant else torch.bfloat16
        v4_per_tensor = tuple(
            (self.return_packed and role == 2 and codec == "e4m3")
            or (self.return_packed and role in (0, 1) and codec in ("int8", "e4m3"))
            for role, codec in enumerate(self.codecs)
        )
        role_warp_counts = tuple(
            (
                4
                if (self.return_packed and role == 2) or v4_per_tensor[role]
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
        mxfp4_k_size = heads_local * ((seq_full + 127) // 128) * 8192
        mxfp4_v_size = heads_local * ((seq_full + 127) // 128) * 128 * 64 + 64
        fp6_k_payload_size = heads_local * ((seq_full + 127) // 128) * 17408 + 256
        fp6_k_scale_size = seq_full * heads_local * 4 + 64
        if self.return_packed and any(codec == "mxfp4" for codec in self.codecs):
            from aiter.ops.mha_v4_quant import (
                mxfp4_k_raw_buffer_size,
                mxfp4_v_raw_buffer_size,
            )

            mxfp4_k_size = mxfp4_k_raw_buffer_size(1, seq_full, heads_local)
            mxfp4_v_size = mxfp4_v_raw_buffer_size(1, seq_full, heads_local)
        if self.return_packed and any(codec == "mxfp6" for codec in self.codecs):
            from aiter.ops.triton.quant.mxfp6_fmha_pack import fp6_k_raw_buffer_sizes

            fp6_k_payload_size, fp6_k_scale_size = fp6_k_raw_buffer_sizes(
                1, seq_full, heads_local
            )
        payload_sizes = tuple(
            (
                mxfp4_k_size
                if self.return_packed and role == 1 and codec == "mxfp4"
                else (
                    fp6_k_payload_size
                    if self.return_packed and role == 1 and codec == "mxfp6"
                    else (
                        mxfp4_v_size
                        if self.return_packed and role == 2 and codec == "mxfp4"
                        else (_transport_bytes(numel, codec) if self.quant else numel)
                    )
                )
            )
            for role, codec in enumerate(self.codecs)
        )
        scale_sizes = tuple(
            (
                1 + world_size * block_num * 4
                if per_tensor
                else (
                    heads_local * ((seq_full + 127) // 128) * 512
                    if self.return_packed and role == 2
                    else (
                        fp6_k_scale_size
                        if self.return_packed and role == 1 and codec == "mxfp6"
                        else numel // 32
                    )
                )
            )
            for role, (codec, per_tensor) in enumerate(zip(self.codecs, v4_per_tensor))
        )
        scale_storage_sizes = scale_sizes
        if self.return_packed:
            from aiter.ops.mha_v4_quant import (
                MHA_V4_KV_SCALE_LOOKAHEAD_ROWS,
                MHA_V4_KV_TILE_ROWS,
                MHA_V4_MXFP4_K_SCALE_SLACK_BYTES,
                MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,
                MHA_V4_QUERY_TILE_ROWS,
            )

            # Mirror mha_v4_quant's consumer-owned backing extent for speculative gathers.
            padding_sizes = tuple(
                (
                    (
                        (
                            (seq_full + MHA_V4_QUERY_TILE_ROWS - 1)
                            // MHA_V4_QUERY_TILE_ROWS
                        )
                        * MHA_V4_QUERY_TILE_ROWS
                        - seq_full
                    )
                    * heads_local
                    * 4
                    if role == 0 and codec.startswith("mxfp")
                    else (
                        (
                            (
                                (seq_full + MHA_V4_KV_TILE_ROWS - 1)
                                // MHA_V4_KV_TILE_ROWS
                            )
                            * MHA_V4_KV_TILE_ROWS
                            + MHA_V4_KV_SCALE_LOOKAHEAD_ROWS
                            - seq_full
                        )
                        * heads_local
                        * 4
                        + MHA_V4_MXFP4_K_SCALE_SLACK_BYTES
                        if role == 1 and codec == "mxfp4"
                        else (
                            MHA_V4_MXFP4_V_SCALE_SLACK_BYTES
                            if role == 2 and codec == "mxfp4"
                            else 0
                        )
                    )
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
            zip(scale_storage_sizes, v4_per_tensor)
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
            ("grid barrier", 4),
        ):
            _require_raw_buffer_extent(name, extent)
        self.outputs_sets = tuple(
            tuple(
                mori_shmem_create_tensor((size,), payload_dtype)
                for size in payload_sizes
            )
            for _ in range(2)
        )
        # Per-tensor outputs retain exchanged amax metadata after their scalar descale.
        self.scale_storage_sets = tuple(
            tuple(
                mori_shmem_create_tensor(
                    (size,),
                    torch.float32 if per_tensor else torch.uint8,
                )
                for size, per_tensor in zip(scale_storage_sizes, v4_per_tensor)
            )
            for _ in range(2)
        )
        self.scales_sets = tuple(
            tuple(
                scale[:1] if per_tensor else scale[:size]
                for scale, per_tensor, size in zip(scales, v4_per_tensor, scale_sizes)
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
        self.grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)

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
        self.p2p_xdb_mem = build_p2p_table(self.xdb_mem, rank, world_size, device)
        self.p2p_reuse_mem_sets = tuple(
            build_p2p_table(ready_mem, rank, world_size, device)
            for ready_mem in self.reuse_mem_sets
        )
        ms.shmem_barrier_all()
        self._epoch = 0
        self._reuse_launch = make_attention_a2a_reuse_jit(rank=rank, npes=world_size)

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
                **({"v_pack": self.v_pack} if i == 2 else {}),
                "q_multiplier": q_multiplier,
                "fp8_fnuz": self.fp8_fnuz,
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
                if v4_per_tensor[i]
                else None
            )
            for i, kwargs in enumerate(self._jit_kwargs)
        )

    def submit_role(self, role, input, stream=None):
        """Submit Q/K/V in order on one stream, retaining one parity for the trio.

        Producers and prior consumers must be ordered before their side-stream
        reads/writes. The caller retains inputs until V completes on that stream.
        """
        if role not in (0, 1, 2) or role != getattr(self, "_next_role", 0):
            raise ValueError("submit Q, K, V in order without interleaving trios")
        if input.dtype != self.dtype or tuple(input.shape) != self.shape:
            raise ValueError(f"expected {self.dtype} input with shape {self.shape}")
        if not input.is_cuda or not input.is_contiguous():
            raise ValueError("input must be a contiguous CUDA tensor")
        torch_stream = torch.cuda.current_stream() if stream is None else stream
        if input.device != self.device or torch_stream.device != self.device:
            raise ValueError("input and stream must use the owning device")
        if role and self._role_stream != torch_stream.cuda_stream:
            raise ValueError("all roles must use the same stream")
        self._role_stream = torch_stream.cuda_stream
        stream = Stream(torch_stream)
        parity = self._epoch % 2
        if role == 0:
            reuse_args = (
                self.reuse_mem_sets[parity].data_ptr(),
                self.p2p_reuse_mem_sets[parity].data_ptr(),
                self.reuse_flags.data_ptr() + parity * self.reuse_flags.element_size(),
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
            self.grid_barrier.data_ptr(),
            stream,
        )
        self._submit_role_launch(role, args)
        self._next_role = (role + 1) % 3
        if role == 2:
            if self.return_packed:
                self._epoch += 1
                return self.outputs_sets[parity], self.scales_sets[parity]
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
        return None

    def _submit_role_launch(self, i, args):
        launch_args = (*(fx.Int64(arg) for arg in args[:-1]), args[-1])
        if self._amax_launches[i] is not None:
            _run_compiled(self._amax_launches[i], *launch_args)
        _run_compiled(self._launches[i], *launch_args)
