# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Quantized inverse-Ulysses attention-output all-to-all over MoRI."""

from __future__ import annotations

import operator
from typing import NamedTuple

import flydsl.expr as fx
import mori.shmem as ms
import torch
from flydsl.expr.typing import Stream
from mori.shmem import mori_shmem_create_tensor

from .communication_ops_host_utils import build_p2p_table
from .kernels.attention_a2a_intranode_kernel import make_attention_a2a_reuse_jit
from .kernels.attention_output_a2a_intranode_kernel import (
    make_attention_output_a2a_jit,
    make_attention_output_packed_copy_jit,
)
from .kernels.tensor_shim import _run_compiled

_DEFAULT_BLOCK_NUM = 128
_DEFAULT_WARP_NUM_PER_BLOCK = 8
_MAX_RAW_BUFFER_BYTES = 0xFFFFFFFF


class PackedAttentionOutputResult(NamedTuple):
    """Codec payload and scales in the following GEMM's physical layout.

    Both tensors alias double-buffered symmetric storage. They are reused two
    calls later, so consumers must finish before the next call that reuses the
    parity. A consumer on another stream must be joined explicitly.
    """

    payload: torch.Tensor
    scale: torch.Tensor


def _checked_p2p_table(tensor, rank, world_size, device):
    table = build_p2p_table(tensor, rank, world_size, device)
    values = table.cpu().tolist()
    missing = [peer for peer, pointer in enumerate(values) if pointer == 0]
    return table, missing


class AttentionOutputA2AIntraNodeOp:
    """Quantize BF16 MHA output and perform the inverse Ulysses redistribution.

    Input is contiguous BSHD ``[1, S_global, H_local, 128]``. E4M3 returns
    destination-local BSHD payload and FP32 per-128 scales. MXFP4 and MXFP6
    return payload/scales directly in their following GEMM's physical layout.

    ``group`` is mandatory and must contain exactly the MoRI participants.
    Construction checks rank agreement before allocating symmetric buffers.
    """

    def __init__(
        self,
        *,
        rank,
        world_size,
        shape,
        group,
        codec="e4m3",
        block_num=_DEFAULT_BLOCK_NUM,
        warp_num_per_block=_DEFAULT_WARP_NUM_PER_BLOCK,
    ):
        if group is None:
            raise ValueError("group is required for cross-rank configuration checks")
        codec = codec.strip().lower()
        if codec not in {"e4m3", "e4m3_tensor", "mxfp4", "mxfp6"}:
            raise ValueError(f"unsupported attention output codec {codec!r}")
        if world_size not in (2, 4, 8):
            raise ValueError(f"world_size must be one of 2, 4, 8; got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if group.size() != world_size:
            raise ValueError(f"group size {group.size()} != world_size {world_size}")
        if torch.distributed.get_rank(group) != rank:
            raise ValueError(
                f"group rank {torch.distributed.get_rank(group)} != rank {rank}"
            )
        if len(shape) != 4 or shape[0] != 1 or shape[-1] != 128:
            raise ValueError(
                f"expected input shape [1, S_global, H_local, 128], got {tuple(shape)}"
            )
        try:
            shape = tuple(operator.index(dim) for dim in shape)
        except TypeError as exc:
            raise ValueError("shape dimensions must be integers") from exc
        if any(dim <= 0 for dim in shape):
            raise ValueError("input dimensions must be nonzero")
        if shape[1] % world_size:
            raise ValueError(
                f"global sequence {shape[1]} must divide by world_size {world_size}"
            )
        if block_num <= 0 or warp_num_per_block <= 0:
            raise ValueError("launch geometry must be positive")
        total_warps = block_num * warp_num_per_block
        if total_warps < world_size or total_warps % world_size:
            raise ValueError(
                f"total waves {total_warps} must be divisible by world_size {world_size}"
            )

        device = torch.device("cuda", torch.cuda.current_device())
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        if arch not in {"gfx942", "gfx950"}:
            raise RuntimeError(
                f"attention output A2A requires gfx942 or gfx950, got {arch}"
            )
        self.fp8_fnuz = arch == "gfx942"
        if codec in {"e4m3_tensor", "mxfp4", "mxfp6"} and arch != "gfx950":
            raise RuntimeError(f"{codec} attention output requires gfx950")
        self.codec = codec
        self.rank = rank
        self.world_size = world_size
        self.shape = shape
        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.seq_local = shape[1] // world_size
        self.heads_local = shape[2]
        self.heads_total = shape[2] * world_size
        self.output_shape = (
            shape[0],
            self.seq_local,
            self.heads_total,
            shape[3],
        )
        self.scale_shape = self.output_shape[:-1]
        rows = self.seq_local
        features = self.heads_total * shape[3]
        if codec in {"e4m3", "e4m3_tensor"}:
            payload_size = math_prod(self.output_shape)
            scale_dtype = torch.float32
            result_payload_shape = self.output_shape
            if codec == "e4m3_tensor":
                scale_size = 1
                result_scale_shape = (1, 1)
            else:
                scale_size = math_prod(self.scale_shape)
                result_scale_shape = self.scale_shape
        elif codec == "mxfp4":
            padded_rows = (rows + 255) // 256 * 256
            padded_scale_columns = (features // 32 + 7) // 8 * 8
            payload_size = rows * features // 2
            scale_size = padded_rows * padded_scale_columns
            scale_dtype = torch.uint8
            result_payload_shape = (rows, features // 2)
            result_scale_shape = (padded_rows, padded_scale_columns)
        else:
            from aiter.ops.gemm_op_a6w6 import mxfp6_gemm_pack_size

            payload_size, scale_size = mxfp6_gemm_pack_size(rows, features)
            scale_dtype = torch.uint8
            result_payload_shape = (payload_size,)
            result_scale_shape = (scale_size,)
        if payload_size > _MAX_RAW_BUFFER_BYTES:
            raise ValueError(f"payload extent {payload_size} exceeds raw-buffer limit")
        scale_extent = scale_size * torch.empty((), dtype=scale_dtype).element_size()
        if scale_extent > _MAX_RAW_BUFFER_BYTES:
            raise ValueError(f"scale extent {scale_extent} exceeds raw-buffer limit")

        config = (
            shape,
            block_num,
            warp_num_per_block,
            world_size,
            codec,
        )
        configs = [None] * world_size
        torch.distributed.all_gather_object(configs, config, group=group)
        if any(peer_config != configs[0] for peer_config in configs):
            raise ValueError(
                f"ranks disagree on AttentionOutputA2AIntraNodeOp config: {configs}"
            )

        self.payload_storage_sets = tuple(
            mori_shmem_create_tensor((payload_size,), torch.uint8) for _ in range(2)
        )
        self.scale_storage_sets = tuple(
            mori_shmem_create_tensor((scale_size,), scale_dtype) for _ in range(2)
        )
        self.results = tuple(
            PackedAttentionOutputResult(
                payload.view(result_payload_shape),
                scale.view(result_scale_shape),
            )
            for payload, scale in zip(
                self.payload_storage_sets, self.scale_storage_sets
            )
        )
        self._canonical_pack = codec in {"mxfp4", "mxfp6"}
        self._pack_per_destination = False
        self._local_payload_segment_size = None
        self._local_scale_segment_size = None
        self.local_payload_sets = ()
        self.local_scale_sets = ()
        self._pack_local = None
        if self._canonical_pack and codec == "mxfp4":
            import aiter
            from aiter.ops.quant import dynamic_per_group_scaled_quant

            local_features = self.heads_local * shape[3]
            self.local_payload_sets = tuple(
                torch.empty(
                    shape[1],
                    local_features // 2,
                    dtype=aiter.dtypes.fp4x2,
                    device=device,
                )
                for _ in range(2)
            )
            self.local_scale_sets = tuple(
                torch.empty(
                    shape[1],
                    local_features // 32,
                    dtype=aiter.dtypes.fp8_e8m0,
                    device=device,
                )
                for _ in range(2)
            )
            self._pack_local = dynamic_per_group_scaled_quant
        elif self._canonical_pack and codec == "mxfp6":
            from aiter.ops.gemm_op_a6w6 import (
                mxfp6_gemm_pack_size,
                quant_mxfp6_gemm_out,
            )

            local_features = self.heads_local * shape[3]
            self._pack_per_destination = self.seq_local % 256 != 0
            pack_rows = self.seq_local if self._pack_per_destination else shape[1]
            segment_payload_size, segment_scale_size = mxfp6_gemm_pack_size(
                pack_rows, local_features
            )
            segments = world_size if self._pack_per_destination else 1
            local_payload_size = segment_payload_size * segments
            local_scale_size = segment_scale_size * segments
            self._local_payload_segment_size = segment_payload_size
            self._local_scale_segment_size = segment_scale_size
            self.local_payload_sets = tuple(
                torch.empty(
                    local_payload_size,
                    dtype=torch.uint8,
                    device=device,
                )
                for _ in range(2)
            )
            self.local_scale_sets = tuple(
                torch.empty(
                    local_scale_size,
                    dtype=torch.uint8,
                    device=device,
                )
                for _ in range(2)
            )
            self._pack_local = quant_mxfp6_gemm_out
        self.xdb_mem = mori_shmem_create_tensor((world_size,), torch.int64)
        self.reuse_mem_sets = tuple(
            mori_shmem_create_tensor((world_size,), torch.int64) for _ in range(2)
        )
        self.reuse_flags = torch.ones(2, dtype=torch.int64, device=device)
        self.xdb_flag = torch.ones(1, dtype=torch.int64, device=device)

        for payload in self.payload_storage_sets:
            payload.zero_()
        for scale in self.scale_storage_sets:
            scale.view(torch.uint8).zero_()
        for payload in self.local_payload_sets:
            payload.view(torch.uint8).zero_()
        for scale in self.local_scale_sets:
            scale.view(torch.uint8).zero_()
        self.xdb_mem.zero_()
        for reuse_mem in self.reuse_mem_sets:
            reuse_mem.zero_()

        # MoRI's host barrier does not order preceding device writes.
        torch.cuda.current_stream(device).synchronize()
        ms.shmem_barrier_all()

        missing = {}
        self.p2p_payload_sets = []
        self.p2p_scale_sets = []
        for parity, payload in enumerate(self.payload_storage_sets):
            table, peers = _checked_p2p_table(payload, rank, world_size, device)
            self.p2p_payload_sets.append(table)
            if peers:
                missing[f"payload[{parity}]"] = peers
        for parity, scale in enumerate(self.scale_storage_sets):
            table, peers = _checked_p2p_table(scale, rank, world_size, device)
            self.p2p_scale_sets.append(table)
            if peers:
                missing[f"scale[{parity}]"] = peers
        self.p2p_xdb_mem, peers = _checked_p2p_table(
            self.xdb_mem, rank, world_size, device
        )
        if peers:
            missing["completion"] = peers
        self.p2p_reuse_mem_sets = []
        for parity, reuse_mem in enumerate(self.reuse_mem_sets):
            table, peers = _checked_p2p_table(reuse_mem, rank, world_size, device)
            self.p2p_reuse_mem_sets.append(table)
            if peers:
                missing[f"reuse[{parity}]"] = peers
        self.p2p_payload_sets = tuple(self.p2p_payload_sets)
        self.p2p_scale_sets = tuple(self.p2p_scale_sets)
        self.p2p_reuse_mem_sets = tuple(self.p2p_reuse_mem_sets)

        missing_by_rank = [None] * world_size
        torch.distributed.all_gather_object(missing_by_rank, missing, group=group)
        if any(peer_missing for peer_missing in missing_by_rank):
            raise RuntimeError(f"MoRI P2P pointers unavailable: {missing_by_rank}")
        ms.shmem_barrier_all()

        self._reuse_launch = make_attention_a2a_reuse_jit(rank=rank, npes=world_size)
        launch_factory = (
            make_attention_output_packed_copy_jit
            if self._canonical_pack
            else make_attention_output_a2a_jit
        )
        launch_kwargs = {
            "rank": rank,
            "npes": world_size,
            "seq_local": self.seq_local,
            "heads_local": self.heads_local,
            "head_dim": shape[3],
            "block_num": block_num,
            "warp_num_per_block": warp_num_per_block,
            "codec": codec,
        }
        if not self._canonical_pack:
            launch_kwargs["fp8_fnuz"] = self.fp8_fnuz
        self._launch = launch_factory(**launch_kwargs)
        self._epoch = 0

    def __call__(self, input, stream=None, tensorwise_scales=None):
        if input.dtype != self.dtype or tuple(input.shape) != self.shape:
            raise ValueError(
                f"expected contiguous {self.dtype} input with shape {self.shape}, "
                f"got {input.dtype} {tuple(input.shape)}"
            )
        if not input.is_cuda or not input.is_contiguous():
            raise ValueError("input must be a contiguous CUDA tensor")
        torch_stream = torch.cuda.current_stream() if stream is None else stream
        if input.device != self.device or torch_stream.device != self.device:
            raise ValueError("input and stream must use the owning device")
        if self.codec == "e4m3_tensor":
            if (
                tensorwise_scales is None
                or tensorwise_scales.dtype != torch.float32
                or tuple(tensorwise_scales.shape) != (self.world_size,)
                or tensorwise_scales.device != self.device
                or not tensorwise_scales.is_contiguous()
            ):
                raise ValueError(
                    "e4m3_tensor requires contiguous FP32 destination scales "
                    f"with shape ({self.world_size},) on {self.device}"
                )
            tensorwise_scales.record_stream(torch_stream)
        elif tensorwise_scales is not None:
            raise ValueError(f"{self.codec} does not accept tensorwise_scales")

        parity = self._epoch % 2
        stream = Stream(torch_stream)
        if self._canonical_pack:
            matrix = input.view(self.shape[1], -1)
            local_payload = self.local_payload_sets[parity]
            local_scale = self.local_scale_sets[parity]
            with torch.cuda.stream(torch_stream):
                if self.codec == "mxfp4":
                    self._pack_local(
                        local_payload,
                        matrix,
                        local_scale,
                        32,
                        shuffle_scale=False,
                    )
                elif self._pack_per_destination:
                    for peer in range(self.world_size):
                        row_start = peer * self.seq_local
                        row_end = row_start + self.seq_local
                        payload_start = peer * self._local_payload_segment_size
                        payload_end = payload_start + self._local_payload_segment_size
                        scale_start = peer * self._local_scale_segment_size
                        scale_end = scale_start + self._local_scale_segment_size
                        self._pack_local(
                            matrix[row_start:row_end],
                            local_payload[payload_start:payload_end],
                            local_scale[scale_start:scale_end],
                        )
                else:
                    self._pack_local(matrix, local_payload, local_scale)
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
        if self._canonical_pack:
            args = (
                self.local_payload_sets[parity].data_ptr(),
                self.local_scale_sets[parity].data_ptr(),
                self.p2p_payload_sets[parity].data_ptr(),
                self.p2p_scale_sets[parity].data_ptr(),
                self.xdb_mem.data_ptr(),
                self.p2p_xdb_mem.data_ptr(),
                self.xdb_flag.data_ptr(),
                stream,
            )
        else:
            args = (
                input.data_ptr(),
                tensorwise_scales.data_ptr() if tensorwise_scales is not None else 0,
                self.p2p_payload_sets[parity].data_ptr(),
                self.p2p_scale_sets[parity].data_ptr(),
                self.xdb_mem.data_ptr(),
                self.p2p_xdb_mem.data_ptr(),
                self.xdb_flag.data_ptr(),
                stream,
            )
        _run_compiled(
            self._launch,
            *(fx.Int64(arg) for arg in args[:-1]),
            args[-1],
        )
        self._epoch += 1
        return self.results[parity]


def math_prod(shape):
    result = 1
    for dim in shape:
        result *= dim
    return result
