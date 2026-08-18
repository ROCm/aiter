# SPDX-License-Identifier: MIT
"""Internal ABI for the two-kernel EP16 MegaMoE Stage-2 pipeline.

This module intentionally contains no public operator arguments.  The public
contract remains identical to MegaMoE v2::

    x_bf16[local_tokens, hidden], routing_weights[local_tokens, topk],
    topk_ids[local_tokens, topk] -> out_bf16[local_tokens, hidden]

The regions below are an implementation detail shared by the fused Stage-1
and Stage-2 kernels. Stage-1 dispatches each token at most once to a
destination rank and writes the aligned proxy's ``node_expected`` scoreboard.
The Stage-2 weighted GMM2 epilogue directly LSA atomic-adds FP32 output into
that proxy's node accumulator.  There is no rank-partial payload and no
eight-rank LSA scan.  The last contributing route publishes node-tile ready;
the kernel returns at most one BF16 node partial per source token through CCO
and writes the final output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _align_up(value: int, alignment: int) -> int:
    value, alignment = int(value), int(alignment)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


@dataclass(frozen=True)
class Stage2NodePartialWire:
    """Fixed-order CCO return format.

    The RAIL connection is rank aligned.  Consequently record ``token`` is
    already destined for the same local rank on the peer node; no source-rank
    or token header is needed.  Four K3 BF16 rows form one 56-KiB aggregate
    request batch.
    """

    hidden: int = 7168
    records_per_group: int = 4

    def __post_init__(self) -> None:
        if self.hidden <= 0 or self.hidden % 256:
            raise ValueError("hidden must be positive and divisible by 256")
        if self.records_per_group <= 0:
            raise ValueError("records_per_group must be positive")
        if self.group_bytes > 64 * 1024:
            raise ValueError("one aggregate group must fit in 64 KiB")

    @property
    def record_bytes(self) -> int:
        return self.hidden * 2

    @property
    def group_bytes(self) -> int:
        return self.records_per_group * self.record_bytes

    def group_count(self, token_count: int) -> int:
        token_count = int(token_count)
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        return (token_count + self.records_per_group - 1) // self.records_per_group


@dataclass(frozen=True)
class Stage2ArenaRegion:
    name: str
    offset: int
    nbytes: int
    alignment: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class Stage2ArenaLayout:
    """Registered-window layout used by one EP rank.

    ``source_nodes`` is two for the MI355 EP16 deployment.  On local rank ``l``
    the two source planes represent global source ranks ``l`` and ``8+l``.

    Weighted GMM2 output never crosses a rank as a route. For a source rank
    ``s``, every expert rank on the current node atomically adds directly to
    local LSA rank ``s % 8`` at ``node_accumulator[s // 8, token]``. Stage-1
    writes the exact number of contributors into ``node_expected``. The last
    route increments ``node_done`` to that value and release-publishes
    ``node_tile_ready``. This is the direct-accumulator form of the MORI
    InterNodeV1 combine hierarchy.

    Payloads are parity buffered.  Absolute-generation readiness protects all
    consumer-visible data, so stale payload bytes are never read and payload
    clearing is unnecessary.
    """

    hidden: int
    topk: int
    max_tokens: int
    world_size: int
    gpus_per_node: int
    source_nodes: int
    tile_n: int
    parity_depth: int
    num_qp: int
    records_per_group: int
    regions: tuple[Stage2ArenaRegion, ...]
    total_bytes: int

    @property
    def hidden_tiles(self) -> int:
        return self.hidden // self.tile_n

    @property
    def return_groups(self) -> int:
        return (self.max_tokens + self.records_per_group - 1) // self.records_per_group

    @classmethod
    def create(
        cls,
        *,
        hidden: int = 7168,
        topk: int = 16,
        max_tokens: int = 128,
        world_size: int = 16,
        gpus_per_node: int = 8,
        source_nodes: int = 2,
        tile_n: int = 256,
        parity_depth: int = 2,
        num_qp: int = 4,
        records_per_group: int = 4,
    ) -> "Stage2ArenaLayout":
        values = {
            "hidden": hidden,
            "topk": topk,
            "max_tokens": max_tokens,
            "world_size": world_size,
            "gpus_per_node": gpus_per_node,
            "source_nodes": source_nodes,
            "tile_n": tile_n,
            "parity_depth": parity_depth,
            "num_qp": num_qp,
            "records_per_group": records_per_group,
        }
        if any(int(v) <= 0 for v in values.values()):
            raise ValueError("all Stage-2 geometry values must be positive")
        if int(hidden) % int(tile_n):
            raise ValueError("hidden must divide tile_n")
        if int(topk) > 32:
            raise ValueError("node_route_mask is a u32 and supports topk <= 32")
        if int(source_nodes) != 2:
            raise ValueError("the first fused CCO Stage-2 supports exactly two nodes")
        if int(world_size) != int(source_nodes) * int(gpus_per_node):
            raise ValueError("world_size must equal source_nodes * gpus_per_node")
        if int(num_qp) not in (1, 2, 4, 8):
            raise ValueError("num_qp must be one of 1,2,4,8")
        wire = Stage2NodePartialWire(int(hidden), int(records_per_group))
        groups = wire.group_count(max_tokens)
        ntiles = int(hidden) // int(tile_n)

        # name, shape, dtype, alignment. node_expected/masks are Stage-1
        # outputs. Stage-2 clears accumulator/done in its resident all-CTA
        # initialization phase before opening the compute gate.
        specs: list[tuple[str, tuple[int, ...], torch.dtype, int]] = [
            (
                "node_dest_rank_mask",
                (parity_depth, source_nodes, max_tokens),
                torch.int32,
                64,
            ),
            (
                "source_token_count",
                (parity_depth, source_nodes),
                torch.int32,
                64,
            ),
            (
                "node_expected",
                (parity_depth, source_nodes, max_tokens, ntiles),
                torch.int32,
                64,
            ),
            (
                "node_accumulator",
                (parity_depth, source_nodes, max_tokens, hidden),
                torch.float32,
                256,
            ),
            (
                "node_done",
                (parity_depth, source_nodes, max_tokens, ntiles),
                torch.int32,
                64,
            ),
            (
                "node_tile_ready",
                (parity_depth, source_nodes, max_tokens, ntiles),
                torch.int64,
                64,
            ),
            (
                "remote_node_tx",
                (parity_depth, max_tokens, hidden),
                torch.bfloat16,
                256,
            ),
            # The peer RAIL rank writes its node partial for this rank's local
            # source tokens directly here, in token order.
            (
                "remote_partial_rx",
                (parity_depth, max_tokens, hidden),
                torch.bfloat16,
                256,
            ),
            (
                "return_group_ready",
                (parity_depth, groups),
                torch.int64,
                64,
            ),
            ("return_count", (parity_depth,), torch.int64, 64),
            ("return_count_ready", (parity_depth,), torch.int64, 64),
            ("return_consumed", (parity_depth,), torch.int64, 64),
            # Stage-1 completion and Stage-2 init are LSA-visible node gates.
            ("stage1_done", (parity_depth,), torch.int64, 64),
            ("stage2_init", (parity_depth,), torch.int64, 64),
            # Persistent launch state. Each work-head shard owns a separate
            # 64-byte line. grid_barrier is monotonic and therefore needs no
            # hot-path memset when the resident grid geometry is unchanged.
            ("grid_barrier", (1,), torch.int64, 64),
            ("gemm_work_head", (8, 16), torch.int32, 64),
            ("final_work_head", (16,), torch.int32, 64),
            ("final_done", (1,), torch.int32, 64),
            # Stage-1 may already have published errors into its own region;
            # Stage-2 keeps a separate counter so its in-kernel reset cannot
            # erase dispatch diagnostics.
            ("stage2_error_count", (1,), torch.int32, 64),
        ]

        offset = 0
        regions: list[Stage2ArenaRegion] = []
        for name, shape, dtype, alignment in specs:
            offset = _align_up(offset, alignment)
            numel = 1
            for dim in shape:
                numel *= int(dim)
            nbytes = numel * torch.empty((), dtype=dtype).element_size()
            regions.append(
                Stage2ArenaRegion(
                    name=name,
                    offset=offset,
                    nbytes=nbytes,
                    alignment=alignment,
                    shape=tuple(int(v) for v in shape),
                    dtype=dtype,
                )
            )
            offset += nbytes

        return cls(
            hidden=int(hidden),
            topk=int(topk),
            max_tokens=int(max_tokens),
            world_size=int(world_size),
            gpus_per_node=int(gpus_per_node),
            source_nodes=int(source_nodes),
            tile_n=int(tile_n),
            parity_depth=int(parity_depth),
            num_qp=int(num_qp),
            records_per_group=int(records_per_group),
            regions=tuple(regions),
            total_bytes=_align_up(offset, 4096),
        )

    def region(self, name: str) -> Stage2ArenaRegion:
        for item in self.regions:
            if item.name == name:
                return item
        raise KeyError(name)

    def offset(self, name: str, *, parity: int | None = None) -> int:
        item = self.region(name)
        off = item.offset
        if parity is not None:
            if not 0 <= int(parity) < self.parity_depth:
                raise ValueError("parity is outside parity_depth")
            if not item.shape or item.shape[0] != self.parity_depth:
                raise ValueError(f"{name} is not parity indexed")
            off += int(parity) * (item.nbytes // self.parity_depth)
        return off

    def allocate_local(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(self.total_bytes, dtype=torch.uint8, device=device)


def validate_public_stage2_contract(
    routing_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    hidden: int = 7168,
    topk: int = 16,
    max_tokens: int = 128,
) -> int:
    """Validate the public-facing portion of a fused Stage-2 call.

    Source-rank, source-token, top-k-slot, route masks, and node partials are
    deliberately absent: they are internal products of fused Stage-1.
    """

    if routing_weights.ndim != 2 or routing_weights.shape[1] != topk:
        raise ValueError("routing_weights must be [local_tokens, topk]")
    local_tokens = int(routing_weights.shape[0])
    if not 0 <= local_tokens <= int(max_tokens):
        raise ValueError("local token count exceeds max_tokens")
    if routing_weights.dtype != torch.float32 or not routing_weights.is_contiguous():
        raise ValueError("routing_weights must be contiguous float32")
    if topk_ids.shape != routing_weights.shape:
        raise ValueError("topk_ids shape must match routing_weights")
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous int32")
    if output.shape != (local_tokens, int(hidden)):
        raise ValueError("output must be [local_tokens, hidden]")
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise ValueError("output must be contiguous bfloat16")
    return local_tokens


__all__ = [
    "Stage2ArenaLayout",
    "Stage2ArenaRegion",
    "Stage2NodePartialWire",
    "validate_public_stage2_contract",
]
