# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The pre-attention monokernel: the self-attention input op under Ulysses parallelism.

One persistent kernel per call computes the fused Q/K/V projection, applies the model's Q/K
prep (``PrepSpec``), quantizes Q, K and V into an Attention A2A profile and sends the
all-to-all as SDMA copies from inside the kernel. Each rank receives the packed MHA-v4
buffers that ``AttentionA2AIntraNodeOp(return_packed=True)`` returns for the same profile
and inputs, byte for byte, so the MHA path is unchanged.

The projection's GEMM hosts the transport as hooks (``kernels/gemm_hooks.py``,
``kernels/pre_attention_monokernel_hooks.py``). ``gemm`` picks the host: the INT8 4-wave
GEMM (``"int8"``) or the MXFP6 A6W6 4-wave GEMM (``"a6w6"``).
"""

import hashlib
import os

import flydsl.expr as fx
import torch
import torch.distributed as dist

from aiter.jit.utils.chip_info import get_cdna4_num_xcds, get_cu_num, get_gfx_runtime

from . import attention_a2a_intranode, pre_attention_monokernel_layout
from .attention_a2a_intranode import PackedRoleResult
from .attention_a2a_profiles import AttentionA2AProfile, attention_a2a_profile
from .kernels import (
    attention_a2a_intranode_kernel,
    gemm_a6w6_4wave_host,
    gemm_a8w8_int8_4wave,
)

# Callers pack the fused weight for gemm="a6w6" in this row order.
from .kernels.gemm_a6w6_4wave_host import host_weight_order  # noqa: F401
from .kernels.gemm_hooks import GemmHooks
from .kernels.qk_prep_epilogue import HEAD_DIM, PrepSpec
from .kernels.tensor_shim import _run_compiled
from .pre_attention_monokernel_layout import role_layouts

# The host GEMMs by name; each host's own block sizes set the session's tile counts.
HOSTS = {"int8": gemm_a8w8_int8_4wave, "a6w6": gemm_a6w6_4wave_host}
GEMMS = tuple(HOSTS)


class _DeviceBytes:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _resolve_profile(profile):
    return attention_a2a_profile(profile) if isinstance(profile, str) else profile


def pre_attention_monokernel_unsupported(
    *, world_size, rows, K, heads, profile, gemm, prep=None, valid_kv_len=None
):
    """Why the monokernel cannot run this configuration, or None when it can.

    ``rows`` are the tokens per rank, ``heads`` the heads of one role before the exchange,
    ``K`` the projection's input width, ``gemm`` the host GEMM. Callers fall back to the
    unfused path on a reason.
    """
    if gemm not in GEMMS:
        return f"gemm must be one of {GEMMS}, got {gemm!r}"
    if get_gfx_runtime() != "gfx950":
        return f"needs gfx950, got {get_gfx_runtime()}"
    try:
        import mori.cco  # noqa: F401
    except ImportError:
        return "needs MORI with CCO"
    try:
        profile = _resolve_profile(profile)
    except ValueError as error:
        return str(error)
    if not isinstance(profile, AttentionA2AProfile):
        return f"profile must be a name or an AttentionA2AProfile, got {profile!r}"
    if prep is not None and not isinstance(prep, PrepSpec):
        return f"prep must be a PrepSpec, got {prep!r}"
    if world_size not in (2, 4, 8):
        return f"world_size must be 2, 4 or 8, got {world_size}"
    if world_size > torch.cuda.device_count():
        return "the Ulysses group must be on one node"
    if heads % world_size:
        return f"{heads} heads do not split over {world_size} ranks"
    host = HOSTS[gemm]
    if heads * HEAD_DIM % host.BLOCK_N:
        return f"heads * {HEAD_DIM} must be a multiple of {host.BLOCK_N}"
    if gemm == "int8" and K % (2 * host.BLOCK_K):
        return f"K must be a multiple of {2 * host.BLOCK_K}, got {K}"
    if K <= 0:
        return f"K must be positive, got {K}"
    if rows <= 0 or rows % profile.pad_multiple:
        return f"rows per rank must be a positive multiple of {profile.pad_multiple}"
    workgroups, xcds = get_cu_num(), get_cdna4_num_xcds()
    if workgroups % xcds:
        return f"{workgroups} CUs do not split over {xcds} XCDs"
    seq_full = rows * world_size
    if valid_kv_len is not None and valid_kv_len != seq_full:
        if not profile.direct_kv_output:
            return f"valid_kv_len needs a profile with direct K/V output, not {profile.name}"
        if not seq_full - rows < valid_kv_len <= seq_full:
            return "valid_kv_len must leave padding only in the last rank's rows"
    codecs = profile.codecs
    if codecs[2] == "mxfp4" and rows % 128:
        return "MXFP4 V needs rows per rank divisible by 128"
    if codecs[2] == "mxfp6_p" and rows % 64:
        return "packed MXFP6 V needs rows per rank divisible by 64"
    try:
        role_layouts(profile, 0, world_size, rows, heads, valid_kv_len)
    except ValueError as error:
        return str(error)
    return None


def _source_key():
    """Digest of the code the hooks are built from, so changed hooks get new binaries."""
    from .kernels import pre_attention_monokernel_hooks, pre_attention_monokernel_quant

    digest = hashlib.sha1()
    for module in (
        pre_attention_monokernel_hooks,
        pre_attention_monokernel_quant,
        pre_attention_monokernel_layout,
        attention_a2a_intranode,
        attention_a2a_intranode_kernel,
    ):
        with open(module.__file__, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()[:8]


class PreAttentionMonokernel:
    """The monokernel on one rank of a Ulysses group, for one shape and profile.

    Owns the MORI CCO communicator and window (each role's receive buffers, double-buffered
    by call parity, and the control words), the cached staging buffer, the aux buffer and
    the projection scratch. Every rank of ``group`` constructs it with the same arguments
    and calls it in the same order on one stream.

    ``rows``: tokens per rank. ``heads``: heads of each role before the exchange, 128
    channels each. ``K``: input width of the projection. ``profile``: an Attention A2A
    profile, by name or ``AttentionA2AProfile``. ``prep``: the Q/K prep, a ``PrepSpec``;
    its ``hadamard`` follows the profile's. ``valid_kv_len``: global tokens before the
    sequence padding, trimming K/V for profiles with direct K/V output. ``gemm``: the host
    GEMM, ``"int8"`` or ``"a6w6"``.
    """

    def __init__(
        self,
        *,
        rank,
        world_size,
        rows,
        K,
        heads,
        profile,
        group,
        gemm,
        prep,
        valid_kv_len=None,
        softmax_scale=None,
        eps=1e-6,
    ):
        from mori.cco import (
            GDA_CONNECTION_NONE,
            CCODevCommRequirements,
            Communicator,
        )

        from .kernels.pre_attention_monokernel_hooks import FACTORS, Plan, make_hooks

        self.profile = _resolve_profile(profile)
        if not isinstance(prep, PrepSpec):
            raise TypeError(f"prep must be a PrepSpec, got {prep!r}")
        self.prep = prep
        reason = pre_attention_monokernel_unsupported(
            world_size=world_size,
            rows=rows,
            K=K,
            heads=heads,
            profile=self.profile,
            prep=self.prep,
            valid_kv_len=valid_kv_len,
            gemm=gemm,
        )
        if reason is not None:
            raise ValueError(f"pre-attention monokernel: {reason}")
        self.gemm = gemm
        if group is None or group.size() != world_size or dist.get_rank(group) != rank:
            raise ValueError(
                "group must be the Ulysses process group, with size world_size and this "
                "process at rank"
            )
        self._factors_at = FACTORS
        self.rank, self.world_size, self.rows, self.K = rank, world_size, rows, K
        self.role_cols = heads * HEAD_DIM
        self.heads_local = heads // world_size
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.layouts = role_layouts(
            self.profile, rank, world_size, rows, heads, valid_kv_len, softmax_scale
        )
        workgroups, xcds = get_cu_num(), get_cdna4_num_xcds()
        self.plan = plan = Plan(self.layouts, world_size, rows, heads, workgroups)
        config = (
            rows,
            K,
            heads,
            self.profile.name,
            self.prep,
            valid_kv_len,
            softmax_scale,
            eps,
            workgroups,
            xcds,
            gemm,
        )
        configs = [None] * world_size
        dist.all_gather_object(configs, config, group=group)
        if any(c != configs[0] for c in configs):
            raise ValueError(
                f"ranks disagree on the monokernel configuration: {configs}"
            )

        uid = [Communicator.get_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(uid, src=dist.get_global_rank(group, 0), group=group)
        # MORI builds the SDMA queues at communicator creation. Enabling SDMA process-wide
        # would also make MORI shmem (the A2A op) take queues and exhaust them.
        sdma_env = os.environ.get("MORI_ENABLE_SDMA")
        os.environ["MORI_ENABLE_SDMA"] = "1"
        try:
            per_rank_vmm = max(1 << 30, -(-plan.window_bytes // (1 << 30)) << 30)
            self.comm = Communicator.init(
                world_size, rank, uid[0], per_rank_vmm=per_rank_vmm
            )
            self.mem = self.comm.alloc_mem(plan.window_bytes)
            self.win = self.comm.register_window(self.mem.ptr, self.mem.size)
            reqs = CCODevCommRequirements()
            reqs.gda_connection_type = GDA_CONNECTION_NONE
            reqs.gda_signal_count = 0
            reqs.gda_counter_count = 0
            reqs.sdma_queue_count = 1
            self.dev_comm = self.comm.create_dev_comm(reqs)
        finally:
            if sdma_env is None:
                os.environ.pop("MORI_ENABLE_SDMA")
            else:
                os.environ["MORI_ENABLE_SDMA"] = sdma_env
        self.bytes = torch.as_tensor(
            _DeviceBytes(self.win.local_ptr, plan.window_bytes), device=self.device
        )
        self.bytes.zero_()

        # Each receive buffer is a tensor of its own, as the A2A op's are: consumers
        # address a packed buffer from the start of its storage. One set per call parity.
        def buffer(offset, nbytes):
            return torch.as_tensor(
                _DeviceBytes(self.win.local_ptr + offset, nbytes), device=self.device
            )

        self._payloads = [
            [
                buffer(
                    p * plan.parity_bytes + plan.recv_payload[role],
                    layout.payload_bytes,
                )
                for role, layout in enumerate(self.layouts)
            ]
            for p in range(2)
        ]
        self._scales = [
            [
                buffer(
                    p * plan.parity_bytes + plan.recv_scale[role], layout.scale_bytes
                )
                for role, layout in enumerate(self.layouts)
            ]
            for p in range(2)
        ]
        self.generation = 0
        self.staging = torch.zeros(
            plan.staging_bytes, dtype=torch.uint8, device=self.device
        )
        self.aux = torch.zeros(
            plan.aux_bytes // 4, dtype=torch.float32, device=self.device
        )
        self.planar = torch.empty(
            3 * rows * self.role_cols, dtype=torch.bfloat16, device=self.device
        )
        self.partials = torch.empty(
            2 * rows * 3 * heads, dtype=torch.float32, device=self.device
        )

        kernels = HOSTS[gemm]
        tiles_per_role = -(-rows // kernels.BLOCK_M) * (
            self.role_cols // kernels.BLOCK_N
        )
        role_hook, first_block_hook, tile_hook = make_hooks(
            rank,
            world_size,
            rows,
            self.role_cols,
            self.layouts,
            plan,
            tiles_per_role,
            workgroups,
            xcds,
            across_heads=self.prep.norm == "rms_across_heads",
            head_dim=HEAD_DIM,
            eps=eps,
        )
        # Everything the hooks were built from that the GEMM's name does not cover.
        key = (
            f"pam{_source_key()}{self.profile.name.replace('-', '')}m{rows}"
            f"k{self.layouts[1].sequence}r{rank}w{world_size}c{workgroups}x{xcds}"
            + (f"e{eps:g}" if eps != 1e-6 else "")
            + (f"s{softmax_scale:g}" if softmax_scale is not None else "")
        )
        host = {
            "xcd_swizzle": 8,
            "num_cu": workgroups,
            "num_xcds": xcds,
            "prep": self.prep,
            "roles": 3,
            "role_cols": self.role_cols,
            "eps": eps,
            "hooks": GemmHooks(role_hook, first_block_hook, tile_hook, key),
            "col_amax_offset": (
                plan.vpc if self.layouts[2].scale == "channel" else None
            ),
        }
        if gemm == "a6w6":
            self.launch = kernels.compile_a6w6_host_4w(
                K=-(-K // kernels.BLOCK_K) * kernels.BLOCK_K, **host
            )
        else:
            self.launch = kernels.compile_int8_gemm_4w(K=K, **host)
        torch.cuda.synchronize()
        self.comm.barrier()

    def __call__(
        self, x_q, x_scale, weight, columns, cos=None, sin=None, weight_scale=None
    ):
        """One call; returns the packed (payload, scale) of Q, K and V.

        INT8 host: ``x_q``: INT8 ``[rows, K]`` activations, ``x_scale``: their FP32 row
        scales. ``weight``: the fused Q/K/V INT8 weight ``[3 * heads * 128, K]``,
        preshuffled with ``shuffle_weight(layout=(16, 16))``.

        A6W6 host: ``x_q`` and ``x_scale``: the activations as ``quant_mxfp6_gemm`` packs
        them. ``weight`` and ``weight_scale``: the fused Q/K/V weight's rows in the host
        layout (``w[host_weight_order(3 * heads * 128)]``), packed the same way.

        ``columns``: FP32 per column of the fused weight: the scales (unused by the A6W6
        host), then the biases (``prep.bias``), then the norm weights (with a norm; V's are
        unused), Q's and K's multiplied by 128 ** -0.5 with ``prep.hadamard``.
        ``cos``/``sin``: FP32 ``[rows, 64]`` per token and rotation pair (with RoPE).

        The results alias this call's half of the receive buffers, which a peer refills two
        calls later; consumers run on this stream before then.
        """
        rows, n = self.rows, 3 * self.role_cols
        prep = self.prep
        column_sets = 1 + prep.bias + (prep.norm != "none")
        rope = prep.rope != "none"
        if self.gemm == "a6w6":
            from aiter.ops.gemm_op_a6w6 import mxfp6_gemm_pack_size

            x_sizes = mxfp6_gemm_pack_size(rows, self.K)
            w_sizes = mxfp6_gemm_pack_size(n, self.K)
            operands = (
                ("x_q", x_q, torch.uint8, x_sizes[0]),
                ("x_scale", x_scale, torch.uint8, x_sizes[1]),
                ("weight", weight, torch.uint8, w_sizes[0]),
                ("weight_scale", weight_scale, torch.uint8, w_sizes[1]),
            )
        else:
            operands = (
                ("x_q", x_q, torch.int8, rows * self.K),
                ("x_scale", x_scale, torch.float32, rows),
                ("weight", weight, torch.int8, n * self.K),
            )
        for name, tensor, dtype, numel in (
            operands + (("columns", columns, torch.float32, column_sets * n),)
        ) + (
            (
                ("cos", cos, torch.float32, rows * 64),
                ("sin", sin, torch.float32, rows * 64),
            )
            if rope
            else ()
        ):
            if (
                tensor is None
                or tensor.dtype != dtype
                or tensor.numel() != numel
                or not tensor.is_contiguous()
                or tensor.device != self.device
            ):
                raise ValueError(
                    f"{name} must be a contiguous {dtype} tensor of {numel} elements on "
                    f"{self.device}"
                )
        if not rope:
            cos = sin = columns
        self.generation += 1
        scales = (x_scale.view(-1),) + (
            (weight_scale.view(-1),) if self.gemm == "a6w6" else ()
        )
        _run_compiled(
            self.launch,
            x_q.view(-1),
            weight.view(-1),
            self.planar,
            *scales,
            columns.view(-1),
            cos.view(-1),
            sin.view(-1),
            self.partials,
            self.aux.data_ptr(),
            self.planar.data_ptr(),
            self.partials.data_ptr(),
            self.staging.data_ptr(),
            self.dev_comm.ptr,
            self.win.handle,
            self.generation,
            rows,
            n,
            fx.Stream(torch.cuda.current_stream()),
        )
        return tuple(PackedRoleResult(self.payload(r), self.scale(r)) for r in range(3))

    def payload(self, role):
        """The role's receive payload of the last call, the A2A op's packed bytes."""
        return self._payloads[self.generation % 2][role]

    def scale_storage(self, role):
        """The role's receive scale storage of the last call, with the consumer padding."""
        return self._scales[self.generation % 2][role]

    def scale(self, role):
        """The role's scale as the A2A op's packed result exposes it: FP32 [1] per tensor,
        FP32 [1, heads_local, 128] per channel, else the E8M0 bytes."""
        layout = self.layouts[role]
        view = self.scale_storage(role)[: layout.scale_view_bytes]
        if layout.scale == "tensor":
            return view.view(torch.float32)
        if layout.scale == "channel":
            return view.view(torch.float32).view(1, self.heads_local, HEAD_DIM)
        return view

    def projection(self, role):
        """The role's BF16 projection of the last call, after the prep, ``[rows, heads * 128]``."""
        return self.planar.view(3, self.rows, self.role_cols)[role]

    def factors(self, role):
        """The RMS factors of Q (0) or K (1) of the last call, FP32 [rows], for a norm
        across heads."""
        start = self._factors_at // 4 + role * self.rows
        return self.aux[start : start + self.rows]

    def close(self):
        torch.cuda.synchronize()
        self.comm.barrier()
        self.dev_comm.close()
        self.comm.destroy()
