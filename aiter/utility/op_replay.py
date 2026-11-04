# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Replay recorded calls of aiter's attention ops -- whose cost follows
per-sequence lengths a shape does not carry -- and of its collectives (under
torchrun, one process per rank), and time them, for
`aiter/utility/op_ab.py` (``driver: replay``) as a tuner's ``--run_config``
serves the GEMM families.

Each CSV row (ATOM's ``tools/op_record/export.py`` writes them) is one call:
``op``, the call's recorded signature (``sig``: each argument's
shape/dtype/stride, or its value), ``work`` -- the query and context length
of each sequence it served, zeros for a CUDA graph's padded rows -- and
``count``. A replayer rebuilds every argument at its recorded shape, dtype
and stride, then the ones the lengths decide: caches holding exactly the
blocks the lengths need, block tables scattered over them as a running
server's are, length and offset tensors, persistent-kernel work plans. It
times the production entry with `aiter.test_common.run_perftest`.

    python aiter/utility/op_replay.py ROWS.csv
"""

import csv
import inspect
import itertools
import json
import math
import sys
from collections.abc import Callable

import torch


def _is_tensor(spec) -> bool:
    return isinstance(spec, dict) and "shape" in spec


def _as_dtype(name):
    found = getattr(torch, str(name).removeprefix("torch."), None)
    return found if isinstance(found, torch.dtype) else None


def tensor(
    spec: dict, *, first: int | None = None, fill: str = "randn"
) -> torch.Tensor:
    """A tensor of the recorded shape, dtype and stride -- a view of a buffer
    just big enough -- with ``first`` replacing the leading dimension.
    ``fill``: "randn" (scaled down), "pos" (small positive, for scales),
    "zeros"; integer and bool tensors are zeros."""
    shape = list(spec["shape"])
    stride = list(spec.get("stride") or [])
    if first is not None:
        shape[0] = first
    if len(stride) != len(shape):
        stride = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    size = 0 if 0 in shape else 1 + sum((d - 1) * s for d, s in zip(shape, stride))
    dtype = _as_dtype(spec["dtype"])
    if fill == "zeros" or not dtype.is_floating_point:
        base = torch.zeros(size, dtype=dtype, device="cuda")
    elif fill == "pos":
        base = (torch.rand(size, device="cuda") * 0.1 + 0.01).to(dtype)
    else:
        base = (torch.randn(size, device="cuda") * 0.5).to(dtype)
    return base.as_strided(shape, stride)


def value(arg):
    """A recorded argument rebuilt: a tensor as `tensor`, a dtype's name as
    the dtype, an object recorded only by its type name as None."""
    if _is_tensor(arg):
        return tensor(arg)
    if isinstance(arg, list):
        return tuple(value(a) for a in arg)
    if isinstance(arg, str):
        if arg.startswith("<") and arg.endswith(">"):
            return None
        dtype = _as_dtype(arg)
        return dtype if dtype is not None else arg
    return arg


class Call:
    """A recorded call bound to ``fn``'s parameter names: ``recorded`` the
    arguments as recorded, ``values`` what the replay passes, by name."""

    def __init__(
        self,
        fn: Callable,
        sig: dict,
        signature_of: Callable | None = None,
        names: list[str] | None = None,
    ):
        self.fn = fn
        names = names or list(inspect.signature(signature_of or fn).parameters)
        self.positional = names[: len(sig["args"])]
        self.keywords = list(sig.get("kwargs", {}))
        self.recorded = dict(zip(names, sig["args"])) | sig.get("kwargs", {})
        self.values = {n: value(a) for n, a in self.recorded.items()}

    def has(self, name: str) -> bool:
        return self.recorded.get(name) is not None

    def args(self) -> tuple[Callable, tuple, dict]:
        """(fn, args, kwargs) as recorded: the dispatcher drops trailing
        arguments left at their defaults, so only what was recorded is
        passed and the op defaults the rest."""
        return (
            self.fn,
            tuple(self.values[n] for n in self.positional),
            {n: self.values[n] for n in self.keywords},
        )


def cu_seqlens(lengths: list[int]) -> torch.Tensor:
    offsets = [0, *itertools.accumulate(lengths)]
    return torch.tensor(offsets, dtype=torch.int32, device="cuda")


def lengths_tensor(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(lengths, dtype=torch.int32, device="cuda")


def paged(lengths: list[int], block: int, width: int):
    """(blocks, tables, per-sequence block ids) for ``lengths``: as many cache
    blocks as the lengths need, each sequence's a random scatter over them."""
    needed = [math.ceil(n / block) for n in lengths]
    blocks = max(1, sum(needed))
    order = torch.randperm(blocks, dtype=torch.int32)
    tables = torch.zeros(len(lengths), width, dtype=torch.int32)
    ids, start = [], 0
    for row, n in enumerate(needed):
        tables[row, :n] = order[start : start + n]
        ids.append(order[start : start + n])
        start += n
    return blocks, tables.cuda(), ids


def _cache_block(spec: dict) -> int:
    """The page size of a paged cache: [blocks, heads, d/x, page, x] (or
    [blocks, page, heads, d])."""
    shape = spec["shape"]
    return shape[3] if len(shape) == 5 else shape[1]


def _scale(call: Call, name: str, blocks: int) -> None:
    """A per-token cache scale lives per block: as many as the cache has."""
    if call.has(name):
        spec = call.recorded[name]
        first = blocks if len(spec["shape"]) >= 2 else None
        call.values[name] = tensor(spec, first=first, fill="pos")


def _caches(call: Call, k: str, v: str, table: str, lengths: list[int]) -> int:
    """Paged caches and a block table for ``lengths``; the cache's blocks."""
    block = _cache_block(call.recorded[k])
    width = call.recorded[table]["shape"][1]
    if max(lengths, default=0) > width * block:
        raise ValueError(f"context {max(lengths)} exceeds the recorded block table")
    blocks, tables, _ = paged(lengths, block, width)
    call.values[k] = tensor(call.recorded[k], first=blocks)
    call.values[v] = tensor(call.recorded[v], first=blocks)
    call.values[table] = tables
    return blocks


def _descales(call: Call, *names: str) -> None:
    for name in names:
        if call.has(name):
            call.values[name] = tensor(call.recorded[name], fill="pos")


# ------------------------------------------------------------------ replayers


def pa_decode_gluon(sig: dict, work: dict):
    from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon as py

    call = Call(torch.ops.aiter.pa_decode_gluon, sig, signature_of=py)
    blocks = _caches(call, "key_cache", "value_cache", "block_tables", work["ctx"])
    call.values["context_lengths"] = lengths_tensor(work["ctx"])
    call.values["output"] = torch.empty_like(call.values["output"])
    for name in ("key_scale", "value_scale"):
        _scale(call, name, blocks)
    return call.args()


def pa_fwd_asm(sig: dict, work: dict):
    import aiter

    call = Call(aiter.pa_fwd_asm, sig)
    blocks = _caches(call, "K", "V", "block_tables", work["ctx"])
    call.values["context_lens"] = lengths_tensor(work["ctx"])
    if call.has("qo_indptr"):
        call.values["qo_indptr"] = cu_seqlens(work["q"])
    for name in ("K_QScale", "V_QScale"):
        _scale(call, name, blocks)
    return call.args()


def pa_persistent_fwd(sig: dict, work: dict):
    """The persistent kernel at ``work``, its work plan computed beforehand
    by `aiter.get_pa_metadata_v1` as ATOM's metadata builder does outside the
    graph, from the same inputs."""
    import aiter

    call = Call(aiter.pa_persistent_fwd, sig)
    ctx = work["ctx"]
    block = _cache_block(call.recorded["K"])
    width = max(1, max(math.ceil(n / block) for n in ctx))
    blocks, _, ids = paged(ctx, block, width)
    call.values["K"] = tensor(call.recorded["K"], first=blocks)
    call.values["V"] = tensor(call.recorded["V"], first=blocks)
    for name in ("K_QScale", "V_QScale"):
        _scale(call, name, blocks)
    qo_indptr = cu_seqlens(work["q"])
    kv_indptr = cu_seqlens([len(b) for b in ids])
    kv_indices = torch.cat([*ids, torch.zeros(1, dtype=torch.int32)]).cuda()
    context_lens = lengths_tensor(ctx)
    q_heads = call.recorded["Q"]["shape"][1]
    kv_heads = call.recorded["K"]["shape"][1]
    buffers = [
        torch.empty(size, dtype=dtype, device="cuda")
        for size, dtype in aiter.get_pa_metadata_info_v1(len(ctx), kv_heads)
    ]
    max_qlen = int(call.values["max_qlen"])
    aiter.get_pa_metadata_v1(
        qo_indptr,
        kv_indptr,
        context_lens,
        q_heads // kv_heads,
        kv_heads,
        True,
        *buffers,
        kv_granularity=max(block, 16),
        block_size=block,
        max_seqlen_qo=max_qlen,
        uni_seqlen_qo=max_qlen,
        fast_mode=True,
        max_split_per_batch=-1,
    )
    _, work_indptr, work_info, reduce_indptr, reduce_final, reduce_partial = buffers
    call.values |= {
        "output": torch.empty_like(call.values["output"]),
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "context_lens": context_lens,
        "work_indptr": work_indptr,
        "work_info": work_info,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final,
        "reduce_partial_map": reduce_partial,
    }
    return call.args()


def unified_attention(sig: dict, work: dict):
    from aiter.ops.triton.attention.unified_attention import unified_attention as fn

    call = Call(fn, sig)
    _caches(call, "k", "v", "block_table", work["ctx"])
    call.values["cu_seqlens_q"] = cu_seqlens(work["q"])
    call.values["seqused_k"] = lengths_tensor(work["ctx"])
    call.values["out"] = torch.empty_like(call.values["out"])
    _descales(call, "q_descale", "k_descale", "v_descale")
    return call.args()


def _varlen(call: Call, work: dict):
    """Dense prefill attention: cu_seqlens from the served lengths, which
    must account for every recorded query and key row."""
    q_rows = call.recorded["q"]["shape"][0]
    k_rows = call.recorded["k"]["shape"][0]
    if sum(work["q"]) != q_rows or sum(work["ctx"]) != k_rows:
        raise ValueError(
            f"served lengths ({sum(work['q'])} q, {sum(work['ctx'])} k) do not "
            f"cover the recorded rows ({q_rows} q, {k_rows} k)"
        )
    call.values["cu_seqlens_q"] = cu_seqlens(work["q"])
    call.values["cu_seqlens_k"] = cu_seqlens(work["ctx"])
    _descales(call, "q_descale", "k_descale", "v_descale")
    if call.has("out"):
        call.values["out"] = torch.empty_like(call.values["out"])
    return call.args()


def flash_attn_varlen_func(sig: dict, work: dict):
    import aiter

    return _varlen(Call(aiter.flash_attn_varlen_func, sig), work)


def _flash_attn_varlen_forward(sig: dict, work: dict):
    from aiter.ops.mha import _flash_attn_varlen_forward as fn

    return _varlen(Call(fn, sig), work)


# ------------------------------------------------------------------ collectives
#
# Run under torchrun, one process per rank of the recorded TP group, on the
# GPUs it ran on: a collective's time is its rank-slowest, since every rank
# waits for the last. Each rank calls exactly as often as every other (no
# input rotation), or the group would hang.


def distributed() -> tuple[int, int]:
    """(rank, world) of a TP group the size of torchrun's world, set up as
    ATOM sets it up: `aiter.init_dist_env`, custom all-reduce on."""
    import os

    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        raise RuntimeError("a collective replays across ranks: launch it with torchrun")
    if not dist.is_initialized():
        import aiter

        rank = int(os.environ["RANK"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        aiter.init_dist_env(
            world,
            rankID=rank,
            backend="nccl",
            distributed_init_method="env://",
            local_rank=local,
        )
    return dist.get_rank(), world


# The group_name ops' parameters: `torch_compile_guard` hides the signature.
_ALL_REDUCE = ["tensor", "group_name", "ca_use_new", "ca_fp8_quant", "prefill_support"]
_AR_RMSNORM = [
    "inp",
    "res_inp",
    "w",
    "eps",
    "group_name",
    "prefill_support",
    "x_pad_to_multiple",
    "gemma_norm",
]
_AR_RMSNORM_QUANT = [
    "inp",
    "res_inp",
    "w",
    "eps",
    "group_name",
    "group_size",
    "prefill_support",
    "transpose_scale",
    "gemma_norm",
]


def _group_op(name: str, params: list[str]):
    def build(sig: dict, work: dict):
        from aiter.dist import parallel_state

        distributed()
        call = Call(getattr(parallel_state, name), sig, names=params)
        # The recorded run's TP group, as this process names it.
        call.values["group_name"] = parallel_state.get_tp_group().unique_name
        return call.args()

    return build


def tensor_model_parallel_all_gather(sig: dict, work: dict):
    from aiter.dist.communication_op import tensor_model_parallel_all_gather as fn

    distributed()
    return Call(fn, sig).args()


COLLECTIVES = {
    "all_reduce_": _group_op("all_reduce_", _ALL_REDUCE),
    "fused_allreduce_rmsnorm_": _group_op("fused_allreduce_rmsnorm_", _AR_RMSNORM),
    "fused_allreduce_rmsnorm_quant_per_group_": _group_op(
        "fused_allreduce_rmsnorm_quant_per_group_", _AR_RMSNORM_QUANT
    ),
    "fused_allreduce_rmsnorm_quant_per_group_emit_bf16_": _group_op(
        "fused_allreduce_rmsnorm_quant_per_group_emit_bf16_", _AR_RMSNORM_QUANT
    ),
    "tensor_model_parallel_all_gather": tensor_model_parallel_all_gather,
}


# Calls one graph replays back to back: alone, a graph of one few-microsecond
# collective times each rank's launch skew as much as the collective (A/A
# rounds of a 64x4096 all-reduce read 8 to 16 us); queued, the ranks run in
# step and the skew is paid once per replay.
GRAPH_REPEATS = 8


def graphed(fn: Callable, args: tuple, kwargs: dict) -> Callable[[], None]:
    """``fn(*args, **kwargs)`` recorded `GRAPH_REPEATS` times into a CUDA
    graph inside aiter's `graph_capture`, as ATOM records a decode step's
    collectives: custom all-reduce then runs on buffers registered at
    capture, not the copy-in path an eager call takes. Returns the graph's
    replay, which keeps the arguments alive; a replay is `GRAPH_REPEATS`
    calls."""
    from aiter.dist.parallel_state import graph_capture

    fn(*args, **kwargs)  # lazy setup happens outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as context, torch.cuda.graph(graph, stream=context.stream):
        for _ in range(GRAPH_REPEATS):
            fn(*args, **kwargs)
    torch.cuda.synchronize()

    def replay_graph(_keep=(args, kwargs)):
        graph.replay()

    return replay_graph


def _across_ranks(value: float, op: str) -> float:
    """``value`` reduced over the ranks (MIN or MAX); itself off torchrun."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=getattr(dist.ReduceOp, op))
    return float(t.item())


REPLAYERS: dict[str, Callable[[dict, dict], tuple[Callable, tuple, dict]]] = {
    **COLLECTIVES,
    "pa_decode_gluon": pa_decode_gluon,
    "pa_fwd_asm": pa_fwd_asm,
    "pa_persistent_fwd": pa_persistent_fwd,
    "unified_attention": unified_attention,
    "flash_attn_varlen_func": flash_attn_varlen_func,
    "_flash_attn_varlen_forward": _flash_attn_varlen_forward,
}


def replay(
    rows_csv: str,
    extra_args: list[str] | None = None,
    replayers: dict | None = None,
) -> tuple[list, list]:
    """(rows, results) for `op_ab.py`: each row's key columns, and its time as a
    tuner's ``run_config`` reports one (``e2e_us``, ``status``)."""
    import aiter  # noqa: F401  registers torch.ops.aiter.*
    from aiter.test_common import run_perftest

    replayers = REPLAYERS if replayers is None else replayers
    iters, warmup = 101, 5
    extra = list(extra_args or [])
    for flag in ("--iters", "--warmup"):
        if flag in extra:
            v = int(extra[extra.index(flag) + 1])
            iters, warmup = (v, warmup) if flag == "--iters" else (iters, v)

    with open(rows_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    results = []
    for row in rows:
        label = row.get("label", row["op"])
        collective = row["op"] in COLLECTIVES
        built, error = None, None
        try:
            built = replayers[row["op"]](
                json.loads(row["sig"]), json.loads(row["work"])
            )
        except Exception as e:  # noqa: BLE001  one row's failure is that row's result
            error = f"error:{e}"
        # A rank that cannot build the call must not leave the others waiting
        # in it: every rank times the row or none does.
        if collective and _across_ranks(error is None, "MIN") < 1:
            error = error or "error:another rank could not build the call"
        if error is not None:
            results.append({"shape": label, "e2e_us": -1, "status": error})
            continue
        fn, args, kwargs = built
        try:
            per_call = 1
            if collective and json.loads(row["work"]).get("graph"):
                fn, args, kwargs = graphed(fn, args, kwargs), (), {}
                per_call = GRAPH_REPEATS
            _, us = run_perftest(
                fn,
                *args,
                num_iters=iters,
                num_warmup=warmup,
                num_rotate_args=1 if collective else 0,
                **kwargs,
            )
            us /= per_call
            if collective:
                us = _across_ranks(us, "MAX")
            status = "ok" if math.isfinite(us) and us > 0 else f"error:time {us}"
            results.append({"shape": label, "e2e_us": us, "status": status})
        except Exception as e:  # noqa: BLE001  one row's failure is that row's result
            results.append({"shape": label, "e2e_us": -1, "status": f"error:{e}"})
        finally:
            torch.cuda.empty_cache()
    return rows, results


if __name__ == "__main__":
    for row, result in zip(*replay(sys.argv[1], sys.argv[2:])):
        print(
            f"{result['e2e_us']:10.2f} us  {result['status']:<10} {row.get('label', '')}"
        )
