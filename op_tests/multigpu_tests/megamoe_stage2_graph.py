# SPDX-License-Identifier: MIT
"""Full-forward graph replay and Stage2 attribution for the breakdown driver.

Formal runner settings are warmup=10/iters=40/tail=20; shorter invocations are
correctness smokes, not comparable performance baselines. Stage2 is attributed
from each rank's GPU trace, and full-forward events are measured separately.
See scripts/megamoe_tile/FUSED_STAGE2_DESIGN_20260909.md for the exact contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel import MoriFusedMoeBaselinePath
from scripts.megamoe_tile.stage2_graph_inputs import (
    IMMUTABLE_INPUTS, PUBLIC_INPUTS, WEIGHTS, aggregate_input_identity,
    changed_inputs, fingerprint_shared_inputs, performance_exclusions, route_histogram,
)
from scripts.megamoe_tile.stage2_graph_protocol import validate_protocol_snapshot
from scripts.megamoe_tile.summarize_stage2_graph import aggregate, parse_trace


def _check(reference, actual, threshold, label):
    expected, output = reference.float(), actual.float()
    finite = bool(torch.isfinite(expected).all().item() and torch.isfinite(output).all().item())
    expected_norm = float(expected.norm().item())
    output_norm = float(output.norm().item())
    value = float((output - expected).norm().item() / max(expected_norm, 1e-12)) if finite else float("inf")
    local_value = value
    error = torch.tensor(value, dtype=torch.float64)
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    value = float(error.item())
    if value >= threshold:
        print("MEGAMOE_STAGE2_GRAPH_NUMERICAL_FAILURE " + json.dumps({
            "rank": dist.get_rank(), "label": label, "finite": finite,
            "local_rel_l2": local_value, "reference_norm": expected_norm,
            "output_norm": output_norm,
            "cosine": float((output * expected).sum().item() / max(expected_norm * output_norm, 1e-12)),
        }), flush=True)
        raise AssertionError(f"{label}: rank-max relL2={value}, threshold={threshold}")
    return {"label": label, "rank_max_rel_l2": value, "threshold": threshold}


def _device_epoch(operator):
    from aiter.ops.flydsl.kernels.megamoe_tile.window_view import read_window_u64

    base = int(operator._runtime.window.local_ptr)
    return int(read_window_u64(base + operator.stage1_layout.offset("epoch_gate"), 1)[0])


def _check_protocol(operator, expected_generation, label):
    import os
    if os.environ.get("MEGAMOE_TWO_KERNEL") == "1":
        # 两 kernel 路径用 plane_slot_inbox 取代了 rank_push_inbox,这套协议的
        # 区域没人写。数值判据仍是 candidate_or_mori_vs_mori 的 relL2。
        return {"rank": dist.get_rank(), "label": label, "error": None,
                "state": "skipped: two-kernel stage2 replaces the rank-push protocol"}
    local = {"rank": dist.get_rank(), "label": label, "error": None}
    try:
        local["state"] = validate_protocol_snapshot(
            operator.debug_direct_tile_snapshot(),
            expected_generation=expected_generation,
            readiness=operator.stage2_ready_granularity,
            tokens=operator.mtpr,
            accumulation_mode=operator.stage2_rank_accumulation_mode,
        )
    except (KeyError, TypeError, ValueError) as error:
        local["error"] = str(error)
    all_ranks = [None] * dist.get_world_size()
    dist.all_gather_object(all_ranks, local)
    errors = [row for row in all_ranks if row["error"]]
    if errors:
        raise AssertionError(f"Graph protocol validation failed: {errors}")
    return local


def _check_input_integrity(shared, original, fields, label):
    changed = changed_inputs(original, fingerprint_shared_inputs(shared, fields))
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, {"rank": dist.get_rank(), "changed": changed})
    failures = [row for row in ranks if row["changed"]]
    if failures:
        raise AssertionError(f"{label}: input contents changed: {failures}")


def run_graph_profile(path, shared, shape, args, contract, rank, world, device):
    if not 1 <= args.tail_iters <= args.iters or args.warmup < 1:
        raise ValueError("graph profiling needs warmup >= 1 and 1 <= tail_iters <= iters")
    if args.direct_packed_weights:
        raise ValueError("graph performance requires random packed weights; constant weights are diagnostic only")
    if os.environ.get("MEGAMOE_TILE_PROFILE_REGIONS", "0") == "1":
        raise ValueError("torch.profiler graph mode cannot use external ROCTx profiler pause/resume")
    candidate = args.path == "candidate"
    if args.graph_coalesce_reference_duplicates and not candidate:
        raise ValueError("coalesced reference is only supported for candidate correctness checks")
    # Record actual constructed data before either path overwrites the mutable
    # reference quantization buffers. Full-forward inputs are BF16 x and original
    # TopK/weights; a_quant/a_scale are only initial reference buffers. Their
    # post-check values can legitimately reflect the changed-input eager oracle.
    initial_inputs = fingerprint_shared_inputs(shared)
    properties = torch.cuda.get_device_properties(device)
    local_inputs = {
        "rank": rank, "tensors": initial_inputs,
        "routes": route_histogram(shared.topk_ids.cpu().tolist(), rank=rank, shape=shape.__dict__),
        "device": {"name": properties.name, "cu_count": properties.multi_processor_count,
                   "architecture": getattr(properties, "gcnArchName", None)},
    }
    input_rows = [None] * world
    dist.all_gather_object(input_rows, local_inputs)
    input_identity = aggregate_input_identity(input_rows, shape=shape.__dict__)
    trace_dir = Path(args.torch_profiler_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"rank{rank}.inputs.json").write_text(json.dumps(local_inputs, indent=2) + "\n")
    if args.route_pattern == "cross_node" and not input_identity["eplb"]:
        raise AssertionError("cross_node inputs do not satisfy the measured EPLB workload")
    if candidate:
        # The public wrapper consumes contiguous byte views of these exact
        # weights. Check aliases so the recorded shared weights are also the
        # actual candidate weights, not an independently prepared copy.
        for name in WEIGHTS:
            actual, expected = getattr(path.operator, "_" + name), getattr(shared.prepared_weights, name)
            if actual.data_ptr() != expected.data_ptr() or actual.numel() * actual.element_size() != expected.numel() * expected.element_size():
                raise AssertionError(f"candidate {name} is not the recorded shared weight storage")
    # 诊断:只跑 fused_stage1(wrapper 同开关跳过 stage2)。不建 MORI 参照、不做正确性检查,
    # 只 eager 1 次 + 预热 3 次 + capture 一个 graph 再 replay N 次,给 ATT 之类的采集用。
    if candidate and os.environ.get("MEGAMOE_TK_S1_ONLY", "0") != "0":
        _n = int(os.environ.get("MEGAMOE_TK_S1_ONLY_REPLAYS", "40"))
        # graph 里只留 k1+k2:bf16->fp4 量化在 graph 外做一次,以 fp4 输入直接进通信。
        _op = path.operator
        _xq = torch.empty((shape.tokens, shape.hidden // 2), dtype=torch.uint8, device=device)
        _xs = torch.empty((shape.tokens, shape.hidden // 32), dtype=torch.uint8, device=device)
        _op._s1_quant_launch(shared.x, _xq, _xs, shape.tokens, _op._s1_quant_grid,
                             stream=_op._flydsl_stream(None))
        torch.cuda.synchronize(device)
        s1_call = lambda: _op.forward(_xq, shared.route_weights, shared.topk_ids, x_scale=_xs)
        s1_call()
        torch.cuda.synchronize(device)
        dist.barrier()
        _st = torch.cuda.Stream(device=device)
        _st.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(_st):
            for _ in range(3):
                s1_call()
        _st.synchronize()
        dist.barrier()
        _g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(_g, stream=_st):
            s1_call()
        dist.barrier()
        _ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(_n)]
        for a, b in _ev:
            a.record()
            _g.replay()
            b.record()
        torch.cuda.synchronize(device)
        _us = sorted(a.elapsed_time(b) * 1000.0 for a, b in _ev)
        if os.environ.get("MEGAMOE_TK_S1_HWID", "0") != "0" and rank == int(os.environ.get("ATT_RANK", "0")):
            from aiter.ops.flydsl.kernels.megamoe_tile.window_view import read_window_u32 as _r32h
            _wb = int(_op._runtime.window.local_ptr) + int(_op.layout.stage2_offset) \
                + int(_op.stage2_layout.region("node_accumulator").offset)
            _nw = int(_op.stage1_worker_blocks)
            _raw = [int(v) & 0xFFFFFFFF for v in _r32h(_wb, 64 * _nw * 4)]
            _tab = {}
            for _slot in range(64):
                for _t in range(_nw):
                    g, bid, hw, xcc = _raw[(_slot * _nw + _t) * 4:(_slot * _nw + _t) * 4 + 4]
                    if g and (g & 63) == _slot:
                        _tab.setdefault(str(g), []).append([_t, bid, hw, xcc])
            _out = os.path.join(os.environ.get("ATT_OUT", "/tmp"), "hwid_rank%d.json" % rank)
            with open(_out, "w") as _f:
                json.dump({"worker_blocks": _nw, "epoch": _device_epoch(_op), "gens": _tab}, _f)
            print("MEGAMOE_S1_HWID wrote %s gens=%s" % (_out, sorted(int(k) for k in _tab)[:3] + ["..."]), flush=True)
        if getattr(_op, "_s1_ts_buf", None) is not None:
            # MEGAMOE_TK_S1_TSTAMP:最后一次 replay 的 k1 打点(s_memrealtime,100MHz)。
            # 每 CTA(ticket)32 个 int64 槽,槽位见 stage1.py _DIAG_TS。
            _ts = _op._s1_ts_buf.cpu().tolist()
            _dir = os.environ.get("MEGAMOE_TK_S1_TS_OUT", "/tmp")
            os.makedirs(_dir, exist_ok=True)
            with open(os.path.join(_dir, "ts_rank%d.json" % rank), "w") as _f:
                json.dump({"rank": rank, "epoch": _device_epoch(_op),
                           "worker_blocks": int(_op.stage1_worker_blocks), "ts": _ts}, _f)
            _e = [r[12] - r[1] for r in _ts if r[1] and r[12]]
            print("MEGAMOE_S1_TS " + json.dumps({
                "rank": rank, "gen": _ts[0][0], "ctas": len(_e),
                "span_us_max": round((max(r[12] for r in _ts if r[12]) - min(r[1] for r in _ts if r[1])) / 100.0, 1)}), flush=True)
        print("MEGAMOE_S1_ONLY " + json.dumps({
            "rank": rank, "replays": _n, "epoch": _device_epoch(path.operator),
            "min_us": round(_us[0], 1), "median_us": round(_us[len(_us) // 2], 1),
            "max_us": round(_us[-1], 1)}), flush=True)
        dist.barrier()
        return
    reference_path = MoriFusedMoeBaselinePath(
        shape, shared, rank, world,
        valid_recv=shape.tokens * world // 2,
        combine_quant_type="none",
        block_num=args.mori_block_num,
        rdma_block_num=args.mori_rdma_block_num,
        exact_capacity=True,
    ) if candidate else path.baseline
    if reference_path.shared is not shared:
        raise AssertionError("MORI must consume the same recorded SharedInputs")
    def reference_forward(*, debug_sync=False):
        return reference_path._forward(
            debug_sync=debug_sync,
            reference_coalesce_duplicates=args.graph_coalesce_reference_duplicates,
        )

    reference = reference_forward(debug_sync=args.graph_debug_reference)[:shape.tokens].clone()
    torch.cuda.synchronize(device)
    dist.barrier()
    # 诊断变体(非对比口径):fp4 输入直接进通信。量化在 graph 外,所以每次
    # 就地改 shared.x 之后都要重新量化,否则 replay 读到的是旧的 fp4。
    fp4_input = candidate and os.environ.get("MEGAMOE_TK_BENCH_FP4_INPUT", "0") != "0"
    def requant():
        if fp4_input:
            op = path.operator
            op._s1_quant_launch(shared.x, fp4_x, fp4_scale, shape.tokens,
                                op._s1_quant_grid, stream=op._flydsl_stream(None))
    if fp4_input:
        fp4_x = torch.empty((shape.tokens, shape.hidden // 2), dtype=torch.uint8, device=device)
        fp4_scale = torch.empty((shape.tokens, shape.hidden // 32), dtype=torch.uint8, device=device)
        requant()
        call = lambda: path.operator.forward(fp4_x, shared.route_weights, shared.topk_ids, x_scale=fp4_scale)
    elif candidate:
        call = lambda: path.operator.forward(shared.x, shared.route_weights, shared.topk_ids)
    else:
        call = path.baseline._forward
    eager = call()[:shape.tokens].clone()
    torch.cuda.synchronize(device)
    checks = [_check(reference, eager, args.graph_rel_l2_threshold, "candidate_or_mori_vs_mori")]
    # Explicit static buffers avoid Dynamo mutation/input/output copies. The
    # graph contains one full forward; the replay loop stays outside capture.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        graph_output = call()
    # Capture enqueues no execution. Verify both parities across replays and
    # refresh input contents without changing the captured pointer addresses.
    start_epoch = _device_epoch(path.operator) if candidate else None
    graph.replay()
    torch.cuda.synchronize(device)
    checks.append(_check(eager, graph_output[:shape.tokens], 1e-2, "first_replay_vs_eager"))
    original = shared.x.clone()
    shared.x.mul_(-0.75)
    requant()
    updated_reference = reference_forward()[:shape.tokens].clone()
    torch.cuda.synchronize(device)
    dist.barrier()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize(device)
    checks.append(_check(updated_reference, graph_output[:shape.tokens], args.graph_rel_l2_threshold, "changed_input_after_five_replays"))
    shared.x.copy_(original)
    requant()
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize(device)
    checks.append(_check(eager, graph_output[:shape.tokens], 1e-2, "restored_input_after_four_replays"))
    protocol_checks = []
    if candidate:
        # Ten correctness replays above: 1 original + 5 changed + 4 restored.
        # Check actual device epochs and BOTH named stage counters; a missing
        # snapshot field must fail rather than silently becoming error_count=0.
        protocol_checks.append(_check_protocol(path.operator, start_epoch + 10, "after_input_replays"))
    correctness_replays = 10
    if args.graph_check_routing_replay:
        original_ids = shared.topk_ids.clone()
        # Swapping the two expert nodes preserves duplicate-slot relations and each
        # token's per-rank route capacity. For local-only fixtures this changes
        # every node mask from present to absent, exercising stale BF16 payload
        # clearing within the same arena. Four replays reuse both parities.
        try:
            shared.topk_ids.copy_((original_ids + shape.experts // 2) % shape.experts)
            swapped_reference = reference_forward()[:shape.tokens].clone()
            torch.cuda.synchronize(device)
            dist.barrier()
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize(device)
            correctness_replays += 4
            checks.append(_check(swapped_reference, graph_output[:shape.tokens],
                                 args.graph_rel_l2_threshold, "swapped_expert_nodes_after_four_replays"))
            if candidate:
                protocol_checks.append(_check_protocol(path.operator, start_epoch + correctness_replays,
                                                       "after_swapped_routing_replays"))
        finally:
            shared.topk_ids.copy_(original_ids)
        for _ in range(4):
            graph.replay()
        torch.cuda.synchronize(device)
        correctness_replays += 4
        checks.append(_check(eager, graph_output[:shape.tokens], 1e-2,
                             "restored_routing_after_four_replays"))
        if candidate:
            protocol_checks.append(_check_protocol(path.operator, start_epoch + correctness_replays,
                                                   "after_restored_routing_replays"))
    if rank == 0:
        print("MEGAMOE_STAGE2_GRAPH_CORRECTNESS " + json.dumps(checks), flush=True)
    # 诊断:stage1/stage2 在 graph 下是否对齐。每次 replay 前在 A/B 两份输入间切换,
    # 只 replay 一次就与两份 MORI 参照各比一次。对齐时每次都贴近本次输入的参照;
    # stage2 若读到上一代 stage1 输出,就会隔一次贴近上一个输入的参照。
    _alt_n = int(os.environ.get("MEGAMOE_TK_DIAG_ALT", "0") or 0)
    if candidate and _alt_n > 0:
        _orig_x = shared.x.clone()
        _alt = []
        for i in range(_alt_n):
            if i % 2 == 0:
                shared.x.copy_(_orig_x)
            else:
                shared.x.copy_(_orig_x * -0.75)
            requant()
            torch.cuda.synchronize(device)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(device)
            _out = graph_output[:shape.tokens]
            _vs_a = _check(reference, _out, 1e9, "alt_vs_A")["rank_max_rel_l2"]
            _vs_b = _check(updated_reference, _out, 1e9, "alt_vs_B")["rank_max_rel_l2"]
            _alt.append(["A" if i % 2 == 0 else "B", round(_vs_a, 6), round(_vs_b, 6),
                         _device_epoch(path.operator)])
        shared.x.copy_(_orig_x)
        requant()
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize(device)
        if rank == 0:
            print("MEGAMOE_ALT_CHECK " + json.dumps(_alt), flush=True)
        dist.barrier()
    # 诊断:stage1 级精度。fused_stage1 的 h1(GMM1+situv2 输出,fp4+e8m0)按
    # (源 token, topk slot, 专家) 解出来,与 torch 参照 situv2(deq(x_q[src]) @ deq(w1[e])^T)
    # 逐行比。MORI dispatch 是逐字节搬运,所以参照等价于「MORI dispatch + 理想 gemm1」。
    # 多次 replay:系统误差所有 replay 一样;竞态只落在个别 replay 的个别行。
    _s1ref_n = int(os.environ.get("MEGAMOE_TK_DIAG_S1_REF", "0") or 0)
    if candidate and _s1ref_n > 0:
        from aiter.ops.flydsl.kernels.megamoe_tile.window_view import (
            read_window_u32 as _r32, read_window_u64 as _r64, window_tensor as _wt,
        )
        from aiter.ops.flydsl.kernels.megamoe_tile.activation import apply_gate_up
        _op = path.operator
        _base = int(_op._runtime.window.local_ptr)
        _l1 = _op.stage1_layout
        H, I, epr, mtpr = shape.hidden, shape.inter, shape.experts // world, int(_op.mtpr)
        _lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=device)

        def _deq(q, s):
            # q: [..., K/2] uint8(低 nibble 在前),s: [..., K/32] e8m0
            q = q.view(torch.uint8)
            v = torch.stack([_lut[(q & 0xF).long()], _lut[(q >> 4).long()]], dim=-1).flatten(-2)
            sc = torch.exp2(s.view(torch.uint8).float() - 127.0)
            return (v.view(*v.shape[:-1], -1, 32) * sc.unsqueeze(-1)).flatten(-2)
        # 本 rank 未 shuffle 的 w1:直接从算子实际用的 shuffled 权重反推。
        # shuffle 是纯排列:把下标拆成 3 个字节平面各过一次 shuffle,拼回来就是排列表。
        from aiter.ops.shuffle import shuffle_weight as _shw
        from aiter.utility.fp4_utils import e8m0_shuffle as _e8s
        from aiter.utility import dtypes as _dt
        _pw = shared.prepared_weights

        def _perm_of(shape, fn):
            n = 1
            for d in shape:
                n *= d
            idx = torch.arange(n, dtype=torch.int64, device=device)
            perm = torch.zeros(n, dtype=torch.int64, device=device)
            for k in range(3):
                plane = ((idx >> (8 * k)) & 0xFF).to(torch.uint8).view(shape)
                perm |= fn(plane).view(torch.uint8).flatten().to(torch.int64) << (8 * k)
            return perm
        _pq = _perm_of((1, 2 * I, H // 2), lambda t: _shw(t.view(_dt.fp4x2), layout=(16, 16)))
        _ps = _perm_of((2 * I, H // 32), lambda t: _e8s(t.view(_dt.fp8_e8m0)))
        _wq_real = _pw.w1.view(torch.uint8).reshape(epr, -1)
        _ws_real = _pw.w1_scale.view(torch.uint8).reshape(epr, -1)
        _w1q, _w1s = [], []
        for e in range(epr):
            q = torch.empty_like(_wq_real[e]); q[_pq] = _wq_real[e]
            sc = torch.empty_like(_ws_real[e]); sc[_ps] = _ws_real[e]
            _w1q.append(q.view(2 * I, H // 2)); _w1s.append(sc.view(2 * I, H // 32))
        # 全体 rank 的已量化输入(就是 k1 读的那份 buffer)与路由
        xq_src = fp4_x if fp4_input else _op._s1_quant_x
        xs_src = fp4_scale if fp4_input else _op._s1_quant_scale
        # quant kernel 在 graph 里:replay 一次让 buffer 装着当前输入的量化结果
        graph.replay()
        torch.cuda.synchronize(device)
        xq_loc = xq_src[:shape.tokens].view(torch.uint8).contiguous()
        xs_loc = xs_src[:shape.tokens].view(torch.uint8)[:, :H // 32].contiguous()
        ids_loc = shared.topk_ids[:shape.tokens].to(torch.int32).contiguous()

        def _gather(t):
            # 进程组是 gloo:走 CPU 列表版 all_gather
            parts = [torch.empty_like(t.cpu()) for _ in range(world)]
            dist.all_gather(parts, t.cpu())
            return torch.stack(parts).to(device)
        xq_all, xs_all, ids_all = _gather(xq_loc), _gather(xs_loc), _gather(ids_loc)
        _gate_first = None
        _ref_cache = {}
        # 自检:反推的 w1 再 shuffle 一次必须逐字节还原成算子实际用的权重
        _w1_ours = _shw(torch.stack(_w1q).view(_dt.fp4x2), layout=(16, 16)).view(torch.uint8).flatten()
        _w1s_ours = _e8s(torch.cat(_w1s, 0).view(_dt.fp8_e8m0)).view(torch.uint8).flatten()
        _selfchk = {
            "perm_unique": [int(torch.unique(_pq).numel() == _pq.numel()), int(torch.unique(_ps).numel() == _ps.numel())],
            "w1_eq_frac": float((_w1_ours == _wq_real.flatten()).float().mean()),
            "w1s_eq_frac": float((_w1s_ours == _ws_real.flatten()).float().mean()),
        }
        del _w1_ours, _w1s_ours, _pq, _ps
        _gen = int(_r64(_base + _l1.offset("epoch_gate"), 1)[0]); _par = _gen & 1
        _nv = int(_r32(_base + _l1.offset("num_valid", parity=_par), 1)[0])
        _rin = _wt(_base + _l1.offset("tile_row_input", parity=_par), _nv, torch.int32).to(torch.int64)
        _rsrc = _wt(_base + _l1.offset("tile_row_source", parity=_par), _nv, torch.int32).to(torch.int64) & 0xFFFFFF
        _gin = _wt(_base + _l1.offset("grouped_input_q", parity=_par),
                   _l1.source_capacity * (H // 2), torch.uint8).view(_l1.source_capacity, H // 2)
        _ok = (_rsrc < world * mtpr) & (_rin >= 0) & (_rin < _l1.source_capacity)
        _a = _gin[_rin[_ok]]
        _b = xq_all[_rsrc[_ok] // mtpr, _rsrc[_ok] % mtpr]
        _selfchk["x_rows"] = int(_ok.sum())
        _selfchk["x_rows_equal"] = int((_a == _b).all(dim=1).sum())
        _selfchk["xq_all_shape"] = list(xq_all.shape)
        _selfchk["xs_loc_shape"] = list(xs_src.shape)
        print("MEGAMOE_S1REF_SELFCHK " + json.dumps({"rank": rank, **_selfchk}), flush=True)

        def _decode():
            gen = int(_r64(_base + _l1.offset("epoch_gate"), 1)[0])
            par = gen & 1
            nv = int(_r32(_base + _l1.offset("num_valid", parity=par), 1)[0])
            rows = ((nv + 31) // 32) * 32
            src = _wt(_base + _l1.offset("tile_row_source_sorted", parity=par), rows, torch.int32).to(torch.int64).clone()
            tex = _wt(_base + _l1.offset("tile_expert_sorted", parity=par), rows // 32, torch.int32).to(torch.int64).clone()
            hq = _wt(_base + _l1.offset("h1_output_q", parity=par), rows * I // 2, torch.uint8).view(rows, I // 2).clone()
            hsr = _wt(_base + _l1.offset("h1_output_scale", parity=par), rows * I // 32, torch.uint8).clone()
            r = torch.arange(rows, device=device).view(-1, 1)
            s = torch.arange(I // 32, device=device).view(1, -1)
            phys, rit = r // 32, r % 32
            if os.environ.get("MEGAMOE_TK_H1_PHYS", "0") == "1":
                # h1 按物理 tile 存放:排序后第 t 个 tile 的数据在物理 tile inv[t]。
                inv = _wt(_base + _l1.offset("tile_src_of_dst", parity=par), rows // 32, torch.int32).to(torch.int64).clone()
                phys = inv[r // 32]
                hq = hq[(inv.view(-1, 1) * 32 + torch.arange(32, device=device).view(1, -1)).flatten()]
            nb, wg = s // 4, s % 4
            dword = phys * ((I // 256) * 64) + (nb // 2) * 64 + wg * 16 + rit % 16
            hs = hsr[dword * 4 + (nb % 2) * 2 + rit // 16]
            return gen, nv, src, tex.repeat_interleave(32), hq, hs

        def _reference(key_src, key_slot, lexp):
            # 按专家分组算参照,缓存(同输入下每次 replay 相同)
            out = torch.empty((key_src.numel(), 2 * I), dtype=torch.float32, device=device)
            for e in torch.unique(lexp).tolist():
                m = lexp == e
                srank, stok = key_src[m] // mtpr, key_src[m] % mtpr
                xa = _deq(xq_all[srank, stok], xs_all[srank, stok])
                w = _deq(_w1q[e], _w1s[e])
                y = xa @ w.t()
                out[m] = torch.cat([y[:, :I], y[:, I:]], dim=1)
            return out
        stats = []
        prev = None
        base_h1 = None
        for it in range(_s1ref_n):
            graph.replay()
            torch.cuda.synchronize(device)
            gen, nv, src, ex, hq, hs = _decode()
            _o = graph_output[:shape.tokens]
            if it == 0:
                _out0 = _o.clone()
            _out_rows_diff = int((_o != _out0).any(dim=1).sum())
            low = src & 0xFFFFFF
            valid = (low < world * mtpr)
            valid[nv:] = False
            idx = torch.nonzero(valid).flatten()
            ks, kslot, le = low[idx], src[idx] >> 24, ex[idx]
            # 映射自检:源 token 的 topk_ids[slot] 必须等于本 rank 的这个专家
            gexp = ids_all[ks // mtpr, ks % mtpr, kslot.clamp(0, shape.topk - 1)].to(torch.int64)
            map_bad = int((gexp != rank * epr + le).sum())
            ck = (ks << 8) | kslot
            order = torch.argsort(ck)
            ck, ks, kslot, le, idx = ck[order], ks[order], kslot[order], le[order], idx[order]
            ckey = tuple(ck[:4].tolist()) + (int(ck.numel()),)
            if ckey not in _ref_cache:
                gu = _reference(ks, kslot, le)
                _ref_cache.clear()
                _ref_cache[ckey] = (ck.clone(), gu)
            ck0, gu = _ref_cache[ckey]
            same_keys = bool(torch.equal(ck0, ck))
            h1 = _deq(hq[idx], hs[idx])
            if _gate_first is None:
                a = apply_gate_up(gu[:, :I], gu[:, I:], "situv2", situ_beta=1.0, situ_linear_beta=1.0)
                b = apply_gate_up(gu[:, I:], gu[:, :I], "situv2", situ_beta=1.0, situ_linear_beta=1.0)
                ea = float((h1 - a).norm() / a.norm()); eb = float((h1 - b).norm() / b.norm())
                _gate_first = ea <= eb
                gate_errs = [round(ea, 5), round(eb, 5)]
            ref = (apply_gate_up(gu[:, :I], gu[:, I:], "situv2", situ_beta=1.0, situ_linear_beta=1.0)
                   if _gate_first else
                   apply_gate_up(gu[:, I:], gu[:, :I], "situv2", situ_beta=1.0, situ_linear_beta=1.0))
            rel = (h1 - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-20)
            bad = torch.nonzero(rel > 0.5).flatten()
            rec = {"it": it, "gen": gen, "nv": nv, "out_rows_diff": _out_rows_diff, "rows": int(idx.numel()), "map_bad": map_bad,
                   "same_keys": same_keys,
                   "rel_all": round(float((h1 - ref).norm() / ref.norm()), 6),
                   "rel_med": round(float(rel.median()), 5), "rel_p999": round(float(rel.quantile(0.999)), 5),
                   "rel_max": round(float(rel.max()), 5), "n_bad": int(bad.numel())}
            if prev is not None and torch.equal(prev[0], ck):
                _cur = torch.cat([hq[idx], hs[idx]], 1)
                _chg = (prev[1] != _cur).any(1)
                rec["rows_changed_vs_prev"] = int(_chg.sum())
                # 相对「正常基线」(第 1 次 replay)变了的行:在哪、错成什么样
                _dv = (base_h1[1] != _cur).any(1) if base_h1 is not None else _chg
                if int(_dv.sum()):
                    ci = torch.nonzero(_dv).flatten()
                    qd = (base_h1[1][ci, :I // 2] != _cur[ci, :I // 2])
                    sd = (base_h1[1][ci, I // 2:] != _cur[ci, I // 2:])
                    phys = idx[ci]
                    rec["chg"] = {
                        "rows": int(ci.numel()),
                        "phys_rows": sorted(set(phys.tolist()))[:64],
                        "tiles": sorted(set((phys // 32).tolist())),
                        "tile128": sorted(set((phys // 128).tolist())),
                        "lexp": sorted(set(le[ci].tolist())),
                        "srcs": ks[ci][:16].tolist(), "slots": kslot[ci][:16].tolist(),
                        "rel": [round(float(v), 4) for v in rel[ci][:32]],
                        "rel_base": [round(float(v), 4) for v in base_h1[2][ci][:8]],
                        "q_bytes_diff_per_row": qd.sum(1)[:32].tolist(),
                        "s_bytes_diff_per_row": sd.sum(1)[:32].tolist(),
                        # 列上哪里坏:q 按 128 列(64 字节)一块、scale 按 4 个一块(=128 列)
                        "q_colblk_hist": qd.view(ci.numel(), -1, 64).any(2).sum(0).tolist(),
                        "s_colblk_hist": sd.view(ci.numel(), -1, 4).any(2).sum(0).tolist(),
                        # 坏值是不是全 0 / 与另一行相同
                        "cur_q_zero_rows": int((_cur[ci, :I // 2] == 0).all(1).sum()),
                    }
            if bad.numel():
                det = []
                for b_ in bad[:12].tolist():
                    phys_row = int(idx[b_])
                    # 坏行最像哪个参照行(同专家内):是不是别的 token 的结果
                    m = (le == le[b_])
                    cand = torch.nonzero(m).flatten()
                    d = (h1[b_].unsqueeze(0) - ref[cand]).norm(dim=1) / ref[cand].norm(dim=1)
                    j = int(cand[int(d.argmin())])
                    det.append({"row": phys_row, "tile": phys_row // 32, "rit": phys_row % 32,
                                "src": int(ks[b_]), "slot": int(kslot[b_]), "lexp": int(le[b_]),
                                "rel": round(float(rel[b_]), 4),
                                "best_src": int(ks[j]), "best_slot": int(kslot[j]),
                                "best_rel": round(float(d.min()), 4),
                                "h1_absmax": round(float(h1[b_].abs().max()), 4),
                                "q_zero": bool((hq[phys_row] == 0).all()),
                                "s_vals": sorted(set(hs[phys_row].tolist()))[:6]})
                rec["bad_detail"] = det
                rec["bad_tiles"] = sorted(set((int(idx[b_]) // 32) for b_ in bad.tolist()))[:32]
            stats.append(rec)
            prev = (ck.clone(), torch.cat([hq[idx], hs[idx]], 1).clone())
            if base_h1 is None:
                base_h1 = (ck.clone(), prev[1], rel.clone())
        _s1ref = {"rank": rank, "gate_first": _gate_first,
                  "gate_errs": gate_errs, "replays": stats}
        # 多 rank 同时 print 会在 run.log 里交错成解析不了的行;每 rank 另写一个文件。
        _s1ref_dir = os.environ.get("MEGAMOE_TK_S1REF_OUT", "/tmp/s1ref")
        os.makedirs(_s1ref_dir, exist_ok=True)
        with open(os.path.join(_s1ref_dir, "s1ref_rank%d.json" % rank), "w") as _f:
            json.dump(_s1ref, _f)
        print("MEGAMOE_S1REF " + json.dumps(_s1ref), flush=True)
        del _w1q, _w1s
        torch.cuda.empty_cache()
        dist.barrier()
    # 诊断:同一份输入连续 replay,逐行比输出。确定性的实现下每一行都应逐位相同,
    # 不一致的 token 行及其路由(目的 rank / 是否跨节点)就是竞态的落点。
    _diag_n = int(os.environ.get("MEGAMOE_TK_DIAG_REPLAY_DIFF", "0") or 0)
    if candidate and _diag_n > 1:
        outs = []
        snaps = []
        from aiter.ops.flydsl.kernels.megamoe_tile.window_view import (
            read_window_u32 as _r32, read_window_u64 as _r64,
        )
        _op = path.operator
        _base = int(_op._runtime.window.local_ptr)
        _l1 = _op.stage1_layout

        def _s1_counters():
            gen = int(_r64(_base + _l1.offset("epoch_gate"), 1)[0])
            par = gen & 1
            p = lambda name, parity=True: _base + int(_l1.offset(name, parity=par if parity else None))
            return {
                "gen": gen,
                "tile_alloc": int(_r32(p("tile_alloc"), 1)[0]),
                "queue_tail": int(_r32(p("h1_queue_tail"), 1)[0]),
                "compute_done": int(_r32(p("h1_compute_done"), 1)[0]),
                "num_valid": int(_r32(p("num_valid"), 1)[0]),
                "expert_count": [int(v) for v in _r32(p("expert_count"), 2 * _l1.local_experts)],
                "error_count": int(_r32(p("error_count", False), 1)[0]),
                # sealer 收齐 EOS 那一刻:[generation, Σtile_row_done, Σexpert_count]
                "eos_check": [int(v) for v in _r64(p("plan_debug") + 100 * 8, 3)],
            }
        from aiter.ops.flydsl.kernels.megamoe_tile.window_view import window_tensor as _wt
        _gen0 = torch.Generator(device="cpu").manual_seed(1234)
        _hid8 = shape.hidden // 2 // 8
        _int8 = shape.inter // 2 // 8
        _wq = torch.randint(1, 2**61, (_hid8,), generator=_gen0, dtype=torch.int64).to(device)
        _wh = torch.randint(1, 2**61, (_int8,), generator=_gen0, dtype=torch.int64).to(device)

        def _row_hashes():
            # 置换不变:每行一个 hash,排序后按多重集合比。物理行位置每代不同,
            # 内容对就应与第 1 次 replay 的多重集合完全相同。
            gen = int(_r64(_base + _l1.offset("epoch_gate"), 1)[0])
            par = gen & 1
            nv = int(_r32(_base + _l1.offset("num_valid", parity=par), 1)[0])
            rin = _wt(_base + _l1.offset("tile_row_input", parity=par), nv, torch.int32).to(torch.int64)
            gin = _wt(_base + _l1.offset("grouped_input_q", parity=par),
                      _l1.source_capacity * _hid8, torch.int64).view(_l1.source_capacity, _hid8)
            h1 = _wt(_base + _l1.offset("h1_output_q", parity=par), nv * _int8, torch.int64).view(nv, _int8)
            hq = (gin[rin.clamp(0, _l1.source_capacity - 1)] * _wq).sum(dim=1)
            hh = (h1 * _wh).sum(dim=1)
            # scale 按 route 行存(fanout 逐字节 swizzle 写),按 8 字节字的多重集合比。
            sc = _wt(_base + _l1.offset("grouped_input_scale", parity=par),
                     nv * (shape.hidden // 32) // 8, torch.int64)
            # 行 -> (源 key, 专家, payload) 的配对:任何一项配错都会改变这个组合 hash。
            src = _wt(_base + _l1.offset("tile_row_source", parity=par), nv, torch.int32).to(torch.int64)
            tex = _wt(_base + _l1.offset("tile_expert", parity=par), nv // 32, torch.int32).to(torch.int64)
            ex = tex.repeat_interleave(32)
            pair = hq * 1000003 + src * 7919 + ex
            return (torch.sort(hq).values, torch.sort(hh).values,
                    torch.sort(sc).values, torch.sort(pair).values,
                    torch.sort(tex).values)
        rowh = []
        # 对照:不走 graph,每次真 forward。host 侧 _generation 先对齐 device 的
        # epoch_gate —— stage2 的 generation/parity 是 host 整数,graph 里被冻结。
        _eager = os.environ.get("MEGAMOE_TK_DIAG_EAGER", "0") != "0"
        for _ in range(_diag_n):
            if _eager:
                torch.cuda.synchronize(device)
                _op._generation = int(_r64(_base + _l1.offset("epoch_gate"), 1)[0])
                dist.barrier()
                _o = call()
                torch.cuda.synchronize(device)
                outs.append(_o[:shape.tokens].clone())
            else:
                graph.replay()
                torch.cuda.synchronize(device)
                outs.append(graph_output[:shape.tokens].clone())
            snaps.append(_s1_counters())
            rowh.append(_row_hashes())
        epr = shape.experts // world
        ids = shared.topk_ids[:shape.tokens].to(torch.int64)
        dest = ids // epr
        my_node = rank // 8
        per_replay = []
        bad_rows = torch.zeros(shape.tokens, dtype=torch.bool, device=device)
        for i in range(1, _diag_n):
            row_bad = (outs[i] != outs[0]).any(dim=1)
            bad_rows |= row_bad
            per_replay.append(int(row_bad.sum()))
        # 以「与多数 replay 不同」近似出错的那次:对每个坏行统计它有几个不同取值。
        bad = torch.nonzero(bad_rows).flatten()
        info = {
            "rank": rank,
            "replays": _diag_n,
            "rows_differ_vs_first": per_replay,
            "bad_rows": int(bad.numel()),
        }
        if bad.numel():
            d = dest[bad]
            remote = (d // 8) != my_node
            info["bad_tokens_head"] = bad[:24].tolist()
            info["bad_has_remote_route"] = int(remote.any(dim=1).sum())
            info["bad_all_local_routes"] = int((~remote).all(dim=1).sum())
            all_remote = (dest // 8) != my_node
            info["all_tokens_has_remote_route"] = int(all_remote.any(dim=1).sum())
            # 每个坏行里哪些列不同(按 256 列的 ntile 归并),看是整行坏还是局部坏。
            cols = torch.zeros(shape.hidden // 256, dtype=torch.int64, device=device)
            for i in range(1, _diag_n):
                diff = (outs[i] != outs[0])[bad]
                cols += diff.view(bad.numel(), -1, 256).any(dim=2).sum(dim=0)
            info["bad_ntile_hist"] = cols.tolist()
            maxabs = max(float((outs[i] - outs[0]).abs().max()) for i in range(1, _diag_n))
            info["max_abs_diff"] = maxabs
            info["ref_absmax"] = float(outs[0].abs().max())
            # 坏 token 的目的 rank 直方图 vs 全体 token 的目的 rank 直方图
            info["bad_dest_rank_hist"] = torch.bincount(d.flatten().clamp(0, world - 1), minlength=world).tolist()
            info["all_dest_rank_hist"] = torch.bincount(dest.flatten().clamp(0, world - 1), minlength=world).tolist()
        # 每次 replay 的计数与第 1 次比:只列出不同的字段(expert_count 列出差异的专家)。
        snap_diff = []
        for i in range(1, _diag_n):
            d = {}
            for key in ("tile_alloc", "queue_tail", "compute_done", "num_valid", "error_count"):
                if snaps[i][key] != snaps[0][key]:
                    d[key] = [snaps[0][key], snaps[i][key]]
            ec = [(e, snaps[0]["expert_count"][e], snaps[i]["expert_count"][e])
                  for e in range(len(snaps[0]["expert_count"]))
                  if snaps[i]["expert_count"][e] != snaps[0]["expert_count"][e]]
            if ec:
                d["expert_count"] = ec[:8]
            snap_diff.append(d)
        info["snap0"] = {k: v for k, v in snaps[0].items() if k != "expert_count"}
        info["snap0_expert_count_sum"] = sum(snaps[0]["expert_count"])
        info["snap_diff_vs_first"] = snap_diff
        info["gens"] = [sn["gen"] for sn in snaps]
        info["eos_check"] = [sn["eos_check"] for sn in snaps]
        info["eos_short"] = [
            (i, sn["eos_check"][1], sn["eos_check"][2])
            for i, sn in enumerate(snaps)
            if sn["eos_check"][0] == sn["gen"] and sn["eos_check"][1] < sn["eos_check"][2]
        ]
        # 与第 1 次 replay 的多重集合比:不在第 1 次集合里的行数(输入行 / GMM1 输出行)
        def _missing(a, b):
            if a.numel() != b.numel():
                return -1
            return int((~torch.isin(a, b)).sum())
        info["input_rows_diff"] = [_missing(rowh[i][0], rowh[0][0]) for i in range(1, _diag_n)]
        info["h1_rows_diff"] = [_missing(rowh[i][1], rowh[0][1]) for i in range(1, _diag_n)]
        info["scale_words_diff"] = [_missing(rowh[i][2], rowh[0][2]) for i in range(1, _diag_n)]
        info["pair_diff"] = [_missing(rowh[i][3], rowh[0][3]) for i in range(1, _diag_n)]
        info["tile_expert_diff"] = [_missing(rowh[i][4], rowh[0][4]) for i in range(1, _diag_n)]
        print("MEGAMOE_REPLAY_DIFF " + json.dumps(info), flush=True)
        dist.barrier()
    _check_input_integrity(shared, initial_inputs, PUBLIC_INPUTS, "before timing")
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier()
    # A separate unprofiled event batch cross-checks full-forward graph timing.
    # Events are read only after all replays; no per-iteration host/rank sync.
    event_pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.iters)]
    for start, end in event_pairs:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize(device)
    event_us = [start.elapsed_time(end) * 1000 for start, end in event_pairs]
    dist.barrier()
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for iteration in range(args.iters):
            with record_function(f"stage2_graph_{args.path}_tpr{shape.tokens}_iter{iteration}"):
                graph.replay()
        torch.cuda.synchronize(device)
    trace_file = trace_dir / f"rank{rank}.json"
    prof.export_chrome_trace(str(trace_file))
    checks.append(_check(eager, graph_output[:shape.tokens], 1e-2, "final_replay_vs_eager"))
    if candidate:
        # Include the unprofiled event batch as well as the profiler batch.
        # Protocol errors during timing must not escape the earlier input check.
        expected_epoch = start_epoch + correctness_replays + args.warmup + 2 * args.iters
        protocol_checks.append(_check_protocol(path.operator, expected_epoch, "after_all_replays"))
    # Recheck all immutable inputs, including complete weights, after timing.
    # Hashing never adds transfers or GPU work to a captured/profiled forward.
    _check_input_integrity(shared, initial_inputs, IMMUTABLE_INPUTS, "after timing")
    local = {"rank": rank, "trace": str(trace_file), "checks": checks, "protocol_checks": protocol_checks, "event_full_forward_us": event_us, "parse_error": None}
    if args.device_timeline:
        from op_tests.multigpu_tests.bench_megamoe_tile_ep16_stage2_breakdown import TIMELINE_INTERVALS

        local["final_replay_timeline"] = path.operator.debug_device_timeline()
        ticks = local["final_replay_timeline"]["ticks"]
        scale = 1000.0 / path._wall_clock_rate_khz
        local["final_replay_intervals_us"] = {
            name + "_us": (ticks[end] - ticks[begin]) * scale
            for name, (begin, end) in TIMELINE_INTERVALS.items()
            if ticks[begin] > 0 and ticks[end] > 0
        }
        if args.device_timeline_history_depth:
            # Read only after BOTH timing batches have finished. Device stores
            # use generation & (depth-1), so the captured graph still has exactly
            # two kernels and the replay loop gets no copy, barrier, or sync.
            # Keep event-batch history too: long tails must not be explained
            # solely by a profiler artifact if they also occur without profiling.
            first_generation = expected_epoch - 2 * args.iters + 1
            history = []
            for offset in range(2 * args.iters):
                snapshot = path.operator.debug_device_timeline(first_generation + offset)
                ticks = snapshot["ticks"]
                history.append({
                    **snapshot,
                    "batch": "event" if offset < args.iters else "profiler",
                    "iteration": offset % args.iters,
                    "intervals_us": {
                        name + "_us": (ticks[end] - ticks[begin]) * scale
                        for name, (begin, end) in TIMELINE_INTERVALS.items()
                        if ticks[begin] > 0 and ticks[end] > 0
                    },
                })
            local["replay_timelines"] = history
    try:
        local["samples"] = parse_trace(json.loads(trace_file.read_text()), path=args.path, tokens=shape.tokens, iterations=args.iters, tail=args.tail_iters)
    except (ValueError, KeyError) as error:
        local["parse_error"] = str(error)
    (trace_dir / f"rank{rank}.samples.json").write_text(json.dumps(local, indent=2) + "\n")
    all_ranks = [None] * world
    dist.all_gather_object(all_ranks, local)
    errors = {row["rank"]: row["parse_error"] for row in all_ranks if row["parse_error"]}
    if errors:
        raise RuntimeError(f"incomplete profiler evidence; preserve traces and rerun: {errors}")
    summary = aggregate(all_ranks, expected_world=world)
    root = Path(__file__).resolve().parents[2]
    source_files = [
        "aiter/ops/flydsl/kernels/megamoe_tile/stage1.py",
        "aiter/ops/flydsl/kernels/megamoe_tile/stage2.py",
        "aiter/ops/flydsl/kernels/megamoe_tile/mega_moe_tile_a4w4.py",
        "op_tests/multigpu_tests/bench_megamoe_tile_ep16_stage2_breakdown.py",
        "op_tests/multigpu_tests/bench_megamoe_tile_ep16_two_kernel.py",
        "op_tests/multigpu_tests/megamoe_stage2_graph.py",
        "scripts/megamoe_tile/summarize_stage2_graph.py",
        "scripts/megamoe_tile/stage2_graph_window.py",
        "scripts/megamoe_tile/stage2_graph_protocol.py",
        "scripts/megamoe_tile/stage2_graph_reference.py",
        "scripts/megamoe_tile/stage2_graph_inputs.py",
        "scripts/megamoe_tile/compare_stage2_graph.py",
        "op_tests/multigpu_tests/bench_mega_moe_v2.py",
        "aiter/ops/flydsl/kernels/megamoe_tile/stage1_abi.py",
        "aiter/ops/flydsl/kernels/megamoe_tile/stage2_abi.py",
        "aiter/ops/flydsl/kernels/megamoe_tile/rank_push_layout.py",
        "op_tests/multigpu_tests/bench_megamoe_tile_ep16_dual_path.py",
    ]
    tune = os.environ.get("AITER_CONFIG_FMOE")
    result = {
        **contract,
        "measurement": "one_full_forward_per_explicit_cuda_graph_replay",
        "input_identity": input_identity,
        "input_integrity_passed": True,
        "graph_debug_reference": args.graph_debug_reference,
        "graph_coalesce_reference_duplicates": args.graph_coalesce_reference_duplicates,
        "reference_route_semantics": (
            "sum duplicate expert weights; zero-weight same-rank filler experts; eager reference only"
            if args.graph_coalesce_reference_duplicates else "original routing"
        ),
        "diagnostic_final_replay_intervals_us": {row["rank"]: row["final_replay_intervals_us"] for row in all_ranks} if args.device_timeline else None,
        "diagnostic_timeline_history": {
            "depth": args.device_timeline_history_depth,
            "records_per_rank": 2 * args.iters,
            "batches": ["event", "profiler"],
            "artifact": "node*/rank*.samples.json: replay_timelines",
            "readback": "after both batches; no per-replay copies or synchronization",
        } if args.device_timeline_history_depth else None,
        "candidate_stage1_input": "fresh fused Stage1 every replay",
        "candidate_stage1_timing": "included in graph; Stage2 extracted from GPU trace",
        "statistic": {
            "pooled_min": "minimum actual Stage2 span across all ranks and selected tail replays (EPLB)",
            "rank_mean_of_min": "per-rank minimum actual Stage2 span over tail replays, then all-rank mean",
            "rank_mean": "per-rank mean actual Stage2 span over tail replays, then all-rank mean",
        }[args.graph_comparison_statistic],
        "comparison_metric": "timing.metrics.stage2_span_us." + args.graph_comparison_statistic,
        "graph_comparison_statistic": args.graph_comparison_statistic,
        "warmup": args.warmup, "iterations": args.iters, "tail_iterations": args.tail_iters,
        "graph_check_routing_replay": args.graph_check_routing_replay,
        "correctness_replays": correctness_replays,
        "graph_contains_output_clone": False,
        "graph_input_quantization": "BF16-to-FP4 included on both paths",
        "activation_parameters": {"beta": 1.0, "linear_beta": 1.0} if shape.activation == "situv2" else None,
        "mori_blocks": args.mori_block_num, "mori_rdma_blocks": args.mori_rdma_block_num,
        "environment": {key: os.environ.get(key) for key in ("MORI_RDMA_DEVICES", "MORI_NUM_QP_PER_PE", "MORI_DEVICE_NIC", "MORI_IB_GID_INDEX", "AITER_SITUV2_A4W4", "AITER_CONFIG_FMOE", "AMD_SERIALIZE_KERNEL")},
        "torch_version": torch.__version__,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files},
        "tune_sha256": hashlib.sha256(Path(tune).read_bytes()).hexdigest() if tune else None,
        "timing": summary,
        "full_forward_event_rank_mean_us": sum(sum(row["event_full_forward_us"][-args.tail_iters:]) / args.tail_iters for row in all_ranks) / world,
        "correctness": checks,
        "protocol_checks": {row["rank"]: row["protocol_checks"] for row in all_ranks},
    }
    result["performance_exclusions"] = performance_exclusions(result, input_identity)
    result["performance_eligible"] = not result["performance_exclusions"]
    if rank == 0:
        (trace_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print("MEGAMOE_EP16_STAGE2_GRAPH_RESULT " + json.dumps(result, sort_keys=True), flush=True)
