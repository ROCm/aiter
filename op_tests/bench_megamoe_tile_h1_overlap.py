# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import time

import torch


def _time_us(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--activation", choices=("silu", "swiglu", "situv2"), default="silu"
    )
    parser.add_argument("--swiglu-limit", type=float, default=None)
    parser.add_argument("--situ-beta", type=float, default=4.0)
    parser.add_argument("--situ-linear-beta", type=float, default=25.0)
    parser.add_argument("--persistent-workers", type=int, default=240)
    parser.add_argument(
        "--work-shards", type=int, choices=(1, 2, 4, 8), default=8
    )
    parser.add_argument(
        "--persistent-wpe", type=int, choices=(1, 2, 3, 4), default=3
    )
    args = parser.parse_args()

    from aiter.fused_moe import moe_sorting
    from aiter.ops.flydsl.kernels.megamoe_tile import hidden_fraction, prepare_local_a4w4_weights
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
        build_copy_put_signal_module,
        compile_hier_stage1_a4w4,
        compile_hier_stage1_a4w4_persistent,
    )
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
    from aiter.ops.flydsl.kernels.mxfp4_gemm1 import gemm1_grid
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility.fp4_utils import moe_mxfp4_sort

    torch.manual_seed(23)
    dev = torch.device("cuda", 0)
    m, h, inter, experts, topk, bm = args.tokens, 3584, 384, 56, 16, 32
    x = (torch.randn(m, h, device=dev) * 0.1).to(torch.bfloat16)
    w1 = (torch.randn(experts, 2 * inter, h, device=dev) * 0.03).to(torch.bfloat16)
    w2 = (torch.randn(experts, h, inter, device=dev) * 0.03).to(torch.bfloat16)
    score = torch.rand((m, experts), device=dev)
    vals, ids = torch.topk(score, topk, dim=1)
    weights = torch.softmax(vals, dim=1).float()
    prepared = prepare_local_a4w4_weights(w1, w2)
    sorted_ids, _, sorted_eids, nvalid, _ = moe_sorting(
        ids.to(torch.int32), weights, experts, h, torch.bfloat16, bm, accumulate=False
    )
    a1q, a1s = per_1x32_f4_quant(x, shuffle=False)
    a1ss = moe_mxfp4_sort(a1s.view(m, 1, h // 32), sorted_ids, nvalid, m, bm)
    m_indices = (sorted_ids & 0x00FFFFFF).to(torch.int32).contiguous()
    max_sorted = sorted_ids.shape[0]
    scale_rows = (max_sorted + 255) // 256 * 256
    scale_cols = ((inter // 32) + 7) // 8 * 8
    out_q = torch.zeros((max_sorted, inter // 2), dtype=torch.uint8, device=dev)
    out_s = torch.zeros(scale_rows * scale_cols, dtype=torch.uint8, device=dev)
    hidden = torch.zeros((m, h), dtype=torch.bfloat16, device=dev)
    grid = gemm1_grid(m, bm, NE=experts, TOPK=topk, INTER=inter, BN=256)
    actual_m_tiles = int(nvalid[0].item()) // bm
    actual_gemm_tiles = actual_m_tiles * ((2 * inter) // 256)

    src = torch.arange(64 * 1024, dtype=torch.int32, device=dev).remainder(251).to(torch.uint8)
    dst = torch.zeros_like(src)
    signal = torch.zeros(1, dtype=torch.int64, device=dev)
    stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    copy = build_copy_put_signal_module()
    activation_args = dict(
        activation=args.activation,
        swiglu_limit=args.swiglu_limit,
        situ_beta=args.situ_beta,
        situ_linear_beta=args.situ_linear_beta,
    )
    fused = compile_hier_stage1_a4w4(
        D_HIDDEN=h, D_INTER=inter, NE=experts, TOPK=topk, **activation_args
    )
    fused_copy_only = compile_hier_stage1_a4w4(
        D_HIDDEN=h,
        D_INTER=inter,
        NE=experts,
        TOPK=topk,
        enable_copy=True,
        enable_signal=False,
        **activation_args,
    )
    fused_idle = compile_hier_stage1_a4w4(
        D_HIDDEN=h,
        D_INTER=inter,
        NE=experts,
        TOPK=topk,
        enable_copy=False,
        enable_signal=False,
        **activation_args,
    )
    persistent = compile_hier_stage1_a4w4_persistent(
        D_HIDDEN=h,
        D_INTER=inter,
        NE=experts,
        TOPK=topk,
        WORK_SHARDS=args.work_shards,
        waves_per_eu_hint=args.persistent_wpe,
        enable_copy=True,
        enable_signal=True,
        **activation_args,
    )
    persistent_idle = compile_hier_stage1_a4w4_persistent(
        D_HIDDEN=h,
        D_INTER=inter,
        NE=experts,
        TOPK=topk,
        WORK_SHARDS=args.work_shards,
        waves_per_eu_hint=args.persistent_wpe,
        enable_copy=False,
        enable_signal=False,
        **activation_args,
    )
    persistent_strided = compile_hier_stage1_a4w4_persistent(
        D_HIDDEN=h,
        D_INTER=inter,
        NE=experts,
        TOPK=topk,
        WORK_SHARDS=args.work_shards,
        waves_per_eu_hint=args.persistent_wpe,
        scheduler="strided",
        enable_copy=True,
        enable_signal=True,
        **activation_args,
    )
    persistent_strided_idle = compile_hier_stage1_a4w4_persistent(
        D_HIDDEN=h,
        D_INTER=inter,
        NE=experts,
        TOPK=topk,
        WORK_SHARDS=args.work_shards,
        waves_per_eu_hint=args.persistent_wpe,
        scheduler="strided",
        enable_copy=False,
        enable_signal=False,
        **activation_args,
    )
    # Device-side entry tickets derive the epoch and let ticket 0 reset the
    # sharded heads. Repeated launches require no host memset or counter kernel.
    entry_count = torch.zeros(
        persistent.entry_count_int64, dtype=torch.int64, device=dev
    )
    epoch_gate = torch.zeros(
        persistent.epoch_gate_int32, dtype=torch.int32, device=dev
    )
    work_head = torch.zeros(
        persistent.work_head_int32, dtype=torch.int32, device=dev
    )
    idle_entry_count = torch.zeros_like(entry_count)
    idle_epoch_gate = torch.zeros_like(epoch_gate)
    idle_work_head = torch.zeros_like(work_head)
    strided_entry_count = torch.zeros_like(entry_count)
    strided_epoch_gate = torch.zeros_like(epoch_gate)
    strided_work_head = torch.zeros_like(work_head)
    strided_idle_entry_count = torch.zeros_like(entry_count)
    strided_idle_epoch_gate = torch.zeros_like(epoch_gate)
    strided_idle_work_head = torch.zeros_like(work_head)
    persistent_generation = 0
    persistent_idle_generation = 0
    strided_generation = 0
    strided_idle_generation = 0

    def comm_only():
        copy(src, dst, signal, src.numel(), 41, stream=stream)

    def compute_only():
        compute_on(stream)

    def compute_on(target_stream):
        flydsl_mxfp4_gemm1(
            a_quant=a1q,
            a_scale_sorted_shuffled=a1ss,
            w1_u8=prepared.w1.view(torch.uint8),
            w1_scale_u8=prepared.w1_scale.view(torch.uint8),
            sorted_expert_ids=sorted_eids,
            cumsum_tensor=nvalid,
            m_indices=m_indices,
            inter_sorted_quant=out_q,
            inter_sorted_shuffled_scale=out_s,
            hidden_states=hidden,
            n_tokens=m,
            BM=bm,
            use_nt=True,
            inline_quant=False,
            NE=experts,
            D_HIDDEN=h,
            D_INTER=inter,
            topk=topk,
            act=args.activation,
            swiglu_limit=args.swiglu_limit,
            situ_beta=args.situ_beta,
            situ_linear_beta=args.situ_linear_beta,
            stream=target_stream,
        )

    def persistent_port_compute_on(target_stream):
        flydsl_mxfp4_gemm1(
            a_quant=a1q,
            a_scale_sorted_shuffled=a1ss,
            w1_u8=prepared.w1.view(torch.uint8),
            w1_scale_u8=prepared.w1_scale.view(torch.uint8),
            sorted_expert_ids=sorted_eids,
            cumsum_tensor=nvalid,
            m_indices=m_indices,
            inter_sorted_quant=out_q,
            inter_sorted_shuffled_scale=out_s,
            hidden_states=hidden,
            n_tokens=m,
            BM=bm,
            use_nt=True,
            inline_quant=False,
            NE=experts,
            D_HIDDEN=h,
            D_INTER=inter,
            topk=topk,
            act=args.activation,
            swiglu_limit=args.swiglu_limit,
            situ_beta=args.situ_beta,
            situ_linear_beta=args.situ_linear_beta,
            persistent=True,
            persistent_blocks=args.persistent_workers,
            stream=target_stream,
        )

    def persistent_port_compute():
        persistent_port_compute_on(stream)

    def serial():
        comm_only()
        compute_only()

    def sidecar_overlap_once():
        copy(src, dst, signal, src.numel(), 47, stream=comm_stream)
        compute_on(compute_stream)

    def persistent_sidecar_overlap_once():
        copy(src, dst, signal, src.numel(), 48, stream=comm_stream)
        persistent_port_compute_on(compute_stream)

    def time_sidecar(fn) -> float:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e6 / args.iters

    def joint():
        _joint_launch(fused, 43)

    def joint_copy_only():
        _joint_launch(fused_copy_only, 44)

    def joint_idle():
        _joint_launch(fused_idle, 45)

    def persistent_joint():
        nonlocal persistent_generation
        persistent_generation += 1
        _persistent_launch(
            persistent,
            entry_count,
            epoch_gate,
            work_head,
            1000 + persistent_generation,
        )

    def persistent_compute_only():
        nonlocal persistent_idle_generation
        persistent_idle_generation += 1
        _persistent_launch(
            persistent_idle,
            idle_entry_count,
            idle_epoch_gate,
            idle_work_head,
            2000 + persistent_idle_generation,
        )

    def persistent_strided_joint():
        nonlocal strided_generation
        strided_generation += 1
        _persistent_launch(
            persistent_strided,
            strided_entry_count,
            strided_epoch_gate,
            strided_work_head,
            3000 + strided_generation,
        )

    def persistent_strided_compute_only():
        nonlocal strided_idle_generation
        strided_idle_generation += 1
        _persistent_launch(
            persistent_strided_idle,
            strided_idle_entry_count,
            strided_idle_epoch_gate,
            strided_idle_work_head,
            4000 + strided_idle_generation,
        )

    def _joint_launch(kernel, generation):
        kernel(
            src.data_ptr(),
            dst.data_ptr(),
            signal.data_ptr(),
            src.numel(),
            generation,
            a1q.data_ptr(),
            a1ss.data_ptr(),
            prepared.w1.data_ptr(),
            prepared.w1_scale.data_ptr(),
            sorted_eids.data_ptr(),
            nvalid.data_ptr(),
            m_indices.data_ptr(),
            m,
            grid,
            out_q.data_ptr(),
            out_s.data_ptr(),
            hidden.data_ptr(),
            stream=stream,
        )

    def _persistent_launch(
        kernel, entry_count_arg, epoch_gate_arg, work_head_arg, generation
    ):
        kernel(
            src.data_ptr(),
            dst.data_ptr(),
            signal.data_ptr(),
            src.numel(),
            generation,
            entry_count_arg.data_ptr(),
            epoch_gate_arg.data_ptr(),
            work_head_arg.data_ptr(),
            a1q.data_ptr(),
            a1ss.data_ptr(),
            prepared.w1.data_ptr(),
            prepared.w1_scale.data_ptr(),
            sorted_eids.data_ptr(),
            nvalid.data_ptr(),
            m_indices.data_ptr(),
            m,
            args.persistent_workers,
            out_q.data_ptr(),
            out_s.data_ptr(),
            hidden.data_ptr(),
            stream=stream,
        )

    t_comm = _time_us(comm_only, warmup=10, iters=args.iters)
    t_compute = _time_us(compute_only, warmup=10, iters=args.iters)
    t_port_persistent = _time_us(
        persistent_port_compute, warmup=10, iters=args.iters
    )
    t_serial = _time_us(serial, warmup=10, iters=args.iters)
    t_joint = _time_us(joint, warmup=10, iters=args.iters)
    t_joint_copy = _time_us(joint_copy_only, warmup=10, iters=args.iters)
    t_joint_idle = _time_us(joint_idle, warmup=10, iters=args.iters)
    t_persistent = _time_us(
        persistent_joint, warmup=10, iters=args.iters
    )
    t_persistent_idle = _time_us(
        persistent_compute_only, warmup=10, iters=args.iters
    )
    t_persistent_strided = _time_us(
        persistent_strided_joint, warmup=10, iters=args.iters
    )
    t_persistent_strided_idle = _time_us(
        persistent_strided_compute_only, warmup=10, iters=args.iters
    )
    t_sidecar = time_sidecar(sidecar_overlap_once)
    t_persistent_sidecar = time_sidecar(persistent_sidecar_overlap_once)
    hidden = hidden_fraction(t_comm, t_compute, t_joint)
    print(
        "H1_OVERLAP",
        f"tokens={m}",
        f"activation={args.activation}",
        f"m_tiles={actual_m_tiles}",
        f"gemm_tiles={actual_gemm_tiles}",
        f"comm_us={t_comm:.3f}",
        f"compute_us={t_compute:.3f}",
        f"persistent_port_compute_us={t_port_persistent:.3f}",
        f"serial_us={t_serial:.3f}",
        f"joint_us={t_joint:.3f}",
        f"joint_copy_no_signal_us={t_joint_copy:.3f}",
        f"joint_idle_comm_us={t_joint_idle:.3f}",
        f"persistent_workers={args.persistent_workers}",
        f"work_shards={args.work_shards}",
        f"persistent_wpe={args.persistent_wpe}",
        f"persistent_us={t_persistent:.3f}",
        f"persistent_idle_us={t_persistent_idle:.3f}",
        f"persistent_strided_us={t_persistent_strided:.3f}",
        f"persistent_strided_idle_us={t_persistent_strided_idle:.3f}",
        f"persistent_strided_speedup_vs_serial={(t_serial / t_persistent_strided - 1.0):.4f}",
        f"persistent_hidden_fraction={hidden_fraction(t_comm, t_compute, t_persistent):.4f}",
        f"persistent_speedup_vs_serial={(t_serial / t_persistent - 1.0):.4f}",
        f"persistent_scheduler_overhead={(t_persistent_idle / t_compute - 1.0):.4f}",
        f"sidecar_overlap_us={t_sidecar:.3f}",
        f"persistent_sidecar_us={t_persistent_sidecar:.3f}",
        f"hidden_fraction={hidden:.4f}",
        f"serial_speedup={(t_serial / t_joint - 1.0):.4f}",
        f"sidecar_hidden_fraction={hidden_fraction(t_comm, t_compute, t_sidecar):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
