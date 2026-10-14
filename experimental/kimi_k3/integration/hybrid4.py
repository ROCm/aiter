"""AMD gfx950 K3 complete latent-MoE adapter, optionally including KDA output.

B8 uses fused route metadata and baseline A4W4 expert kernels.
Smaller batches retain their A16W4 monokernels. Dense weights are packed once.
IPC workspaces belong to one layer/batch and survive CUDA graph replay.
"""

import ctypes as C
import os
import torch
import torch.distributed as dist
import time
from pathlib import Path
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm._aiter_ops import rocm_aiter_ops
from hip_ipc import PeerMemory, DevicePointer
from k3_gluon.moe_complete import pack_dense
from k3_gluon_serving_tuned.perf_reference_chain import chain as complete_moe
from k3_gluon_serving_front32.perf_reference_chain import chain as complete_moe_b1
from k3_gluon_serving_b4tiles.perf_reference_chain import (
    chain as complete_moe_smallbatch,
)
from k3_gluon_serving_b4grid512.perf_reference_chain import (
    chain as complete_moe_grid512,
)
from k3_gluon_hybrid_sort_b8.perf_reference_chain import chain as hybrid_b8
from k3_gluon_hybrid_sort_b8.pipeline import launch_hybrid, stage_options
from k3_gluon_hybrid_sort_b8.host import (
    expert_weights,
    run_experts,
    prepare_metadata,
    run_prepared_experts,
)
from vllm.distributed import tensor_model_parallel_all_reduce

from k3_gluon.complete_host import workspace, arguments
from k3_gluon.projection import pack_weights
from k3_gluon.serving import allocate_workspace

logger = init_logger(__name__)
LIVE_ADAPTERS = []


def warmup_kernels(entry, call, options):
    kernel, grid = entry["kernel"], entry["grid"]
    variants = [options]
    if entry.get("hybrid"):
        variants = stage_options(options, entry["sort_metadata"], grid)
    return [kernel.warmup(*call, grid=(grid,), **variant) for variant in variants]


def launch(entry, call, options):
    kernel, grid = entry["kernel"], entry["grid"]
    if entry.get("hybrid"):
        launch_hybrid(
            kernel,
            call,
            options,
            grid=grid,
            state=entry["state"],
            weights=entry["expert_weights"],
            metadata=entry["sort_metadata"],
            experts=run_prepared_experts,
        )
    else:
        kernel[(grid,)](*call, **options)


def reference_moe(adapter, x, entry):
    if not entry.get("hybrid"):
        return adapter.layer.mlp(x)
    # Independent baseline operators around the unchanged AITER A4W4 experts.
    moe = adapter.layer.mlp
    tp = get_tp_group()
    logits = moe.gate(x)[0]
    scores = logits.sigmoid()
    ids = (
        torch.argsort(
            scores + moe.gate.e_score_correction_bias, descending=True, stable=True
        )[:, :16]
        .int()
        .contiguous()
    )
    weights = scores.gather(1, ids.long())
    weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
    latent = moe.routed_expert_down_proj(x)[0]
    local = run_experts(latent, entry["expert_weights"], ids, weights)
    latent = tensor_model_parallel_all_reduce(local)
    latent = moe.routed_expert_norm(latent)
    shared = moe.shared_experts(x)
    shard_size = 7168 // tp.world_size
    shard_start = tp.rank_in_group * shard_size
    up = moe.routed_expert_up_proj.weight.narrow(0, shard_start, shard_size)
    shared.narrow(-1, shard_start, shard_size).addmm_(latent, up.t())
    baseline = tensor_model_parallel_all_reduce(shared)

    # Independent mathematical reference. Keep the baseline-operator result
    # for diagnostics, so backend-specific rounding is visible separately.
    def reduce_exact(value):
        parts = [torch.empty_like(value) for _ in range(tp.world_size)]
        dist.all_gather(parts, value, group=tp.device_group)
        total = torch.zeros_like(value, dtype=torch.float32)
        for part in parts:
            total += part.float()
        return total.bfloat16()

    def nrmse(a, b):
        return float(
            (a.float() - b.float()).square().mean().sqrt()
            / b.float().square().mean().sqrt().clamp_min(1.0e-12)
        )

    exact_scores = (x.float() @ moe.gate.weight.float().t()).sigmoid()
    exact_ids = (
        torch.argsort(
            exact_scores + moe.gate.e_score_correction_bias,
            descending=True,
            stable=True,
        )[:, :16]
        .int()
        .contiguous()
    )
    exact_weights = exact_scores.gather(1, exact_ids.long())
    exact_weights = (exact_weights / exact_weights.sum(-1, keepdim=True)).contiguous()
    exact_latent = (
        x.float() @ moe.routed_expert_down_proj.weight.float().t()
    ).bfloat16()
    exact_local = run_experts(
        exact_latent, entry["expert_weights"], exact_ids, exact_weights
    )
    exact_routed = reduce_exact(exact_local).float()
    exact_norm = (
        exact_routed
        * torch.rsqrt(
            exact_routed.square().mean(-1, keepdim=True)
            + moe.routed_expert_norm.variance_epsilon
        )
        * moe.routed_expert_norm.weight.float()
    ).bfloat16()
    gu = (x.float() @ moe.shared_experts.gate_up_proj.weight.float().t()).bfloat16()
    gate, value = gu.float().chunk(2, -1)
    mid = (
        4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(value / 25)
    ).bfloat16()
    exact_shared = (
        mid.float() @ moe.shared_experts.down_proj.weight.float().t()
    ).bfloat16()
    exact_shared[:, shard_start : shard_start + shard_size] = (
        exact_shared[:, shard_start : shard_start + shard_size].float()
        + exact_norm.float() @ up.float().t()
    ).bfloat16()
    exact = reduce_exact(exact_shared)
    check = dict(
        layer=adapter.layer.layer_idx,
        reference_nrmse=nrmse(baseline, exact),
        ids_equal=bool(torch.equal(ids, exact_ids)),
        scores_nrmse=nrmse(scores, exact_scores),
        routed_local_nrmse=nrmse(local, exact_local),
        normed_nrmse=nrmse(latent, exact_norm),
        combined_local_nrmse=nrmse(shared, exact_shared),
    )
    if check["reference_nrmse"] > 0.025:
        logger.warning("Hybrid baseline reference diagnostic: %s", check)
    entry["reference_diagnostic"] = check
    return exact


class CompleteMoE:
    def __init__(self, layer, projection):
        self.layer = layer
        self.projection = projection
        self.weights = None
        self.workspaces = {}
        self.calls = 0
        LIVE_ADAPTERS.append(self)

    def prepare_weights(self):
        tp = get_tp_group()
        moe = self.layer.mlp
        experts = moe.experts.routed_experts
        assert rocm_aiter_ops.get_fused_moe_situv2_activation() == "a16w4"
        assert moe.tp_size == tp.world_size and moe.routed_scaling_factor == 1.0
        assert experts.w13_weight.shape == (896, 6144 // tp.world_size, 1792)
        assert experts.w2_weight.shape == (896, 3584, 1536 // tp.world_size)
        shard = slice(
            tp.rank_in_group * 7168 // tp.world_size,
            (tp.rank_in_group + 1) * 7168 // tp.world_size,
        )
        self.weights = dict(
            Router=pack_dense(moe.gate.weight),
            Bias=moe.gate.e_score_correction_bias,
            Down=pack_dense(moe.routed_expert_down_proj.weight),
            Shared=pack_dense(moe.shared_experts.gate_up_proj.weight),
            WUp=experts.w13_weight.view(torch.uint8),
            SUp=experts.w13_weight_scale.view(torch.uint8),
            WDown=experts.w2_weight.view(torch.uint8),
            SDown=experts.w2_weight_scale.view(torch.uint8),
            NormWeight=moe.routed_expert_norm.weight,
            SharedDown=pack_dense(moe.shared_experts.down_proj.weight),
            LatentUp=pack_dense(moe.routed_expert_up_proj.weight[shard]),
        )
        if self.projection:
            self.projection_weight = pack_weights(self.layer.self_attn.o_proj.weight)

    def __call__(self, x, prefix=None, blocks=None):
        tp = get_tp_group()
        layer = self.layer
        batch = x.shape[0]
        use_front32 = (batch <= 4 or batch == 8) and tp.world_size == 8
        kernel = (
            (complete_moe_b1 if batch == 1 else complete_moe_smallbatch)
            if use_front32
            else complete_moe
        )
        grid = 512 if batch in (2, 4, 8) and tp.world_size == 8 else 256
        if grid == 512:
            kernel = hybrid_b8 if batch == 8 else complete_moe_grid512
        capturing = torch.cuda.is_current_stream_capturing()
        if self.weights is None:
            assert not capturing
            self.prepare_weights()
        fresh = batch not in self.workspaces
        if fresh:
            assert not capturing
            down = PeerMemory(
                2 * tp.world_size * batch * 1792 * 8,
                tp.rank_in_group,
                tp.world_size,
                group=tp.cpu_group,
            )
            final = PeerMemory(
                2 * tp.world_size * batch * 3584 * 8,
                tp.rank_in_group,
                tp.world_size,
                group=tp.cpu_group,
            )
            state = workspace(batch, tp.world_size)
            if grid != 256:
                state["Epochs"] = torch.zeros(grid, device=x.device, dtype=torch.int32)
                state["Errors"] = torch.zeros(
                    grid * 512, device=x.device, dtype=torch.int32
                )
            if use_front32:
                state["LatentReady"] = torch.zeros(
                    (batch, 112), device=x.device, dtype=torch.uint64
                )
                state["SharedReady"] = torch.zeros(
                    (batch, 48), device=x.device, dtype=torch.uint64
                )
            entry = dict(down=down, final=final, state=state, launched=False, grid=grid)
            entry["hybrid"] = batch == 8 and tp.world_size == 8
            if entry["hybrid"]:
                entry["expert_weights"] = expert_weights(self.weights)
                # Compile the independent AITER launches before any mailbox wait.
                entry["sort_metadata"] = prepare_metadata(
                    state["IDs"], state["Weights"]
                )
                entry["local_routed"] = run_prepared_experts(
                    state["Latent"],
                    entry["expert_weights"],
                    state["IDs"],
                    state["Weights"],
                    entry["sort_metadata"],
                )

            if self.projection:
                entry["projection"] = PeerMemory(
                    2 * tp.world_size * batch * 3584 * 8,
                    tp.rank_in_group,
                    tp.world_size,
                    group=tp.cpu_group,
                )
                entry["outputs"] = allocate_workspace(batch)
                entry["delta"] = torch.empty(
                    batch, 7168, dtype=torch.bfloat16, device=x.device
                )
                entry["attention"] = torch.zeros(
                    batch * 3584, dtype=torch.uint64, device=x.device
                )
            self.workspaces[batch] = entry
        entry = self.workspaces[batch]
        state = entry["state"]
        options = dict(
            WORLD=tp.world_size,
            AITER_LAYOUT=True,
            num_warps=8,
            enable_fp_fusion=False,
            launch_cooperative_grid=False,
            WATCHDOG_TICKS=entry["down"].wall_clock_khz * 1000 * 5,
        )
        options.update(
            BATCH_TAIL=True,
            BUFFER=True,
            LDS=True,
            ZERO_GUARD=3,
            DOWN_CHUNK=128,
            DOWN_UNROLL=3,
            DOWN_LDS=True,
            MFMA_LANES=True,
        )
        if use_front32:
            options.update(WAVE_PUBLISH=True, TAIL_ROWS=32)
        options.update(
            Residual=prefix,
            FinalPrefix=state["FinalPrefix"],
            START=0 if self.projection else 2,
            END=4,
            BEGIN=True,
            TUNED=True,
        )
        if self.projection:
            outputs = entry["outputs"]
            options.update(
                ProjectionX=x,
                ProjectionW=self.projection_weight,
                PeersProjection=entry["projection"].addresses,
                LocalProjection=DevicePointer(entry["projection"].ptr.value),
                Delta=entry["delta"],
                Attention=entry["attention"],
                Prefix=outputs[0] if prefix is None else prefix,
                Blocks=blocks,
                Norm=layer.mlp_res_norm.weight,
                Query=layer.mlp_res_proj.weight,
                OutNorm=layer.post_attention_layernorm.weight,
                Updated=outputs[0],
                Quant=outputs[2],
                Scales=outputs[3],
                Stats=outputs[4],
                InputReady=outputs[5],
                N_BLOCKS=layer.prev_valid_blocks + int(layer.is_block_write_layer),
                BOUNDARY=layer.is_block_write_layer,
                BLOCK_STRIDE_M=blocks.stride(0),
                Residual=outputs[0],
            )
            x = outputs[1]
        if entry["hybrid"]:
            options.update(entry["sort_metadata"])
        call = arguments(
            x,
            self.weights,
            state,
            entry["down"],
            entry["final"],
            tp.rank_in_group,
            tp.world_size,
        )
        entry["kernel"] = kernel
        entry["options"] = options
        if fresh:
            compiled_kernels = warmup_kernels(entry, call, options)
            occupancy = entry[
                "down"
            ].hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
            occupancy.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
            occupancy.restype = C.c_int
            cus = torch.cuda.get_device_properties(x.device).multi_processor_count
            capacities = []
            for compiled in compiled_kernels:
                _ = compiled.run
                assert compiled.n_spills == 0, (
                    layer.layer_idx,
                    batch,
                    compiled.n_spills,
                )
                resident = C.c_int()
                entry["down"].call(
                    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
                    C.byref(resident),
                    C.c_void_p(compiled.function),
                    512,
                    compiled.metadata.shared,
                )
                assert resident.value * cus >= grid, (
                    "resident grid capacity",
                    resident.value,
                    cus,
                    grid,
                )
                capacities.append(resident.value)
            entry["resident_blocks_per_cu"] = min(capacities)
            entry["num_cus"] = cus
            torch.cuda.synchronize()
            dist.barrier(group=tp.cpu_group)
            logger.info(
                "K3 coherent staged Gluon MoE prepared: layer=%d batch=%d projection=%s registers=%d",
                layer.layer_idx,
                batch,
                self.projection,
                compiled.n_regs,
            )
        metadata = get_forward_context().attn_metadata
        comm = tp.device_communicator
        warmup = (
            (comm.ca_comm is not None and comm.ca_comm._IS_CAPTURING)
            or (
                comm.aiter_ar_comm is not None
                and comm.aiter_ar_comm.aiter_ca._IS_CAPTURING
            )
        ) and not capturing
        inactive = metadata is None or warmup
        if self.projection:
            inactive = inactive or metadata.get(layer.self_attn.prefix) is None
        if not inactive:
            # Profiling can prepare a specialization long before its first
            # execution with attention metadata. Synchronize THAT first eager
            # execution too, so other kernels' lazy compilation is not charged
            # against another rank's device mailbox deadline.
            first_eager = not entry["launched"] and not capturing
            if first_eager:
                entered = time.monotonic_ns()
                torch.cuda.synchronize()
                ready = time.monotonic_ns()
                timing = [None] * tp.world_size
                dist.all_gather_object(timing, (entered, ready), group=tp.cpu_group)
                entry["first_eager_arrival_spread_ms"] = (
                    max(t[1] for t in timing) - min(t[1] for t in timing)
                ) / 1.0e6
                dist.barrier(group=tp.cpu_group)
            launch(entry, call, options)
            self.calls += 1
            if first_eager:
                torch.cuda.synchronize()
                bad = int(state["Errors"].count_nonzero().item())
                if bad:
                    path = Path(os.environ.get("KIMI_K3_RESULTS", "results"))
                    path.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {name: value.cpu() for name, value in state.items()},
                        path
                        / f"failure-layer{layer.layer_idx}-b{batch}-rank{tp.rank_in_group}.pt",
                    )
                    logger.error(
                        "K3 fused mailbox failure: layer=%d batch=%d rank=%d flags=%d",
                        layer.layer_idx,
                        batch,
                        tp.rank_in_group,
                        bad,
                    )
                failures = [None] * tp.world_size
                dist.all_gather_object(failures, bad, group=tp.cpu_group)
                assert not any(failures), (layer.layer_idx, batch, failures)
                # Experiment-only validation, outside captured/timed execution:
                # run the installed unfused MoE on the identical live input.
                reference = reference_moe(self, x, entry)
                difference = state["Out"].float() - reference.float()
                nrmse = (
                    difference.square().mean().sqrt()
                    / reference.float().square().mean().sqrt().clamp_min(1.0e-12)
                )
                validation = dict(
                    nrmse=float(nrmse.item()),
                    max_abs=float(difference.abs().max().item()),
                )
                checks = [None] * tp.world_size
                dist.all_gather_object(checks, validation, group=tp.cpu_group)
                if not all(v["nrmse"] < 0.025 for v in checks):
                    path = Path(os.environ.get("KIMI_K3_RESULTS", "results"))
                    path.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        dict(
                            input=x.cpu(),
                            actual=state["Out"].cpu(),
                            expected=reference.cpu(),
                            ids=state["IDs"].cpu(),
                            weights=state["Weights"].cpu(),
                        ),
                        path
                        / f"numerical-layer{layer.layer_idx}-b{batch}-rank{tp.rank_in_group}.pt",
                    )
                assert all(v["nrmse"] < 0.025 for v in checks), (
                    layer.layer_idx,
                    batch,
                    checks,
                )
                entry["unfused_validation"] = validation
                logger.info(
                    "K3 complete MoE live validation passed: layer=%d batch=%d max_rank_nrmse=%.6f first_arrival_spread_ms=%.3f",
                    layer.layer_idx,
                    batch,
                    max(v["nrmse"] for v in checks),
                    entry["first_eager_arrival_spread_ms"],
                )
                entry["launched"] = True
        return state["FinalPrefix"], None


@torch.inference_mode()
def validate_batch(batch):
    """Post-benchmark MoE checks for every layer, including graph-only shapes."""
    tp = get_tp_group()
    checks = []
    for adapter in LIVE_ADAPTERS:
        entry = adapter.workspaces[batch]
        state = entry["state"]
        generator = torch.Generator(device=state["Out"].device)
        generator.manual_seed(9000 + adapter.layer.layer_idx)
        x = torch.randn(
            (batch, 7168),
            device=state["Out"].device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        x *= adapter.layer.post_attention_layernorm.weight
        options = dict(entry["options"], START=2, Residual=None, FinalPrefix=None)
        call = arguments(
            x,
            adapter.weights,
            state,
            entry["down"],
            entry["final"],
            tp.rank_in_group,
            tp.world_size,
        )
        grid = entry["grid"]
        for compiled in warmup_kernels(entry, call, options):
            _ = compiled.run
            assert compiled.n_spills == 0
            resident = C.c_int()
            entry["down"].call(
                "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
                C.byref(resident),
                C.c_void_p(compiled.function),
                512,
                compiled.metadata.shared,
            )
            assert resident.value * entry["num_cus"] >= grid
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)
        reference = reference_moe(adapter, x, entry)
        launch(entry, call, options)
        torch.cuda.synchronize()
        error = state["Out"].float() - reference.float()
        nrmse = (
            error.square().mean().sqrt()
            / reference.float().square().mean().sqrt().clamp_min(1.0e-12)
        )
        check = dict(
            layer=adapter.layer.layer_idx,
            batch=batch,
            nrmse=float(nrmse),
            max_abs=float(error.abs().max()),
            errors=int(state["Errors"].count_nonzero()),
        )
        checks.append(check)
    return checks


def inspect_adapters():
    torch.cuda.synchronize()
    rows = []
    for adapter in LIVE_ADAPTERS:
        for batch, entry in adapter.workspaces.items():
            s = entry["state"]
            epoch = s["Epochs"].cpu()
            rows.append(
                dict(
                    layer=adapter.layer.layer_idx,
                    batch=batch,
                    projection=adapter.projection,
                    grid_ctas=entry["grid"],
                    hybrid=entry.get("hybrid", False),
                    sorting_fused=entry.get("hybrid", False),
                    expert_precision="a4w4" if entry.get("hybrid") else "a16w4",
                    launch_mode="normal",
                    resident_blocks_per_cu=entry.get("resident_blocks_per_cu"),
                    num_cus=entry.get("num_cus"),
                    errors=int(s["Errors"].count_nonzero().item()),
                    epoch_min=int(epoch.min()),
                    epoch_max=int(epoch.max()),
                    python_calls=adapter.calls,
                    unfused_validation=entry.get("unfused_validation"),
                    first_eager_arrival_spread_ms=entry.get(
                        "first_eager_arrival_spread_ms"
                    ),
                )
            )
    return rows
