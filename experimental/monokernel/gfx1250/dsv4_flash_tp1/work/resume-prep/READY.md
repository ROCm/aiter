# Resume preparation — GPU work remains paused

This preparation only read files and Docker metadata, copied source from a stopped container, parsed Python source without importing it, checked shell syntax and verified the archive checksum. No containers, GPU runtimes, servers, clients, or benchmarks were started. No kernel or integration implementation was changed.

## Recovered state

- Image: `rocm/fw-bringup:dsv4_tp4_mk_expts`, digest `sha256:aabe8ff74d659cd4cd25bbe7c25e5b85e4cd72f26cf3377dd63511a8b9509c24`.
- Existing container: `vigokhal_dsv4_tp4`, stopped at preparation time. Leave stopped until GPU work is authorized again.
- Project: `/home/vigokhal/dsv4-monokernel`, mounted as `/work`; external checkpoint `/data/DeepSeek-V4-Flash`.
- Complete archive: `/home/vigokhal/dsv4-monokernel-resume-20261021.tgz`. Checksum and CPU checks in `CPU_CHECKS.json`.
- ATOM model source recovered from the stopped container exactly matches `modelwide/atom-source/atom/models/deepseek_v4.py`.
- Authoritative prior results and limitations: `../concurrency-rollout/REPORT.md`. C1 full-layer baseline/candidate total throughput152.31/199.23tok/s; C2 QKV-only287.53/301.15tok/s. C2 remains preliminary; C4/C8 candidate long-run memory faults are unresolved.

## Prepared diagnostic matrix

`prepare_commands.py` writes `commands.json` and does not execute commands. Four independent single-GPU configurations are ready: native control through custom-op boundary, fused graph, fused projection with native norm/RoPE/cache tail, and fused eager. Each has separate GPU/ports/logs. Confirm resource availability before eventual use. Start the container only after authorization; do not automatically start these jobs.

The general `run_flash_experiment.sh` intentionally clears native-control/native-tail flags. Diagnostic commands therefore call `server.sh` or `eager-server.sh` directly with explicit environments. They clear inherited diagnostic flags and avoid HIP/ROCR double-remapping. They preserve the original scripts and measured results.

## Next work after GPU authorization

1. Run the short diagnostic matrix at C8,1K input/64 output,16requests. This is only localization; it is not comparable to1K/1K performance.
2. Extend surviving configurations to1K/1K. Native control previously survived16requests. Native-tail graph previously also failed, so scatter alone is not an established explanation. Eager long diagnosis was interrupted during warmup and must be completed.
3. Capture matching native and candidate stage outputs/private state. Record actual shapes, strides, destination bounds, padded-token metadata, and barrier health. Inspect persistent output buffers and custom-op mutation/alias contracts under capture/replay. These are diagnostic hypotheses, not confirmed root causes.
4. Fix the first demonstrated divergence or invalid access; recheck all layers and state transitions. Preserve WMMA, early TDM/L2 prefetch unless an ablation establishes a causal fault. Do not re-enable the rejected serialized full-batch kernel.
5. After stable long-run correctness and paired generation checks, benchmark C2/C4/C8 model-wide. Start each server once, run the client twice, report ONLY run2 TTFT/TPOT/total tok/s. Keep diagnostic synchronization out of performance measurements. Native baselines exist, but rerun if environment/settings change.
6. Extend fusion beyond QKV into batched attention/mHC/MoE only after the current integration is stable. C1 already fuses both attention and MoE per layer; C2+ currently does not.

No new performance or correctness claims are made by this CPU-only preparation.
