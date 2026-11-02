# Restore C16 latency tuning

The complete bundle `/home/vigokhal/monokernel-c16-latency-20261022.tgz` contains this experiment directory and the complete preceding bundle `monokernel-c16-20261021.tgz`. The latter includes the earlier C1–C8 archive and all checkpoint microfixture dependencies. It does not bundle the full DeepSeek checkpoint or Docker image layers.

1. Extract the outer archive into `/home/vigokhal`. Follow `dsv4-monokernel/c16-monokernel/RESTORE.md` from the nested C16 archive to restore its dependency tree and prior archives. Preserve the newer `c16-latency-tuning` directory.
2. Use the existing `vigokhal_dsv4_tp4` container, or recreate its documented environment. Host `/home/vigokhal/dsv4-monokernel` must be mounted at `/work`; ATOM is `/app/ATOM`, AITER `/app/aiter`, Flash weights `/data/DeepSeek-V4-Flash`. The earlier bundle records image identity, commits and local dependency modifications.
3. Read REPORT.md. C16 serving is not enabled. Do not substitute these layer timings for serving throughput. New configurations are tested only at C16.
4. Inside the container, run `HIP_VISIBLE_DEVICES=0 bash /work/c16-latency-tuning/run_worker.sh 0`, with workers 1/2/3 on GPUs 1/2/3. The workers may run concurrently. They source the selected configuration and write to `native-c16/rerun/` plus paired full-layer files. Repeated runs overwrite that named result directory; set EXPERIMENT to retain multiple runs.
5. `frozen_before` pins the old kernel implementation. `paired_swa.py` and `paired_compressed.py` compare old/new in the same process, on the same inputs, alternating timed HIP graphs and reporting round two. Current native region checks use checkpoint weights but some synthetic activations/cache fixtures. Full-layer checks use synthetic full-layer fixtures.

The selected recipe requires the enlarged counters in this directory's `scratch_batch.py`. Fine-grained mailbox trials remain disabled. Do not activate old C8 mailbox layouts at C16; they overlap. Experimental N128 MoE and shared retile paths have guards against unsupported legacy mailbox modes.

Source and raw rejected candidates are retained to avoid repeating failed work. In particular, `hca_overlap*` precede the coefficient-dtype fix, and `ob32_8_512_2` violates the 256-CTA grid capacity; ignore their performance results. See REPORT.md for accepted variants and validation limits.
