> Updated 2026-10-21: corrected C1 model-wide graph serving completed two clients and a limited GSM8K regression check (31/32, same as native). See `corrected-c1-summary.json`. Earlier rejected results remain for diagnosis. C2/C4/C8 work is tracked in `../concurrency-rollout`; it is not a completed batched full-layer monokernel.

# Flash model-wide monokernel experiment

Target: gfx1250, TP1, concurrency 1, 1024 input / 1024 output, maximum context
2048. This is experimental and not accuracy-qualified. All 43 layers are bound
with their own weights: 2 ratio-0, 21 ratio-4 and 20 ratio-128.

Environment: `vigokhal_dsv4_tp4` container, image
`rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, checkpoint
`/data/DeepSeek-V4-Flash`. Host project `/home/vigokhal/dsv4-monokernel` is mounted
as `/work`. ATOM base commit f19778df4054e94d64d6f6bf6dd68f5681bd5945;
AITER 7cddd8e632e8d07eebdf9e1cab7e7a9f1e8c65fe.

The complete archive includes previous full-layer dependencies plus this folder.
Extract it into the directory mounted at `/work`. The checkpoint and container
image are external dependencies and are not in the archive.

`atom-source/` contains exact modified/additional ATOM source files from the
experiment. Compare before overlaying onto a matching checkout; do not discard
unrelated local edits. `atom-working.patch` records tracked changes and
`atom-modelwide.patch` isolates the latest hook extension. To restore this exact saved integration onto the documented compatible checkout, run `python /work/modelwide/install.py`; it backs up differing files before overlaying the snapshot. Prefer the supplied image.

The current installer provides `/app/scripts/run_dsv4_flash_experiment.sh`. Use the commands in `../concurrency-rollout/REPORT.md`; the older wrapper below is retained only as historical documentation.
Example server (choose an idle GPU and ports):

```bash
HIP_VISIBLE_DEVICES=0 PORT=8301 HTTP_PORT=8300 \
 SERVER_LOG=/work/modelwide/fused-server.log \
 bash /app/scripts/run_dsv4_flash_modelwide_c1.sh fused
```

Run the client twice against that same live server (the script does both):

```bash
HTTP_PORT=8300 bash /app/scripts/run_dsv4_flash_modelwide_c1.sh client-fused
```

Use `baseline` / `client-baseline` for native. Compare only `*-client-round2.json`.
Do not change the 2048 context cap: short-context identity top-k elimination is
only valid while compressed rows <=512. C>1 and prefill use native operations.
The wrapper materializes pending residual post operations during prefill.

`DESIGN.md`, `coverage.json`, validation JSONs, ISA dumps and logs document the
implementation and its limits. `quality-smoke.json` (when present) is only a few
deterministic completion comparisons, not GSM8K or an accuracy qualification.
