# Restore and reproduce

Use the image, model and mount recorded in ../latency-overlap/environment.json. The full archive contains the source dependencies; the checkpoint and container image are external prerequisites.

Extract the archive into /home/vigokhal/dsv4-monokernel and mount it at /work. Restore the ATOM overlay described in ../latency-overlap/RESTORE.txt if starting from the base image. Do not rerun integrated-c8/integrate.py.

Updated decode serving:

```bash
HIP_VISIBLE_DEVICES=0 HTTP_PORT=8280 PORT=8281 CONCURRENCY=8 CAPTURE_SIZES='[1,2,4,8]' bash /work/rollout-c1248/run-selected.sh fused
```

With that server running once, run each concurrency twice using the existing helper and read round2 only:

```bash
for c in 1 2 4 8; do
  bash /work/concurrency-rollout/clients.sh new-selected 8280 "$c"
done
```

Native baseline: integrated-c8/run.sh baseline. Previous monokernel: integrated-c8/run.sh fused with ATOM_DSV4_EDGE_MASK=0, ATOM_DSV4_TILE_MASK=0 and ATOM_DSV4_OVERLAP=0, and all per-concurrency overrides unset. Use separate server ports and GPU assignments, and distinct client tags.

The updated policy uses TILE128/OVERLAP0 at C1/C2 and TILE192/OVERLAP2 at C4/C8. All paths remain one full attention + shared/routed MoE kernel per layer, for all 43 eligible decode layers. Prefill stays native. Experimental fine-grained handshakes and broad extra prefetch remain disabled.

The temporary prefill-audit decorator and benchmark save-detailed default are restored after the experiment. Their backups and helper are retained here for diagnostics. The generic serving launcher does not require this instrumentation.

The microbenchmark generators copy the actual serving region bodies into standalone entry points. Initial QKV timings with duplicated coefficient loads are invalid; only final-micro-*.json and the report are the final selected measurements. Rejected flag combinations and barrier failures remain in separate logs. Exact numerical comparisons are against the previous monokernel, not a new native-ATOM accuracy qualification.
