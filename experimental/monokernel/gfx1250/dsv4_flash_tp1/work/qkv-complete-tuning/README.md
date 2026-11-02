# Follow-up Q/KV tuning

Read REPORT.md for measurements and FLOW.md for the detailed ownership table and ASCII diagram. This is a separate artifact; the previous qkv-complete-ablation directory/archive is unchanged.

Recommended isolated API: region.QKVRegion(W,WS,NW,QB,QBS,mode='two'), called with BF16input4096 and FP32cos/sin tables32. The docstring specifies ABI. One state per stream, no overlapping calls sharing state. Outputs are reused. It uses QA192CTAs as before, followed by QB256CTAs/N128/Kunroll4. The single-launch option remains the previous unroll1 version; the new high-register QB variants have NOT been transplanted into a persistent single-launch grid.

All required layer0 region weights/inputs are in fixture.pt. No full checkpoint is required. Scripts run from this directory with HIP_VISIBLE_DEVICES=0 in rocm/fw-bringup-mlperf:gfx1250-atom-20261015 (existing vigokhal_dsv4_tp4 container):

```
python validate.py
python bench.py
python check_api.py
python stress.py
```

Check GPU idle first. Reruns overwrite results. Use second timing round. selected.py defines prior/unroll2/unroll4/fullunroll8 comparisons. harness.py supplies independent FP64 reference with exact precision boundaries. sweep.py retains22initial geometry/unroll configurations; main comparisons are final-benchmark/timings.json.

No ATOM serving change, image publication, or branch push. Timing is C1 isolated Q/KV preparation, not an end-to-end layer or serving benchmark. Rotary tests use controlled tables, not a live YaRN cache. region.py single mode requires256available CUs; inspect state.health after synchronization.
