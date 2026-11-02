# Optimized gfx1250 mHC and one-launch prefix

See REPORT.md for accepted second-round measurements, selection, accuracy and
rejected variants. Main source: tdm_tail.py (mHC), optimized_prefix.py (prefix),
setup.py (research harness). full_prefix.py remains the old fused control.

```sh
docker exec -e HIP_VISIBLE_DEVICES=0 vigokhal_dsv4_tp4 bash -c 'cd /work/mhc-optimized && bash run.sh'
```

Use an otherwise idle GPU with all256 gfx1250 CUs available for grid barriers.
This runs isolated benchmarks, not a model server. All three required fixture
files are included. Copy the directory before replay: results are overwritten.
The validation/ directory also preserves independent GPU1/GPU3 acceptance data.
The report generator uses those archived independent checks plus current GPU0
results. Experimental scripts under experiments/ share the same fixture files
via relative symlinks. No workload is automatically resumed from this artifact.
