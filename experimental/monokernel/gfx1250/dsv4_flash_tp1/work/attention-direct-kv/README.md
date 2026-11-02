# Direct KV attention experiment

Run in gfx1250 container with this folder mounted, e.g.:

```sh
docker exec vigokhal_dsv4_tp4 bash -c 'cd /work/attention-direct-kv && bash run.sh'
```

Fixture, sink, source, upstream license, and skills are included. No model load
is needed. See DESIGN.md for synchronization and fallback details, REPORT.md
for current results. Report round 2 only. `ab.py` compares otherwise identical
DIRECT=False/True kernels with reused weights; `rotating_ab.py` uses sixteen
weight copies. Older auxiliary scripts are retained for provenance, but the
current run.sh defines this experiment's checks. Baseline archive remains
attention-aiter-20261020.tgz.
