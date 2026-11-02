# L2 prefetch + all-weight TDM development default

```python
from region import QKVRegion
r = QKVRegion(W, WS, NW, QB, QBS)  # early=True, prefetch=True
q, kv = r(x, cos, sin)
```

Read REPORT.md for historical comparison and limitations. prefetch=False is an ablation, early=False delays QB TDM until after polling. mode="global_control" disables tiled TDM and prefetch for the benchmark control. No old mode="two" here. Read region.py for shapes/dtypes. Do not overlap invocations sharing state; check health after stream synchronization.

```bash
docker exec vigokhal_dsv4_tp4 bash -lc 'cd /work/qkv-all-tdm-l2 && ./run.sh'
```

Use an idle GPU. Only round2 performance is reported. Software versions in scratch-benchmark/environment.json. This artifact is isolated QKV preparation, not an ATOM server integration.
