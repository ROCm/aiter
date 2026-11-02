# All tiled GEMM weights through TDM

Read REPORT.md. Default is the new early TDM development path:

```python
from region import QKVRegion
region = QKVRegion(W, WS, NW, QB, QBS)  # mode="single", early=True
q, kv = region(x, cos, sin)
```

Use early=False for matched late QB staging. mode="global_control" selects the inherited direct-load comparison only. The old mode="two" is not implemented in this artifact; use qkv-load-poll for that control. L2 prefetch remains off. Check health after synchronization; do not overlap calls sharing state. See region.py for exact shapes/dtypes.

```bash
docker exec vigokhal_dsv4_tp4 bash -lc 'cd /work/qkv-all-tdm && ./run.sh'
```

Mount this directory into an equivalent gfx1250 environment to reproduce elsewhere. Environment versions are in scratch-benchmark/environment.json. Source and fixture are self-contained. All reported times use the second timing round; this is not a server/client benchmark. No serving integration in this artifact.
