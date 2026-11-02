# QB prefetch experiment

Read REPORT.md for measurements and limitations. This extends qkv-load-poll; all code and the fixture needed for this region are included here.

```python
from region import QKVRegion
region = QKVRegion(W, WS, NW, QB, QBS, mode="single",
                   prefetch="non_speculative", stage_first_tile=True)
q, kv = region(x, cos, sin)
# After stream synchronization, health must be zero.
```

prefetch options: "none", "speculative", "non_speculative". TDM independently selectable with stage_first_tile. Default keeps both off due to rotating-weight regression. mode="two" is the prior matched control and rejects staging options. No overlapping invocations sharing state. Full input/output shapes and device requirements in region.py. No full attention, MoE, KV cache update or serving integration in this region.

Reproduce inside existing image/container:

```bash
docker exec vigokhal_dsv4_tp4 bash -lc 'cd /work/qkv-qb-prefetch && ./run.sh'
```

On another host, mount this directory and use the gfx1250-capable software stack recorded in scratch-benchmark/environment.json; see the parent handoff for image context. GPU must be idle before persistent-grid experiments. The script takes two timing rounds, retaining only the second for reported comparisons. It does not change clocks or launch a server.
