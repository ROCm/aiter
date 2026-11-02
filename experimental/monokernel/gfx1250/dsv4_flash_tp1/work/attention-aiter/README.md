# AITER-derived attention and QKV integration

Read REPORT.md first. Source provenance and MIT license: UPSTREAM.json and upstream/.

```python
from region import QKVAttentionRegion
r = QKVAttentionRegion(W, WS, NW, QB, QBS, cache, indices, sink,
                       fused=True)  # one launch; TDM + L2 prefetch enabled
attention_output = r(x, cos, sin)
```

`fused=False` uses two regions and is currently faster. `AttentionRegion` is the standalone cache-update + attention region. Default heads/tile16, KV tile32; automatic split1 below1024 slots, split4 from1024. Override `splits=1/2/4` only when every split has a tile. Cache is contiguous BF16[pages,512]; indices contiguous int32[T] (-1 or valid physical row); sink FP32[64]. Default write_slot is pages-1. QKV shapes/dtypes in qkv_region.py. Outputs are reused. No overlapping calls sharing state; synchronize before checking `r.check()`. Runtime index changes must retain bounds validity.

Reproduce in the existing container, with an idle GPU and no concurrent persistent grids:

```bash
docker exec vigokhal_dsv4_tp4 bash -lc 'cd /work/attention-aiter && ./run.sh'
```

The directory is mounted from /home/vigokhal/dsv4-monokernel/attention-aiter. A different host needs an equivalent gfx1250 Gluon/TDM-capable environment (environment.json). Fixture and sink.pt are included: no /data model dependency for replay. No AITER Python import is needed. First timing round is ignored; report.py uses second-round data only. Results files are overwritten on reproduction; copy the artifact before tuning.

Important files: aiter_adapt.py (adapted upstream), attention.py (cache publication, last-arriver merge, single-launch composition), region.py (API), qkv_region.py and associated projection sources, validate.py/stress.py, bench.py/tune.py/rotating_bench.py, trace.py and report.py.

This is a tested kernel-region integration, not an ATOM serving integration or full attention-layer/MoE monokernel. Native FP8/quantized-cache and compressed-cache paths are outside this artifact.
