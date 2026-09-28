# FlyDSL PA decode tuning

`aiter/ops/flydsl/pa_decode_tuning.py` uses FlyDSL's `Autotuner`, `Config`,
`do_bench`, and native configuration cache to select a workgroup budget.
Use the repository's FlyDSL dependency (0.3.4.1 or newer with `validate_hook`
and `select_config`).

Run the tuner on the device that will execute PA decode, using device-local
head counts:

```bash
export FLYDSL_AUTOTUNE_CACHE_DIR=/tmp/pa-decode-autotune
python aiter/ops/flydsl/pa_decode_tuning.py \
  --shape 16,1,128 --batch-sizes 1,12,32 \
  --context-lengths 4096,16384 --budget-cu 0.5,1,2,4
```

A cache miss measures candidates and persists the winner. A hit reuses the
budget without allocating benchmark inputs or measuring again. Set
`FLYDSL_AUTOTUNE=1` for a fresh search, including after changing candidate
budgets or benchmark sampling options. Without an override, FlyDSL stores the
cache in `~/.flydsl/autotune/_pa_decode_budget.json`.

`--output measurements.csv` optionally writes a new measurement report.
It contains freshly measured candidates; cache hits have no new timing rows.
`--help` and `--dry-run` require only Python. For a CPU preview, supply
`--dry-run --architecture gfx950 --num-cu 256`.

The tuner deduplicates budgets by plan capacity, always includes the `2*CU`
baseline, and checks eager and graph outputs against streamed FP32 attention.
Each candidate is measured repeatedly using FlyDSL event timing. A graph
contains `--iterations` decode-plus-reduction calls; `--rounds` controls repeated
measurements of each candidate. The selected budget is the smallest one with
at least 97% of the fastest candidate's measured performance. Measurements are
per candidate, rather than interleaved rounds across candidates.

For inference, query the cache before building the required explicit plan.
The following example assumes tensors matching BF16 queries, per-token FP8 KV,
transposed V, B=12, L=16384, Q=4, Hq=16, Hkv=1, D=128, and page size 128:

```python
import torch
from aiter.ops.flydsl.pa_decode import plan_pa_decode
from aiter.ops.flydsl.pa_decode_tuning import (
    get_cached_budget, make_shape, storage_key,
)

shape = make_shape(
    12, 16384, 4, num_query_heads=16, num_kv_heads=1, head_dim=128,
)
with torch.cuda.device(query.device):
    props = torch.cuda.get_device_properties(query.device)
    budget = get_cached_budget(
        shape, props.gcnArchName.split(":")[0], props.multi_processor_count,
        storage_key=storage_key(query, key_cache, value_cache, key_scale, value_scale),
    )
    plan = plan_pa_decode(
        context_lengths, 1, workgroup_budget=budget, query_length=4,
    )
```

Pass `work_plan=plan` and the same host context bound,
`max_context_length=16384`, to `pa_decode`. Refresh the plan after changing
lengths; allocate scratch before graph capture. Decode and graph replay do not
access the tuning cache. `get_cached_budget` returns `2*CU` on a cache miss,
an invalid cached budget, or when `FLYDSL_AUTOTUNE=1` is set; it never launches
an online search. The measured shapes exclude sinks and use `head_dim**-0.5`
as the softmax scale.

The cache key includes static attention geometry, architecture, CU count,
tensor layout/address width, PA source contents, and FlyDSL's toolchain and
codegen environment fingerprint. Pass actual tensor storage metadata for
lookup; 3-D and trailing-singleton 4-D per-token scales share a key. Omitting
`storage_key` assumes the tuner's synthetic contiguous allocation.

FlyDSL reads disk entries when constructing its tuner. Finish offline tuning
before starting inference processes so they see those entries. Tuning through
this module also updates an existing resolver in the same process. Use
separate cache directories for parallel tuning processes because native cache
writes do not merge concurrent writers.

The previous CSV configuration APIs (`load_tuned_results`,
`PADecodeTunedResults`, and the tuner's CSV lookup/load/save helpers) are
replaced by `get_cached_budget` and FlyDSL's cache. Existing CSV reports are
not imported as configurations; rerun offline tuning to populate the cache.
