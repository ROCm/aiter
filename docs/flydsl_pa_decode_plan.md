# GPU work plans for FlyDSL PA decode

`plan_pa_decode` assigns different KV partition counts to requests according to
their GPU-resident context lengths. It is intended for batches with unequal work,
such as one long context mixed with many short contexts. It is opt-in: a planning
launch and packed reduction can cost more than static splits for uniform or short
contexts. Compare **plan + attention + reduction** against the static path.

The attention math, per-token scaling and causal MTP masking are shared with the
static FlyDSL kernel. The planner is a Triton GPU kernel, following the existing
variable-context scheduling approach in `mqa_logits/pa_mqa_logits_fp4.py`.

## Scheduling and storage

Each nonempty request first receives one task. Remaining task slots are apportioned
by its number of 256-token compute tiles using integer prefix sums. Counts are
capped by useful tiles and `max_partitions` (1 through 256). The workgroup budget is
a target across all KV heads; capacity is rounded up to at least one slot per
request/head. Clamping can leave unused slots. Physical page size is independent
of this scheduling tile size.

The planner writes two contiguous int32 CUDA tensors:

- `work_info[capacity, 4]`: request index, first tile, exclusive last tile, context length.
- `reduce_info[batch, 2]`: first packed task slot and actual partition count.

The launch grid has a fixed capacity for graph capture. Padded work records are
zeroed and skipped by the attention kernel, leaving their scratch slots untouched.
Each real task processes one contiguous KV range; ranges cover each request exactly
once. Reduction reads only the request's valid partitions and produces zero for
empty contexts. No GPU-to-CPU readback, dynamic allocation, or CPU length
inspection is required when refreshing a preallocated plan.

Planned scratch is packed as `[num_kv_heads, capacity, query_rows]` for max/sum and
`[num_kv_heads, capacity, query_rows, head_dim]` for partial outputs, where
`query_rows = query_length * num_query_heads // num_kv_heads`.
The static API continues to use its dense per-request scratch layout.

## Usage

Use the Python FlyDSL entry point for planned execution. The shared
`torch.ops.aiter.pa_decode_flydsl` entry point retains its existing static interface.

```python
import torch
from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

# query/output: [batch * query_length, num_query_heads, head_dim]
# K/V cache layouts and scales are the same as for static pa_decode.
num_kv_heads = key_cache.shape[1]
plan = plan_pa_decode(context_lengths, num_kv_heads, max_partitions=256)
rows = query_length * query.shape[1] // num_kv_heads
shape = (num_kv_heads, plan.capacity, rows)
exp_sums = torch.empty(shape, dtype=torch.float32, device=query.device)
max_logits = torch.empty_like(exp_sums)
partials = torch.empty((*shape, query.shape[2]), dtype=query.dtype, device=query.device)

def step():
    # Refresh after changing lengths, on the same stream as attention.
    plan_pa_decode(context_lengths, num_kv_heads, max_partitions=256, plan=plan)
    pa_decode(
        output, query, key_cache, value_cache, context_lengths, block_tables,
        softmax_scale, query_length, plan.max_partitions,
        compute_type=key_cache.dtype,
        key_scale=key_scale, value_scale=value_scale,
        exp_sums=exp_sums, max_logits=max_logits, temporary_output=partials,
        work_plan=plan,
    )

step()  # Warm up compilation before graph capture.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    step()
# Update existing input buffers, then graph.replay().
```

Reuse an already built plan without refreshing only when lengths are unchanged.
Changing batch size, KV-head count, the partition limit or launch capacity requires
a compatible new plan and scratch buffers. Metadata values should be produced by
the planner, not edited independently. The planner accepts contiguous int32 CUDA
length vectors with 1 through 4096 requests. Lengths and block-table indices must
satisfy the same validity requirements as static attention.

## Verification

```bash
python -m pytest -q op_tests/test_flydsl_pa_decode.py
```

Tests cover exact work ownership, sparse pages, empty requests, causal tails,
non-power-of-two partition caps, multiple KV heads, supported reducer branches,
NaN-filled scratch, and graph replay after changing lengths.
