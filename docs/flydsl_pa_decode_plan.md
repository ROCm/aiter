# GPU work plans for FlyDSL PA decode

`plan_pa_decode` assigns different KV partition counts to requests according to
their GPU-resident context lengths. It is intended for batches with unequal work,
such as one long context mixed with many short contexts. It is opt-in: a planning
launch and packed reduction can cost more than static splits for uniform or short
contexts. Compare **plan + attention + reduction** against the static path.

The attention math, per-token scaling, causal MTP masking and sliding windows are
shared with the static FlyDSL kernel. The planner is a Triton GPU kernel, following
the existing variable-context scheduling approach in
`mqa_logits/pa_mqa_logits_fp4.py`.

A positive `sliding_window=W` retains at most W causal tokens **including the
query's own position**; W=1 attends only to that token. Values 0 and -1 disable the
window. Context lengths include the MTP query tokens: for context length C, query
length Q and query position p, the visible range is
`[max(0, C - Q + 1 + p - W), max(0, C - Q + 1 + p))`.

## Scheduling and storage

Each nonempty request first receives one task. Remaining task slots are apportioned
by its number of 256-token compute tiles using integer prefix sums. Counts are
capped by useful tiles and `max_partitions` (1 through 256). The workgroup budget is
a target across all KV heads; capacity is rounded up to at least one slot per
request/head. Clamping can leave unused slots. Physical page size is independent
of this scheduling tile size.

With a window, only tiles intersecting the union of the Q query windows count
toward the allocation: from `max(0, C - (Q - 1) - W) // 256` through
`ceil(C / 256)` (exclusive). Unaligned windows can touch an extra tile. The
attention kernel applies the exact per-query mask within those tiles; it does
not load the skipped prefix tiles.

The default budget is two workgroups per compute unit. For long MTP workloads,
pass the scheduler's host-known `total_context_length` and `query_length` when
creating a plan. The budget then also accounts for sequential work: it targets
about 256 query/compute-tile pairs per partition, rounded to a power of two.
For B200/C200000/MTP4/Hkv1 on a 256-CU GPU, this selects capacity 3200 and 16
partitions per request. A single long request among short requests does not
inflate the budget as if every request were long. An explicit `workgroup_budget`
takes precedence, and `max_partitions` still clamps each request's count.

These optional hints size buffers once; they do not replace GPU lengths or
perform a GPU-to-CPU copy. Without a total-length hint, or for single-query
decode, the default budget retains its previous policy.

For a windowed plan, the work hint is capped by the query-window union,
including a possible partial tile at its left edge.

The planner writes two contiguous int32 CUDA tensors:

- `work_info[capacity, 4]`: request index, absolute first tile, exclusive absolute
  last tile, original context length.
- `reduce_info[batch, 2]`: first packed task slot and actual partition count.

The launch grid has a fixed capacity for graph capture. Padded work records are
zeroed and skipped by the attention kernel, leaving their scratch slots untouched.
Each real task processes one contiguous KV range; ranges cover each request's
active tiles exactly once. Reduction reads only the valid partitions and produces
zero for empty contexts. No GPU-to-CPU readback, dynamic allocation, or CPU length
inspection is required when refreshing a preallocated plan.

Planned scratch is packed as `[num_kv_heads, capacity, query_rows]` for max/sum and
`[num_kv_heads, capacity, query_rows, head_dim]` for partial outputs, where
`query_rows = query_length * num_query_heads // num_kv_heads`.
The static API continues to use its dense per-request scratch layout.

The plan owns both kinds of partition information: `plan.num_partitions` is a
GPU tensor view with each request's actual count, while `plan.max_partitions`
is the host-side upper bound used to compile the attention and reduction
kernels. Planned calls to `pa_decode` infer that bound from `work_plan`; callers
do not need to pass `max_context_partition_num`. Existing calls that explicitly
pass a matching bound remain supported. Static calls still require an explicit
partition count.

## Attention sinks

Both static and planned `pa_decode` accept optional `sinks`: a contiguous
`[num_query_heads]` BF16, FP16 or FP32 tensor on the query device. Each entry is
a zero-value attention logit shared across sequences and MTP query positions.
It is not multiplied by `softmax_scale` and remains visible independently of
the sliding window. For head h, the output is mathematically
`sum(exp(score_i) * V_i) / (sum(exp(score_i)) + exp(sinks[h]))`.

The sink participates in the stable softmax maximum and contributes to the
denominator exactly once, not once per partition. Static NP=1 handles it in
the compute epilogue without an extra launch; multi-partition and planned calls
add it in the final reducer. KV partials and FP8 Q/P quantization are unchanged.
Empty contexts or query rows with no visible KV tokens still produce zero.
`-inf` disables an individual head's sink; `+inf` produces zero for that head.

Sinks do not change work ownership, so they require no planner arguments or new
metadata. Their values can be updated in place between graph replays without
rebuilding a plan, provided the context lengths remain unchanged.

## Usage

Use the Python FlyDSL entry point for planned execution. The shared
`torch.ops.aiter.pa_decode_flydsl` entry point retains its existing static interface.

```python
import torch
from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

# query/output: [batch * query_length, num_query_heads, head_dim]
# K/V cache layouts and scales are the same as for static pa_decode.
num_kv_heads = key_cache.shape[1]
sliding_window = 4096  # Use 0 or -1 for full causal attention.
plan = plan_pa_decode(
    context_lengths,
    num_kv_heads,
    # Optional: use a total already known by the CPU request scheduler.
    # Omit this hint if only the GPU knows the context lengths.
    total_context_length=host_total_kv_tokens,
    query_length=query_length,
    sliding_window=sliding_window,
)
rows = query_length * query.shape[1] // num_kv_heads
shape = (num_kv_heads, plan.capacity, rows)
exp_sums = torch.empty(shape, dtype=torch.float32, device=query.device)
max_logits = torch.empty_like(exp_sums)
partials = torch.empty((*shape, query.shape[2]), dtype=query.dtype, device=query.device)
sinks = torch.zeros(query.shape[1], dtype=torch.float32, device=query.device)
# Use sinks=None for attention without the extra normalization logit.

def step():
    # Refresh after changing lengths, on the same stream as attention.
    plan_pa_decode(
        context_lengths, num_kv_heads, plan=plan,
        sliding_window=sliding_window, query_length=query_length,
    )
    pa_decode(
        output, query, key_cache, value_cache, context_lengths, block_tables,
        softmax_scale, query_length,
        compute_type=key_cache.dtype,
        key_scale=key_scale, value_scale=value_scale,
        exp_sums=exp_sums, max_logits=max_logits, temporary_output=partials,
        work_plan=plan, sliding_window=sliding_window, sinks=sinks,
    )

step()  # Warm up compilation before graph capture.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    step()
# Update existing input buffers, then graph.replay().

# Optional diagnostics: this is a live GPU view, with no host readback.
counts = plan.num_partitions
```

New plans default to a partition upper bound of 256. To impose a smaller bound,
pass `max_partitions` when creating the plan. Refreshing with `plan=plan`
inherits that bound, so it does not need to be repeated. An explicitly supplied
bound must match the reused plan. Actual counts continue to be computed on the
GPU from the current context lengths; reading them on the CPU is unnecessary
for attention or graph replay.

Reuse an already built plan without refreshing only when lengths are unchanged.
Reuse its buffers and refresh the metadata when lengths change, even if no new
256-token tile is needed: each task also stores its exact causal context length.
The total-length hint is only accepted at creation; the capacity remains fixed
on refresh. If the workload grows substantially, a new capacity can improve
performance, while the existing capacity remains correct.
Changing batch size, KV-head count, the partition limit or launch capacity requires
a compatible new plan and scratch buffers. A plan's window is fixed, as is its
query length when the window is enabled; changing either requires a new plan.
Dense plans are independent of query length. The attention call checks these
settings to avoid silently using a plan that omits visible tokens.
Metadata values should be produced by the planner, not edited independently.
The planner accepts contiguous int32 CUDA
length vectors with 1 through 4096 requests. Lengths and block-table indices must
satisfy the same validity requirements as static attention.

## Verification

```bash
python -m pytest -q op_tests/test_flydsl_pa_decode.py
python -m op_tests.benchmark_flydsl_pa_decode_plan --output plan_results.json

# B200/C200000/MTP4, single TP rank: Hq16/Hkv1/D128.
HIP_VISIBLE_DEVICES=7 python -m op_tests.benchmark_flydsl_pa_decode_plan \
    --cases b200_uniform --block-size 16 128 --trans-v 0 1 \
    --num-iters 51 --num-warmup 5 --num-rotate-args 3 \
    --output b200_results.json
```

The benchmark uses the repository timer with allocation rotation on identical
inputs, checks the planned output against a causal FP32 reference, and reports
static, plan-per-call, reused-plan and standalone-plan timings separately. Static
selection uses the host-known maximum length and query length. Plan creation
uses the host-known sum of lengths and query length. All timed plan refreshes
reuse the allocated buffers.

`--num-partitions` overrides the static partition count exactly;
`--workgroup-budget` controls the plan's total CTA budget; `--max-partitions`
controls the plan's per-request upper bound independently of static splits.
`--num-iters`, `--num-warmup`, and `--num-rotate-args` expose the repository
timer's settings. Zero allocation sets selects its automatic rotation policy.

Each attention mode reports microseconds and effective bandwidth in decimal
GB/s and TB/s. The byte count includes Q/O, valid K/V tokens and their scales,
referenced page-table entries, and context lengths once per rank. It excludes
padding, sparse holes, repeated loads, scratch and plan metadata. This measures
logical bandwidth, not hardware-counter HBM traffic; KV bytes are not multiplied
by MTP or TP. `plan_each_call` includes planner GPU time in the denominator.

The uniform and one-long sweeps cover batches 1/2/4/8/16/200. Uniform contexts
have 200000 tokens; the existing one-long cases use 200003 and 257. Other cases
include B64 with equal 100000-token contexts, B32 one-long, a ramp, and short
contexts. All use BF16 MTP4, Hq16/Hkv1/D128, per-token FP8 scales, page16/page128
and both V layouts.

Tests cover exact absolute-tile ownership, sparse pages, empty requests, causal tails,
non-power-of-two partition caps, multiple KV heads, supported reducer branches,
NaN-filled scratch, and graph replay after changing lengths. Window tests include
W=1, non-aligned boundaries, windows larger than the context, QL1/2/3/4, per-token
and per-tensor scales, static/query-split execution and incompatible plan reuse.
Sink tests cover per-head logits, direct output and all reducer paths, mixed
input/sink dtypes, large and infinite logits, window composition, and graph
replay after updating sink values.
