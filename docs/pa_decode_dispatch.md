# Prepared paged-attention decode dispatch

`prepare_pa_decode_plan`, `refresh_pa_decode_plan`, and `pa_decode` provide one
output-only entry for ASM and FlyDSL. Existing `pa_persistent_fwd` and direct
FlyDSL callers keep their existing behavior; callers adopt this entry explicitly.

## Calling the entry

```python
import math
from aiter import prepare_pa_decode_plan, refresh_pa_decode_plan, pa_decode

# Q: [B * 4, 16, 128], BF16.
# Packed K: [pages, 1, 8, 16, 16], packed V: [pages, 1, 1, 128, 16], FP8 E4M3.
# Lengths and block_table: contiguous GPU int32 [B] and [B, capacity_pages].
# K/V scales: contiguous FP32 [pages, 1, 16] or [pages, 1, 16, 1].
# host_lengths is an existing scheduler-owned sequence of Python integers.
scale = 1.0 / math.sqrt(128)
plan = prepare_pa_decode_plan(
    q, k, v, lengths, block_table, 4,
    backend="auto",
    context_lengths_cpu=host_lengths,
    key_scale=k_scale, value_scale=v_scale,
    softmax_scale=scale,
    max_context_length=block_table.shape[1] * 16,
    # Set False only if unused KV positions and their scales are finite.
    # The safe default True retains FlyDSL.
    mask_unowned_kv=False,
)
print(plan.backend, plan.fly_workgroup_budget)

# Update lengths/table contents in place, then refresh once per scheduling step.
refresh_pa_decode_plan(plan)
pa_decode(out, q, k, v, lengths, block_table, scale, 4, plan=plan,
          key_scale=k_scale, value_scale=v_scale)
# Other layers with the same tensor geometry can reuse this step's plan.
```

If packed page indices already exist, pass both `kv_indptr` and `kv_indices` to
prepare. They must represent exactly `ceil(length / 16)` pages per sequence in
logical order. The caller updates them before refresh. With only a rectangular
block table, the ASM path runs two GPU conversion kernels on every refresh;
these costs are included in the dispatch measurements.

Prepare allocates workspaces outside graph capture. Refresh and decode can be
captured. GPU metadata addresses stay fixed and their contents may be updated
between replays. The backend and reducer remain fixed for a plan/graph: select
another prepared plan and capture outside replay when the CPU policy changes.
The caller must order streams and use separate workspaces for concurrent calls.
Host hints must describe the same step as GPU lengths; the dispatcher does not
read back GPU tensors to verify this association.

## Automatic policy

The performance policy was calibrated on MI355X, gfx950 with 256 CUs, with BF16
Q4, 16 query heads, one KV head, D128, page16, per-token FP8 scales, dense causal
attention, and the packed contiguous layout above. ASM additionally requires
the default `1/sqrt(D)` scale and finite unused KV/scales (`mask_unowned_kv=False`).
Query scaling, ALiBi, sinks, and sliding-window attention retain FlyDSL.

Within this configuration, all lengths must be at least four, the longest at
most 786,432 tokens, and the total at most 6,291,456 tokens. The thresholds below
are **token counts**, not cache sizes or byte counts.

| Input metadata | Batch | Total context tokens | Automatic backend |
|---|---:|---:|---|
| Existing packed CSR | 24 or 32 | At least 3,145,728 | ASM |
| Rectangular table requiring conversion | 32 | At least 4,194,304 | ASM |
| Other measured or uncalibrated cases | Any | Any | FlyDSL |

Missing CPU length hints, unknown/non-256 CU devices, or explicit FlyDSL
`max_partitions`/`workgroup_budget`/different `compute_type` retain FlyDSL.
The caller can force `backend="asm"` or `backend="flydsl"`; unsupported forced
ASM configurations raise an error. Compiler and execution errors are never
silently retried with another backend. If only the optional top-level FlyDSL
package is missing, auto may use ASM for its supported contract with host hints;
this availability fallback is distinct from the performance policy.

For compatible FlyDSL calls without explicit budget/partition overrides, the
entry uses workgroup budget 512 for B8/16/24/32 when the total is at least
`B * 32768`. B4 additionally requires longest KV at least 65,536 and total at
least 262,144. The same calibration limits apply. Otherwise it preserves the
native budget. `plan.fly_workgroup_budget` records the requested tuning; inspect
the FlyDSL work plan for its actual partition/task counts.

## Evidence and limits

The base is AITER main `47d23011bde636eb16613a21156f47836ec7b65e` with FlyDSL
0.5.0. The packaged optimized ASM CO has SHA256
`39753eeed171d4cf1caaa28476e1992659c729495306e9b562babd889aed1f50`.
Calibration uses the ten original Q4 shapes, uniform 64K/128K neighbors, and
held-out mixtures of long and short KV lengths. Each comparison has three
independent process seeds, random physical pages, three independent KV pools,
and complete refresh plus three-layer output-only GPU graph timing. Both
FlyDSL budgets 256 and 512 are retained; ASM is compared against the faster
same-device, same-seed control. Measurements on different physical GPUs are
not pooled into an absolute latency average.

The smallest selected per-step/process margins in these calibration datasets
are approximately 1.19% for packed CSR and 1.39% for table conversion. These
are observed margins, not guarantees for all unseen distributions, models,
layer counts, concurrent workloads, or hardware. Recalibrate when these change.
The table/CSR crossover includes refresh amortization over three layers; it is
not a producer-only kernel threshold.

The existing main and candidate ASM kernels both propagate NaNs from some
unused tail V/scales. The default mask keeps such inputs on FlyDSL. Supplying
finite padding is a caller contract when opting into ASM, not something inferred
from the host length hints. This interface does not return final LSE.

The new multi-entry reducer CO and wave selection still need GPU revalidation
before these historical crossover thresholds can be considered recalibrated.
