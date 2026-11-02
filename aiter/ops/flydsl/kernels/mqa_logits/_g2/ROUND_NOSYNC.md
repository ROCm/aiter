# ROUND-NOSYNC: sync-free int64 metadata narrowing

Base: champion `1429dd28` (kernel sha `f10c6cc9…`). Branch `qsa-index-scorer/opt-nosync`.

## Problem (measured a08-1 GPU3, rocprofv3)

Kernel-only decode bf16 c65536: ours 29.2 µs vs ATOM 38.6 µs, but the event-timed
wrapper call is ~74 µs. Golden fixtures pass **int64 `query_positions`**; the
historical `_to_i32_checked` ran `int(t.min())`, `int(t.max())` (2 reductions +
2 D2H copies + **2 host synchronizations**) and `t.to(int32)` per int64 tensor.

## Change (wrapper only — option (b); kernel builders untouched)

* `_to_i32_saturated(t, hi)`: int32 → returned as-is (zero copy). int64 → one
  `torch.clamp(t, lo_t, hi_t, out=<int32>)` (single TensorIterator launch with the
  int64→int32 cast on store; no host sync). Bounds are cached 0-d device tensors
  created once per (device, stream, hi) with `torch.full` (capture safe).
* Positions saturate at `INT32_MAX-1` so the kernel's i32 `pos + 1` never wraps
  (fixes the historical int64 `pos = INT32_MAX` → visible 0 bug for int64 input).
* `QSA_STRICT_I32=1` (read once at import) restores the historical
  `_to_i32_checked` path exactly (raise on out-of-int32 int64; host syncs).
  `_to_i32_checked` itself is byte-identical.
* int32 `query_positions == INT32_MAX` keeps its historical behaviour (zero-copy
  path; `QSA_PAD_INT32_GUARD=1` diagnostic unchanged).

## Semantics proof (summary; full text in `_to_i32_saturated` docstring)

Plane caps give rows/columns/requests/pages < 2**29 and every column index
≤ 2**29-2. Request/page validity `0 <= x < N` is unchanged by saturation (in-range
values fixed; > INT32_MAX → INT32_MAX ≥ N; < INT32_MIN → INT32_MIN < 0). For
`V = max(0, min((p+1)//4, c//4))` the saturated `V' = min(V, 2**29-1)`: equal for
every V < 2**29 (all inputs the strict contract accepted), and the logits masks
`col < V` / `col < V'` are identical for every addressable column. Low-word
aliases (e.g. page `2**32+1`, request `2**32`) stay invalid (no wrap).

## Per-call GPU work (default env, PAD unset; all routes share the preamble)

| metadata dtypes | before | after |
|---|---|---|
| all int32 | main kernel | main kernel |
| golden (qp int64) | 2×reduce + 2×D2H copy (2 syncs) + 1 cast + main | 1 clamp/cast + main, 0 syncs |
| all four int64 | 8 reduce + 8 D2H (8 syncs) + 4 cast + main | 4 clamp/cast + main, 0 syncs |

First call per (device, stream) additionally fills two 0-d bound tensors per `hi`
(one-time). Routes: bf16 r16 TDM decode/draft, r32 TDM/persist prefill, r64,
fp8 r16/r32, scalar — identical preamble. `QSA_PAD_BUCKET=1` keeps its opt-in
padding kernels (unchanged).

## Verification (no device)

`test_nosync_i32_cpu.py` (boundary matrix × 4 tensors vs golden formula, real
wrapper + model vs `qsa_golden.reference_logits` with int64 boundary metadata for
bf16/fp8, TorchDispatchMode census: 0 ops for int32, exactly one
`aten::clamp.Tensor_out` per int64 tensor, no `_local_scalar_dense`; strict mode).
Container gates: `host_nosync_cpu.sh STAGE cpu`; dispatch ISA identity:
`host_nosync_cpu.sh STAGE isa` (base vs final, i32/qp64/all64 metadata).
