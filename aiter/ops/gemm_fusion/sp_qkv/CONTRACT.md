# `sp_qkv_fused` v1 — AITER contract sheet (P0)

Status: **draft contract for the provider-architecture refactor of PR #6689.** AITER owns
this sheet; a provider (or the native runtime) implements it and does not redefine it. A
breaking semantic change requires an op-contract version bump (`sp_qkv_fused` v2), not a
silent change here.

Source of truth for the current native behaviour: the `aiter.gemm_fusion`/`sp_qkv`
framework at the PR head, its `CONTRACT`-levant checks in
`aiter/ops/gemm_fusion/sp_qkv/inputs.py`, and the `#6689` evidence record. Public module:
`aiter/ops/gemm_fusion/sp_qkv/op.py`. Native runtime entry point:
`aiter.ops.gemm_fusion.sp_qkv.runtime.sp_qkv_native(payload: dict) -> None` (runtime owned by
the runtime implementor). The provider interface (`aiter.plugin`) is **not merged at this
PR head**; the op decorator is guarded and the interface stack (#7465/#7466/#7484) is the
follow-up dependency.

## 1. Op name / contract version

| field | value |
|---|---|
| public op name | `sp_qkv_fused` (backend-free: semantic SP4 QKV, no `_hip`/`_asm`/`_triton` suffix) |
| contract version | **1** |
| Python module | `aiter/ops/gemm_fusion/sp_qkv/op.py` |
| public export | `aiter.sp_qkv_fused` (lazy; `from aiter.ops.gemm_fusion.sp_qkv.op import sp_qkv_fused` also works) |
| routing | `@route(op="sp_qkv_fused", version=1)` when `aiter.plugin` is present; identity shim otherwise (documented in the module) |
| native implementation | `runtime.sp_qkv_native(payload)` = today's fused framework path; provider implementations are out-of-tree and must match this sheet |

## 2. Public signature (exact)

```python
sp_qkv_fused(
    q, x, x_scale, w, w_scale, q_norm_w, k_norm_w, cos_cache, sin_cache,
    positions, slot_mapping, k_cache, v_cache, k_scale, v_scale,
    *,
    mode,
    k=None, v=None, index_q=None,
    index_q_norm_w=None, index_k_norm_w=None, index_cache=None,
    eps=1e-6, group=None, stream=None,
) -> None
```

Canonicalized form handed to a provider by the real interface: the 15 required
positional-or-keyword parameters arrive as a positional tuple (with defaults applied);
everything after `*` arrives as a read-only keyword mapping. No `*args`/`**kwargs`, no
dict-shaped tensor bags, no `res`/`profile`/`bucket`/`variant`/`call_id` policy arguments.

| parameter | meaning |
|---|---|
| `q` | output Q, bf16 `[T, 16, 128]`, written in place |
| `x` | activation, fp8 e4m3fn `[M, 6144]`; row-strided view allowed (`stride(0)` as in `inputs.check_activation`) |
| `x_scale` | fp32, `[M]` or `[M, 1]`, contiguous |
| `w` | fp8 e4m3fn **(16,16)-preshuffled**, contiguous `[rows >= N, 6144]` |
| `w_scale` | fp32, `[N]` or `[N, 1]`, contiguous, rows may be >= `N` |
| `q_norm_w`, `k_norm_w` | fp32 `[128]` Gemma-RMSNorm weights |
| `cos_cache`, `sin_cache` | bf16 `[max_pos, 32]` (rotary dim 64 = 32 pairs) |
| `positions`, `slot_mapping` | int64, at least `T` entries; pad rows are position 0 / slot `-1` and skip cache writes |
| `k_cache`, `v_cache` | fp8 golden-shuffle KV caches; page 128 for dense, page 16 for indexer/reuse |
| `k_scale`, `v_scale` | fp32 KV cache scales |
| `mode` | **keyword-only, required**: `"dense"` \| `"indexer"` \| `"reuse"` |
| `k`, `v` | bf16 `[T, 1, 128]` outputs, required for dense/reuse |
| `index_q` | bf16 `[T, 1, 128]` output, required for indexer |
| `index_q_norm_w`, `index_k_norm_w` | fp32 `[128]`, required for indexer |
| `index_cache` | fp8 flat page-128 index cache, required for indexer |
| `eps` | Gemma-RMSNorm epsilon, default `1e-6` |
| `group` | torch.distributed ProcessGroup of the SP group (execution context; world = 4) |
| `stream` | current stream (execution context; the op never creates a stream) |

`M` = local token count = `x.shape[0]`; `T` = `4 * M` for the SP world 4 of the covered
coverage. `N` = 9216 for dense, 9856 for indexer and reuse; `K` = 6144 hardwired by the
kernels. `reuse` keeps the indexer GEMM N but drops `index_q`/`index_k` in the exchange
(compact F = 2304), so its indexer-only tensors are `None`.

## 3. Dtypes / shapes / layouts (authoritative)

The kernels hardwire `K = 6144`, a compile-time `N` and `T = 4*M`; violating a layout is a
silent wrong result, so the native path rejects it in `inputs.py` before binding/launch.

- `x`: 2-D fp8 e4m3fn, `K`-contiguous, `stride(1) == 1`, `stride(orbitrary) >= K`; a
  K-padded activation is not supported (pass `x[:, :6144]` as a row-strided view).
- `w`: fp8 e4m3fn, contiguous, exactly `K = 6144` columns, rows `>= N`, and
  `(16,16)`-preshuffled (no K-padding, no unshuffled weight).
- `x_scale`: fp32 with exactly `M` contiguous elements, `[M]` or `[M,1]`.
- `w_scale`: fp32 contiguous, `[N]` or `[N,1]`, rows `>= N`.
- `positions`/`slot_mapping`: int64 contiguous; length may exceed `T`; each may be shorter
  than `T` only in the small-plan v3 per-launch contract, where the kernel pads missing
  rows as position 0 / slot `-1`; every output still has `T` rows.
- `q`, `k`, `v`, `index_q`: bf16 outputs with at least `T` rows; the exact row layout
  above is the golden case configuration.
- Caches: golden SHUFFLE layout; `k_scale`/`v_scale` fp32.

## 4. Outputs and mutation

- Returns **`None`**.
- Writes `q` in place; writes `k`/`v` for dense/reuse, `index_q` for indexer (and the
  index cache), plus `k_cache`/`v_cache`/`index_cache` for tokens whose `slot_mapping >= 0`
  (slot `-1` skips cache writes).
- Does **not** allocate or return workspace, scales, caches or metadata.
- Aliasing: the caller owns all tensors; outputs must be distinct from inputs where the
  kernels assume so. Moving a bound pointer after `prepare` is an error (see §scale/lifetime).

## 5. Tolerance and determinism

- Reference: the golden MiniMax-M3 `case_config` outputs used by `#6689` (Gate 2
  byte-exactness). The covered fused native cells are certified **byte-exact** (Gate 2:
  108/108 byte-exact; the four ABI-7 tag seams are negative controls that bite). A
  provider that serves a contract cell must reproduce the same bytes; a provider that
  changes numerics must declare an op-contract version bump rather than silently relaxing
  the tolerance.
- Determinism: after `prepare`/bind, the launch sequence is fixed; all ranks run identical
  launch counts (epochs are device-owned). Graph replay is deterministic. No relaxed
  tolerance is currently offered for the covered cells.
- Out-of-coverage calls are not a tolerance question: they are a coverage error today
  (§8) until option A lands.

## 6. Workspace / lifetime / graph capture

- The public op has **no workspace/resource argument**. The native runtime (or provider)
  owns the `FusionResources`: MORI CCO comm, registered VM windows (default profile
  468 MiB/rank; r19 alternate profile 788 MiB/rank), devcomm, ERR/STEP/DIAG page and the
  reset/gate machinery. None of that is caller state.
- **`prepare`/bind must run outside graph capture.** The caller must complete at least one
  warm-up call outside capture before capturing; a first call inside capture is an error.
  Pointers for all bound tensors (scales, outputs, metadata including caches) must not move
  after binding; `x`/`w` are per-launch kernargs and may vary.
- Resource lifetime is process/device/group scoped; teardown/reset is the implementation's
  responsibility (the provider interface has no lifecycle hook yet — a named interface gap,
  see skill §7). The op itself never allocates device workspace.
- `group` is execution context, not policy: it selects the SP world (4) and the runtime's
  per-group resource session.

## 7. Error behaviour

| condition | behaviour |
|---|---|
| `mode` not in `{"dense","indexer","reuse"}` | `ValueError` from the public op before any runtime/provider call |
| mode-specific required tensors missing (`k`/`v` for dense/reuse; index tensors for indexer) | `ValueError` from the public op |
| dtype/shape/stride/layout/positions violation on the native path | `ContractError` from `inputs.py` (before binding; no partial device write) |
| valid call outside the native fused coverage (`M > 256` today) | **coverage error today**: `ContractError` (skill §5 option B); see §8; `sp_qkv_covers` returns False |
| `aiter.plugin` absent | op still imports and calls native (guarded decorator); no provider dispatch |
| provider declines a call | native serves it (normal operation) |
| provider `select`/candidate errors before `run` | only that provider is removed; next enabled provider is asked, else native |
| provider `run` raises after starting | exception propagates; **no native retry** (side effects must not repeat) |
| runtime implementation absent | `ImportError` naming `sp_qkv_native` (native unavailable, not silently wrong) |
| graph capture reached before warm-up/prepare | error from the runtime/prepare contract; not supported |

## 8. Coverage matrix

Fused-native coverage today is the **prologue cells the native runtime can actually
serve**. Those are the active profile's 2-stage (small/prologue) plans, including pad
hits into their buckets. For the default profile (`minimax_m3_sp4_gfx950_v1`, 123 rows)
the fused-native range is **local `M = 1..256` for all three modes**: the exact prologue
buckets `{1, 2, 4, 8, 12, 16, 32, 64, 128, 256}` per mode plus every `m` that pads into
one of those buckets. The alternate r19 profile does not add fused-native prologue cells
beyond the 2-stage plans it serves.

Covered on **gfx950 / MI355X**, SP world **4**, local `M = x.shape[0]`, `T = 4*M`:

| mode | fused-native M coverage today | notes |
|---|---|---|
| dense | `1..256` (2-stage prologue cells incl. pad hits) | exact buckets `{1,2,4,8,12,16,32,64,128,256}` + pad hits; prologue inserts/writes |
| indexer | `1..256` (same) | N = 9856; index cache writes |
| reuse | `1..256` (same) | indexer GEMM, compact exchange (F = 2304), no index writes |

Any `M` outside `1..256` is a **coverage error** today (`ContractError`), not a silent
fallback; callers must gate on `sp_qkv_covers(...)` before selecting the fused path.
`sp_qkv_covers` returns `False` for the uncovered band, and `sp_qkv_native` raises for it.

### Phase-2 coverage / follow-up (the one-stage large and r19 band)

The one-stage `sp_qkv.large.v1` plans (r13/r18, `M = 512..8192`) and the r19 ABI-7 plans
(`M = 9216..16384`) implement **GEMM + SP exchange only**. This contract's payload also
carries the prologue outputs and cache writes (`q`/`k`/`v`/`index_q`, head norm + RoPE +
KV/index-cache insert), so serving the coarse contract for those `M` values needs a
**native post-exchange prologue stage** first. Today that prologue is ATOM-owned Triton
(`atom/model_ops/triton_fused_qkv_norm_rope_cache.py` + `aiter.fused_qknorm_idxrqknorm`),
not an aiter native stage; importing it into this HIP PR would be a second backend
(one backend per PR), so it is deliberately out of scope here.

Two honest follow-ups, to be settled by the framework owners:

1. **(i) new aiter-owned prologue op/stage** wired behind the native runtime, then extend
   this contract's coverage to the one-stage large/r19 band; or
2. **(ii) split the contract** into `sp_qkv_gemm_exchange` (covers `M 512..16384`,
   including the r19 alternate profile) plus a separate prologue op — the skill's open
   question #1 (`framework-to-provider.md` §9).

Until either lands, `sp_qkv_covers(...)` is `False` for the one-stage large/r19 band and
`sp_qkv_native(...)` raises `ContractError`; option A (an unfused fallback owned by
native, so the native path serves every valid call) remains the named follow-up for the
general path.

**Explicit phase statement:** *the native path is coverage-limited today (skill §5 option
B: prologue cells `M 1..256` for the default profile); option A (an unfused fallback owned
by native) is the named follow-up.* A provider may decline a call only when native can
serve it; otherwise the call is a coverage error, not a silent fallback. Consumers must
use `sp_qkv_covers` to decide whether to route a call to this op.

## 9. Provider payload (native handoff)

The native runtime receives one positional `dict` with exactly the contract argument names
as keys, in signature order: `q, x, x_scale, w, w_scale, q_norm_w, k_norm_w, cos_cache,
sin_cache, positions, slot_mapping, k_cache, v_cache, k_scale, v_scale, mode, k, v,
index_q, index_q_norm_w, index_k_norm_w, index_cache, eps, group, stream`. Optional
tensors are present with `None` when not supplied. This payload is an implementation
handoff, not the public provider contract (a real provider receives canonicalized
positional args + keyword mapping from `aiter.plugin`).

## 10. Runtime interface expected by this P0

The runtime implementor owns `aiter/ops/gemm_fusion/sp_qkv/runtime.py`; this contract
assumes:

```python
def sp_qkv_native(payload: dict) -> None: ...   # fused/native selection, resources, prepare/run
def sp_qkv_covers(payload: dict) -> bool: ...   # cheap, non-executing coverage predicate (P0 surface)
def sp_qkv_available() -> bool: ...             # optional readiness query used by callers/providers
def profile_name() -> str: ...                  # optional provenance of the active profile
```
`op.py` re-exports the predicate as `sp_qkv_covers(...)` with the same flat arguments;
it builds the payload and returns `False` when the runtime/predicate is unavailable or
the mode is unknown. `aiter.sp_qkv_covers` is the lazy public spelling.

Any mismatch with these names/shapes must be reported before the P1 wiring lands (the
op module is intentionally importable before `runtime.py` exists; the first call raises a
named `ImportError` if the runtime is still absent).
