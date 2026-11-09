# QSA index scorer — `aiter.ops.qsa.qsa_paged_mqa_logits` (op contract v1)

`aiter.ops.qsa.qsa_paged_mqa_logits` scores compressed key groups for the Qwen sparse-attention
(QSA) indexer: paged MQA logits. It is the one public entry (the neutral op of #6469); this
change adds a fused FlyDSL gfx1250 implementation behind it, chosen per call class by a dispatch
decision, and keeps every existing caller and default behaviour unchanged.

```
visible[t]   = clamp(min((pos[t] + 1) // ratio, ctx[req(t)] // ratio), min=0)   (0 if req(t) invalid)
logits[t][c] = sum_h relu(q[t,h,:] . k[c]) / divisor     if c < visible[t] and page(c) is valid
             = -inf                                       otherwise
k[c]         = cache[page_table[req, c // P], c % P, 0, :]                    (unquantized)
             = cache_e4m3[...] * scale[page_table[req, c // P], c % P, 0]     (fp8 + fp32 row scale)
columns      = max(1, min(width * P, max_columns))        (width * P when max_columns is None)
```

## Contract sheet (op `qsa_paged_mqa_logits`, version 1)

| field | contract |
|---|---|
| op name / version | `qsa_paged_mqa_logits`, `OP_CONTRACT_VERSION = 1` (`aiter.ops.qsa`). Backend-neutral; the backend is a dispatch decision (`explain(...).path` / `.vid`). |
| public signature | `qsa_paged_mqa_logits(q, compressed_k_cache, page_table, token_to_request, query_positions, context_lens, compress_ratio=4, score_divisor=None, max_columns=None, row_starts=None, allow_flydsl=False, compressed_k_scale=None, stacked_decode=False, *, logits=None, visible_groups=None) -> tuple[Tensor, Tensor]`. The 12 leading parameters are the frozen ATOM scorer's (names, order, defaults); `stacked_decode` (13th positional) is main's Triton scheduling hint; `logits` / `visible_groups` are optional caller-owned output buffers (keyword-only). Existing 11-argument and keyword callers (`qsa_select_paged_tokens`, ...) are unchanged. |
| dtypes | `q` bf16 / fp16 / fp32; `compressed_k_cache` unquantized in q's dtype, FP8 E4M3 (`dtypes.fp8` of the arch) + `compressed_k_scale` fp32 `[pages, P, 1]`, or MXFP4 + E8M0 scale (Triton); metadata int32 / int64. |
| shapes / layouts | `q [rows, heads, head_dim]`; cache `[pages, P, 1, head_dim]`; `page_table [requests, width]`; `token_to_request`, `query_positions [rows]`, `context_lens [requests]` 1-D contiguous. Fused path additionally: `heads = 4`, `head_dim = 128`, `P in {4, 8, 16}`, `ratio = 4`, `divisor = sqrt(128)`, contiguous 16-B aligned `q`, contiguous 128-B aligned cache, a profile row / rule / bucket covering (rows, requests, columns, width, P, dtype). |
| outputs / mutation | returns `logits` fp32 `[rows, columns]` and `visible_groups` int32 `[rows]` (the caller's buffers when supplied, filled in place); `row_starts`, if given, is zero-filled. Inputs are never written; outputs must not alias inputs. |
| tolerance / determinism | Reference: the frozen torch golden (`qsa_golden.reference_logits`). Pass = `rtol = atol = 1e-4` over `valid_page & c < visible`, `visible_groups` exact, no NaN, every element outside the mask `-inf` (bit-identical), `row_starts == 0`. Each implementation is deterministic (no atomics); implementations agree within the tolerance, not bitwise. |
| workspace / lifetime | Fused: none per call (MTP-verify bucket staging planes are allocated at the class decision and owned by the decision cache). Native torch: one bounded tile workspace (65536 (row, column) elements, ~54 MiB at H=4, D=128) per decided class. Triton: as before (allocates its outputs). |
| graph capture | Decide every class before capture (warm-up calls, or `explain(...)`) and `prewarm(...)` the fused kernels: a decided class is a dict lookup plus the implementation's launch, no host sync, no tensor reads, no environment reads; with caller outputs the native-torch path creates no tensor (dispatch-audited on CPU). A class first seen during capture decides on the host inside capture (allocations land in the graph pool). |
| error behaviour | Never raises for a valid call: a call the fused path does not cover runs natively. Contract violations raise `ValueError` (the existing Triton error where Triton validates the call; `explain` reports `path == "invalid"`). A fused class whose first launch fails (compile / load error, before any device write) is re-decided natively with reason `fused_launch_error` (logged); later errors propagate. |
| coverage | Fused: gfx1250 with 256 CUs, profile `qsa_page4_gfx1250_v1` (below). Existing native: Triton (any arch it supports), FlyDSL page-16 prefill (`allow_flydsl`, > 256 rows, bf16). Native torch: every other valid call on any torch device (CPU included). |
| non-goals | fused path on other arches / head shapes; MXFP4 / MXFP8 on the fused or torch path (Triton serves MXFP4); top-k selection / index expansion (`qsa_select_paged_tokens`). |

## Dispatch (per call class, decided once on the host)

| `Decision.path` | implementation | `vid` | when |
|---|---|---|---|
| `fused` | `aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py` via `aiter/ops/qsa_index_scorer/api.py` | `qsa.page<P>.<kind>.v1` | a profile is active and a verified entry covers the class |
| `native-triton` | aiter's existing dispatch (unchanged) | `native.triton.v1` / `native.flydsl_page16.v1` | GPU tensors the existing kernels take |
| `native-torch` | `aiter/ops/qsa_index_scorer/native.py` (pure torch) | `native.torch.v1` | other valid calls (CPU, an fp8 dtype the arch's Triton does not use, ...) |
| `invalid` | — (raises `ValueError`) | — | contract violation |

`Decision.reason` is `ok` for `fused`, otherwise why the fused path did not take the class
(`disabled`, `kill_switch`, `profile_unusable`, `fused_launch_error`, or one of the profile reasons
below). The class key is built from tensor metadata only (shapes, strides, dtypes, devices, the
cache base address and q / scale alignment, the scalar arguments, output-buffer layouts); the
cache keeps at most 4096 classes (FIFO). With the fused path off (default), GPU calls go straight
to the existing dispatch: no class key, no decision cache, no FlyDSL import.

## Policy (`aiter.ops.qsa`, process-global, idempotent)

| call | effect |
|---|---|
| `configure(*, profile: str \| None, source: str)` | enable the fused path with a profile name (`aiter/configs/qsa_index_scorer/<name>.json`) or path; `None` = off (default). `source` names the caller (e.g. `"atom:<config>"`). Re-reads the environment; a different effective profile drops the decided classes. An unusable profile is reported (`status()["profile_error"]`, warning), never raised. |
| `$AITER_QSA_PROFILE` | operator override, wins over `configure` (`off` / `0` / `none` disables) |
| `$AITER_QSA_FUSED=0` | kill switch (also `off` / `false` / `no`): fused path off whatever is configured |
| `prewarm(*, page_sizes, cache_dtypes, pos_dtypes=(torch.int64,), compressed_k_cache=None, compressed_k_scale=None, device=None) -> list[str]` | load / compile every fused variant the active profile can select for these caches; with an AOT-built FlyDSL cache (`aiter.aot.flydsl.qsa`) every variant is a disk hit. `[]` when off. Uses FlyDSL's cache directory (`$FLYDSL_RUNTIME_CACHE_DIR`, set by `import aiter` to the wheel's AOT cache); `$AITER_QSA_INDEX_SCORER_CACHE_DIR` redirects it. |
| `explain(<op args>) -> Decision(path, reason, vid, detail, profile, cell)` | the decision for this call's class (decided and cached like a first call), without running it |
| `status() -> dict` | `enabled`, `profile`, `origin` (configure source / `env:AITER_QSA_PROFILE` / kill switch), `kernel_sha256` / `profile_kernel_sha256`, `op` / `op_version` / `api_version`, decided `classes` and their `paths`, `profile_error`, `routed` |

Serving protocol (ATOM): `configure(profile=..., source="atom:<config>")` before model load;
`prewarm(page_sizes=..., cache_dtypes=..., compressed_k_cache=...)` after KV allocation, before
graph capture; warm-up decides the classes; every step calls the op. Startup line from
`status()`; per-class counters from `explain()`.

## Route-ready (provider interface, RFC #7240)

`qsa_paged_mqa_logits` carries `@route(op="qsa_paged_mqa_logits", version=1)` from `aiter.plugin`
when that module exists (provider stack #7465 / #7484); in a tree without it the decorator is a
no-op shim (`aiter.ops.qsa.ROUTED is False`, the function records `__aiter_route__`). The routed
contract is the signature above; its body (fused when profiled, else native) is the AITER-owned
native implementation, so a provider may decline any call. Profile, buckets and workspaces are
not in the signature.

## AOT (`aiter/aot/flydsl/qsa.py`)

Replays the neutral op on FakeTensors inside `aiter.aot.flydsl.dry_run` (`COMPILE_ONLY=1`) with
the profile configured, for every call class the profile can serve (each verified row's shape,
each verified rule's and bucket's sample classes; page 4/8/16, bf16/fp8, int64 positions), and is
part of `run_aot` (`AOT_MODULES`). `--check` (`FLYDSL_RUNTIME_RUN_ONLY=1`) fails on any cache miss.
Measured without a device (256 cores, flydsl 0.4.0): 2578 classes, 39 kernels, cold 9.2 s,
`--check` 4.4 s with every launch a hit; an empty cache fails all 2578 under `--check`.

FlyDSL folds the value of every module global a kernel's code reads into its disk-cache key; the
kernel module therefore keeps no lazily filled module state on its trace path (kernel 6b537f3b),
so a variant's key does not depend on what else the process compiled first.

## Profile `aiter/configs/qsa_index_scorer/qsa_page4_gfx1250_v1.json`

Pins kernel source sha256 `6b537f3bc8caa54d7b6dfd66a5ff7ca396d0647c189febeadf4d48c1f9f5d8ad` and
every import-time `QSA_*` switch (`kernel_env`). GPU-stamped by `gen_profile check` (frozen golden,
criterion above) on kernels `58051973` / `c01cf365`; each stamp reaches the pinned source through
no-device ISA-identity links recorded per entry in `isa_identity` (AITER-LINT: black/ruff only;
AITER-R3: deterministic cache key; 276/276 default and routable builds byte-identical, 0 WMMA
hazards each). **Rows 20/20, range rules 39/39, buckets 6/12 pass**; the 6 `fail` buckets are the
`verify3_pad4_*` / `tail_round4_*` staging buckets, never selected (only `pass` entries serve).
Rules per page P ∈ {4, 8, 16} and dtype ∈ {bf16, fp8}: `draft`, `decode`, `tail`,
`prefill_*_kt`, `prefill_*_persist` / `prefill_fp8_r32_mid`, `*_wide` (prefill rules admit
requests 1..512, mixed prefill+decode steps). Buckets: `verify3_g3_{bf16,fp8}[_p8|_p16]` (MTP verify,
3 rows per request, one kernel-native 3-row decode-group launch).

Runtime refusals (→ native with that reason): kernel source sha ≠ profiled
(`profile_kernel_mismatch`), any `QSA_*` switch ≠ `kernel_env` (`kernel_env_mismatch`), kernel
router ≠ the entry's variant (`route_mismatch`). `ipf` kernel variants are rejected by the loader.

```
python -m aiter.ops.qsa_index_scorer.gen_profile verify     # CPU: committed file == generator
python -m aiter.ops.qsa_index_scorer.gen_profile coverage --plan-file cap.json
QSA_GOLDEN_DIR=... python aiter/ops/qsa_index_scorer/gen_profile.py check --out restamped.json   # GPU
```

Profile reasons (stable values): `ok`, `disabled`, `profile_not_found`, `profile_invalid`,
`profile_kernel_mismatch`, `kernel_env_mismatch`, `kernel_unavailable`, `device_not_gpu`,
`device_mismatch`, `arch_unsupported`, `cu_unsupported`, `compress_ratio_unsupported`,
`q_dtype_unsupported`, `q_shape_unsupported`, `q_layout_unsupported`, `cache_format_unsupported`,
`cache_shape_unsupported`, `cache_layout_unsupported`, `cache_alignment_unsupported`,
`scale_layout_unsupported`, `metadata_unsupported`, `output_layout_unsupported`, `empty_call`,
`divisor_uncovered`, `max_columns_invalid`, `plane_too_large`, `rows_rule_uncovered`,
`columns_uncovered`, `table_width_uncovered`, `row_unverified`, `route_mismatch`.

## Implementation package (private)

`aiter/ops/qsa_index_scorer/` is the implementation, not a public API: `dispatch.py` (policy +
class cache of the neutral op), `api.py` (fused path: profile match, router, launch plans,
prewarm), `native.py` (pure torch), `profiles.py`, `gen_profile.py` (profile regeneration / GPU
check tool). The path is kept (rather than renamed) because the profile `feature`, the
`gen_profile` commands and the recorded ISA / golden evidence chain name it.

## Tests

| file | runs on | what |
|---|---|---|
| `op_tests/test_qsa_neutral_op_cpu.py` | CPU | neutral-op contract (signature, 11-arg / keyword calls, no top-level export, private package), default off = existing dispatch (no cache, no kernel import), decisions at page 4/8/16 × bf16/fp8, native numerics vs an independent reference, per-class cache (one decision, no allocation / sync on the hot path), configure / env / kill-switch precedence, explain / status / prewarm, route-ready shim, AOT job enumeration, cache-key determinism (static; three compile orders with FlyDSL) |
| `op_tests/test_qsa_index_scorer_native_cpu.py` | CPU | native implementation contract and numerics (pages 2–64, invalid pages/requests, `max_columns`, divisor, multi-tile), graph safety audit, ipf not routable |
| `op_tests/test_qsa_index_scorer_cpu.py`, `test_qsa_index_scorer_r2_cpu.py`, `test_qsa_rules_requests_cpu.py`, `test_qsa_page16_cpu.py`, `test_qsa_r2*_cpu.py` | CPU | profile format/matcher/generator drift, buckets, prewarm scoping, kernel-side gates without a device |
| `op_tests/test_qsa_index_scorer.py` | gfx1250 GPU | golden cases and benchmark cells through the neutral op (fused and native paths), graph ≡ eager, profile rows / rule samples, fused vs frozen ATOM |
| `op_tests/op_benchmarks/qsa_index_scorer/*` | CPU / GPU | ISA audit, perf (formal protocol), host cost, prewarm time, bucket benchmark |

## Known limitations

- Fused path: gfx1250, 256 CUs, 4 heads × 128 dims, ratio 4, `score_divisor = sqrt(128)`,
  128-B aligned cache base.
- Native torch path: correctness floor, not performance; non-contiguous caches are copied per
  launch.
- `QSA_*` kernel switches are read at kernel import time; a non-default value makes every call
  native (`kernel_env_mismatch`).
- Kernel evidence is on flydsl 0.3.4.1; main pins flydsl 0.4.2 (the AOT measurement above ran on
  0.4.0). GPU behaviour of the converged dispatch, AOT on a GPU host and the GPU cache-hit check
  are not verified in this revision.
