# QSA index scorer (`qsa_paged_mqa_logits`) — op contract v1, FlyDSL gfx1250 + native torch

`aiter.qsa_paged_mqa_logits` scores compressed key groups for the Qwen sparse-attention
(QSA) indexer: paged MQA logits with ATOM's semantics and signature. One contract, two
implementations, chosen per call by aiter:

| path | implementation | when |
|---|---|---|
| `fused` | FlyDSL kernel `aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py` (index page 4 / 8 / 16, gfx1250) | a profile is selected **and** covers the call (golden-verified row / range rule / bucket) |
| `native` | `aiter/ops/qsa_index_scorer/native.py` (pure torch, vid `native.torch.v1`) | every other valid call (no profile = default, other arch, uncovered shape, `score_divisor`, alignment, …) |

The caller needs no scorer of its own: every valid call returns a correct result.
The decision and its reason are observable before anything runs (`decide`).

```
visible[t]   = clamp(min((pos[t] + 1) // ratio, ctx[req(t)] // ratio), min=0)   (0 if req(t) invalid)
logits[t][c] = sum_h relu(q[t,h,:] . k[c]) / divisor     if c < visible[t] and page(c) is valid
             = -inf                                       otherwise
k[c]         = cache[page_table[req, c // P], c % P, 0, :]                    (bf16)
             = cache_e4m3[...] * scale[page_table[req, c // P], c % P, 0]     (fp8)
```

## Contract sheet (op `qsa_paged_mqa_logits`, contract version 1)

| field | contract |
|---|---|
| op name / version | `qsa_paged_mqa_logits`, `OP_CONTRACT_VERSION = 1` (`aiter/ops/qsa_index_scorer/op.py`). Backend-neutral; the backend is a dispatch decision recorded in `Decision.path` / `Decision.vid`. |
| public signature | `aiter.qsa_paged_mqa_logits(q, compressed_k_cache, page_table, token_to_request, query_positions, context_lens, compress_ratio=4, score_divisor=None, max_columns=None, row_starts=None, allow_flydsl=False, compressed_k_scale=None) -> tuple[Tensor, Tensor]` — exactly ATOM's 12 parameters, names, order and defaults (`api.ATOM_SCORER_SIGNATURE`, drift-checked against the frozen ATOM source by `inspect.signature` and AST tests). No implementation policy (profile, buckets, output buffers) in the public signature. |
| dtypes | `q` bf16 (fp16/fp32 accepted with an fp8 cache on the native path); `compressed_k_cache` bf16, or fp8 E4M3 with `compressed_k_scale` fp32 `[pages, P, 1]` (one scale per compressed row); metadata int32 or int64. |
| shapes / layouts | `q [rows, heads, head_dim]`; cache `[pages, P, 1, head_dim]`; `page_table [requests, width]`; `token_to_request`, `query_positions [rows]` and `context_lens [requests]` 1-D contiguous. `columns = width * P`, or `max(1, min(width * P, max_columns))`. Native: any `P`, `heads`, `head_dim`, any cache alignment. Fused: `heads = 4`, `head_dim = 128`, `P in {4, 8, 16}`, `ratio = 4`, `divisor = sqrt(128)`, contiguous 16-B aligned `q`, 128-B aligned contiguous cache. |
| outputs / mutation | returns `logits` fp32 `[rows, columns]` and `visible_groups` int32 `[rows]`; `row_starts` (int32 `[rows]`), if given, is zero-filled in place. Inputs are never written. No aliasing between outputs and inputs is allowed. Module API (`aiter.ops.qsa_index_scorer`): caller-supplied `logits` / `visible_groups` are written in place and returned. |
| tolerance / determinism | Reference: the frozen torch golden (`qsa_golden.reference_logits`). Pass = `compare_logits(rtol=1e-4, atol=1e-4)` over `valid_page & c < visible`, `visible_groups` exact, no NaN, every element outside that mask bit-identical (`-inf`), `row_starts == 0`. Both paths are deterministic for a given decision (no atomics); fused and native agree within the tolerance, not bitwise. |
| workspace / lifetime | Fused: no workspace (bucket staging planes for the `pad4` buckets are owned by aiter, allocated at `decide`). Native: one bounded tile workspace (`TILE_ELEMS = 65536` (row, column) elements, ~54 MiB at H=4, D=128, independent of the problem size) allocated at `decide(..., native=True)` and owned by the `Decision`; freed with it. |
| graph capture | `prewarm` and `decide` run **outside** capture (they may compile, allocate and read tensor metadata). `launch(decision, ...)` is capture-safe on both paths: no host sync, no tensor reads on the host, no environment reads after the first launch; with caller-supplied `logits` / `visible_groups` the native path creates no tensor (dispatch-audited on CPU; contiguous caches). |
| error behaviour | Contract violations (shape, dtype, device, non-positive `score_divisor` / `compress_ratio`, MXFP4 / MXFP8 caches) raise `ValueError` (`NativeContractError`); `decide` reports them as `path == "invalid"`. An uncovered call is not an error: it runs native with a typed reason. `strict=True` (module API) and `try_qsa_paged_mqa_logits` return/raise instead of running native. An error inside a launched kernel propagates (no transparent re-run). |
| coverage | Native: every valid call, any device torch supports. Fused: gfx1250 with 256 CUs, the profile `qsa_page4_gfx1250_v1` (below): bf16 and fp8, index page 4 / 8 / 16, draft / decode / MTP-verify / tail / prefill regimes, 64–65536 columns. |
| non-goals | MXFP4 / MXFP8 index caches (ATOM supports MXFP4; this op raises `ValueError`); fused path on other arches; `allow_flydsl` (accepted and ignored — it selects ATOM's own page-16 kernel); top-k selection / index expansion (`aiter.ops.qsa.qsa_select_paged_tokens`). |

Existing neutral op with the same name: `aiter.ops.qsa.qsa_paged_mqa_logits` (Triton / page-16
FlyDSL prefill dispatcher from #6469, 11 parameters, bf16 only, not exported at the top
level). This op is the 12-parameter superset (adds the fp8 scale) and is the one exported as
`aiter.qsa_paged_mqa_logits`; unifying the two dispatchers is a follow-up.

## Module API (`aiter.ops.qsa_index_scorer`, `API_VERSION` 1.2)

| name | purpose |
|---|---|
| `decide(<12 args>, *, logits=None, visible_groups=None, profile=None, allow_unverified=False, buckets=None, native=True) -> Decision` | which path serves this call class and why; prepares it for `launch` |
| `launch(decision, <12 args>, *, logits=None, visible_groups=None)` | runs the decided path (fused or native) without re-deciding; O(1) class-key guards |
| `qsa_paged_mqa_logits(<12 args>, *, logits=None, visible_groups=None, profile=None, strict=False)` | `decide` + `launch`; `strict=True` raises `QSAIndexScorerUnsupported` instead of running native |
| `try_qsa_paged_mqa_logits(...) -> (out or None, Decision)` | fused or `None` (no native workspace allocated) |
| `eligibility(...) -> (ok, reason)`, `layout_eligibility(cache, scale=None, *, profile=None)` | fused-path checks only |
| `prewarm(profile=None, *, cache_dtypes=("bf16",), pos_dtypes=(int32, int64), device=None, page_sizes=None, compressed_k_cache=None, compressed_k_scale=None, cache_dir=None, jobs=1, compile_only=False)` | build + JIT-compile every fused variant the profile can select (scoped to a bound cache); persistent FlyDSL cache under `compile_cache_dir()` |
| `Decision(ok, reason, detail, profile, cell, vid, columns, path)` | `ok` ⇔ `path == "fused"`; `path` `"native"` has `vid == "native.torch.v1"` and keeps the fused-decline `reason`; `"invalid"` = contract violation |
| `Reason`, `NATIVE_VID`, `PATH_FUSED/NATIVE/INVALID`, `OP_CONTRACT_VERSION` | stable values |

Serving protocol (ATOM): model load → `prewarm(...)`; per call class (bucket), outside graph
capture → `d = decide(*call, profile=...)` (log `d.path`, `d.reason`, `d.vid` once); every step,
eager or captured → `launch(d, *call, logits=..., visible_groups=...)`. API 1.2 vs 1.1:
`Decision.path`, `decide(native=)`, `launch` runs native decisions, `qsa_paged_mqa_logits`
serves uncovered calls natively (`strict=True` restores the 1.1 raise).

## Enablement: fused path off by default

Nothing fused runs — and neither FlyDSL nor the kernel module is imported, no device is
queried — unless a profile is named: `profile="qsa_page4_gfx1250_v1"` (module API) or
`AITER_QSA_INDEX_SCORER_PROFILE=qsa_page4_gfx1250_v1` (a name in `aiter/configs/qsa_index_scorer/`
or a JSON path; the only switch for the public op). Without one every call is served natively
with reason `disabled`. `AITER_QSA_INDEX_SCORER_BUCKETS=0` disables token buckets.

## Native implementation (`native.py`)

Pure torch, no Triton / HIP. Per call class: validate the contract, build the column → (page
slot, offset) tables, allocate one tile workspace. Per launch: for each row tile, compute
`visible` and convert `q` to fp32; for each column tile, gather the page-table slots, the
physical pages (invalid → masked) and the key rows (`index_select` on a byte view, fp8 →
fp32 with the row scale), `bmm` per row, `relu`, head sum in head order, divide, mask with
`-inf`, copy into the output. All per-tile ops are `out=` / in-place. Rows are tiled, so
memory is bounded for any `rows × columns`. Per-row key gathers avoid a host-side
per-request loop (the golden's `nonzero`), which is what makes the launch sync-free.
Performance is not a goal of this path (it is the correctness floor for uncovered calls).

## Profile `aiter/configs/qsa_index_scorer/qsa_page4_gfx1250_v1.json`

Pins kernel source sha256 `58051973626d598b36b0c51c4ac89a8830fa05b3a8f78d9628aeadd436c1f0ad` and
every import-time `QSA_*` switch (`kernel_env`). GPU-stamped by `gen_profile check` on that
kernel (frozen golden, criterion above): **rows 20/20, range rules 39/39, buckets 6/12 pass**;
the 6 `fail` buckets are the `verify3_pad4_*` / `tail_round4_*` staging buckets, which are
routing-only and never selected (only `pass` entries ever serve). Rules per page P ∈ {4, 8, 16}
and dtype ∈ {bf16, fp8}: `draft` (rows ≤ requests), `decode` (decode band), `tail`,
`prefill_*_kt` (lean ktile, ≤ 2048 columns; fp8 ×C2 up to 2048 rows, ×C4 above),
`prefill_*_persist` / `prefill_fp8_r32_mid` (4096–8192 columns), `*_wide` (16384–65536
columns, rows × columns ≤ 2^25). Buckets: `verify3_g3_{bf16,fp8}[_p8|_p16]` (MTP verify, 3 rows
per request, one kernel-native 3-row decode-group launch, zero copy).

Runtime refusals (→ native with that reason): kernel source sha ≠ profiled
(`profile_kernel_mismatch`), any `QSA_*` switch ≠ `kernel_env` (`kernel_env_mismatch`),
kernel router ≠ the entry's variant (`route_mismatch`). `ipf` (`s_prefetch_inst`) kernel
variants fault on GPU and are not routable: the profile loader rejects any entry pinning `ipf`.
The kernel's in-kernel phase trace (`trace=1|2` builder keyword) is opt-in, off by default,
never selected by the op and outside every profiled variant.

```
python -m aiter.ops.qsa_index_scorer.gen_profile verify     # CPU: committed file == generator
python -m aiter.ops.qsa_index_scorer.gen_profile coverage --plan-file cap.json
QSA_GOLDEN_DIR=... python aiter/ops/qsa_index_scorer/gen_profile.py check --out restamped.json   # GPU
```

## Reasons (stable values)

`ok`, `disabled`, `profile_not_found`, `profile_invalid`, `profile_kernel_mismatch`,
`kernel_env_mismatch`, `kernel_unavailable`, `device_not_gpu`, `device_mismatch`,
`arch_unsupported`, `cu_unsupported`, `compress_ratio_unsupported`, `q_dtype_unsupported`,
`q_shape_unsupported`, `q_layout_unsupported`, `cache_format_unsupported`,
`cache_shape_unsupported`, `cache_layout_unsupported`, `cache_alignment_unsupported`,
`scale_layout_unsupported`, `metadata_unsupported`, `output_layout_unsupported`, `empty_call`,
`divisor_uncovered`, `max_columns_invalid`, `plane_too_large`, `rows_rule_uncovered`,
`columns_uncovered`, `table_width_uncovered`, `row_unverified`, `route_mismatch`.

## Provider readiness (no dependency on `aiter.plugin`)

The op is shaped for the draft kernel-provider interface (RFC #7240, `aiter.plugin`, not
merged) without importing it:

- `op.qsa_paged_mqa_logits` is the routing point. Once `aiter.plugin` merges, the change is
  one decorator, `@route(op="qsa_paged_mqa_logits", version=1)`, above it; its body (fused
  when profiled, else native) stays the AITER-owned native implementation, so a provider may
  decline any call and native still serves it.
- The public signature carries no implementation policy (profile, buckets, workspace), as a
  routed contract requires; canonicalized provider args are the 12 positional parameters.
- A provider would own its own kernels/profile and serve `select()` from data equivalent to
  `decide` (pure, no device work); `prewarm` is the warm-up a provider lifecycle hook would
  call outside capture (an open interface gap, RFC #7240 open questions).
- Observability already exists before execution: `Decision.path / reason / vid / cell`.

## Tests and tools

| file | runs on | what |
|---|---|---|
| `op_tests/test_qsa_index_scorer_native_cpu.py` | CPU | contract (public signature = ATOM's 12 params, neutral name, top-level export, torch-only native), default off → native with reason, fused vs native routing at page 4/8/16 × bf16/fp8 with typed reasons, contract violations → `invalid` + `ValueError`, native numerics vs an independent float64 reference (pages 2–64, invalid pages/requests, `max_columns`, divisor, ratio, multi-tile) and vs the frozen golden, graph safety (no tensor created, no sync op, no env read in `launch`; negative control), ipf not routable |
| `op_tests/test_qsa_index_scorer_cpu.py` | CPU | profile format/loader/matcher, every reason, off-by-default import side effects, ATOM signature drift, kernel API kept, generator drift, range rules, dispatch of every pass row/rule through the real wrapper (stubbed launch), `launch` ≡ op, `prewarm` sets |
| `op_tests/test_qsa_index_scorer_r2_cpu.py` | CPU | token buckets (g3 kernel-native / pad4 staging / rows), prewarm scoping + persistent cache + workers, page 8, fp8 rules and buckets |
| `op_tests/test_qsa_page16_cpu.py`, `test_qsa_r2_kernel_cpu.py`, `test_qsa_r2b_kernel_cpu.py`, `test_qsa_r2c_kernel_cpu.py`, `test_qsa_r2d_kernel_cpu.py`, `test_qsa_r2e_cpu.py` | CPU | kernel-side gates without a device: page 16, decode-group knobs, K-stationary / persist / ktile / lean / kconv lane emulations vs the frozen golden, trace-off defaults, route presets, committed evidence state |
| `op_tests/test_qsa_index_scorer.py` | gfx1250 GPU | Phase-1 golden cases (fused, strict), the 69 logits_indexer benchmark cells × bf16/fp8 on both paths (fused strict + native, golden compare, native CUDA-graph replay bitwise), side outputs, int32 ≡ int64, `launch` ≡ op ≡ wrapper, graph ≡ eager, profile rows / rule samples, fused vs frozen ATOM |
| `op_tests/op_benchmarks/qsa_index_scorer/*` | CPU / GPU | ISA audit, perf (formal protocol), zero-overhead host cost, prewarm time, bucket benchmark |

## Known limitations

- Fused path: gfx1250, 256 CUs, 4 heads × 128 dims, ratio 4, `score_divisor = sqrt(128)`;
  128-B aligned cache base (16-B / 8-B aligned subviews run native by policy).
- Native path: correctness floor, not performance; with non-contiguous caches it makes a
  contiguous copy per launch (allocates; not graph-safe). Its GPU behaviour is covered by
  `op_tests/test_qsa_index_scorer.py` but has not been run on a GPU in this revision.
- `QSA_*` kernel switches are read at kernel import time; a non-default value makes every call
  native (`kernel_env_mismatch`).
- `launch` trusts the caller's class key beyond its O(1) guards (rows, heads/dim, page-table
  shape/dtype, cache dtype/pages, output shapes): re-run `decide` per bucket.
- Two public QSA scorer dispatchers exist (`aiter.ops.qsa` from #6469 and this op); see the
  contract sheet.
