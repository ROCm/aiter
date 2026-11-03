# QSA index scorer (paged MQA logits, compressed page 4) — FlyDSL gfx1250

One backend: FlyDSL on gfx1250 (MI455). Kernel:
`aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py`
(public kernel entry `qsa_paged_mqa_logits_page4(...)`, unchanged). Public op:
`aiter.ops.qsa_index_scorer`.

```
visible[t]   = clamp(min((pos[t] + 1) // 4, ctx[req(t)] // 4), min=0)
logits[t][c] = -inf                                              if c >= visible[t] or page invalid
             = sum_{h<4} relu(q[t,h,:] . k[c]) / sqrt(128)       otherwise
k[c]         = dequant(compressed_k_cache[page_table[req, c // 4], c % 4, 0, :])
```

## API (`aiter.ops.qsa_index_scorer`, API_VERSION 1.0)

| name | purpose |
|---|---|
| `qsa_paged_mqa_logits(q, compressed_k_cache, page_table, token_to_request, query_positions, context_lens, compress_ratio=4, score_divisor=None, max_columns=None, row_starts=None, allow_flydsl=False, compressed_k_scale=None, *, logits=None, visible_groups=None, profile=None)` | same 12 leading parameters as ATOM's scorer; returns `(logits fp32 [rows, columns], visible_groups int32 [rows])`; raises `QSAIndexScorerUnsupported` (`.reason`) for any uncovered call |
| `eligibility(<same args>) -> (ok: bool, reason: str)` | decision only, no launch |
| `decide(<same args>) -> Decision(ok, reason, detail, profile, cell, vid, columns)` | full decision; iterable as `(ok, reason)`; `.static` = reason holds for the whole cache/bucket |
| `layout_eligibility(compressed_k_cache, compressed_k_scale=None, *, profile=None)` | prepare-time checks (profile, kernel build, device, cache layout) |
| `try_qsa_paged_mqa_logits(...) -> (out or None, Decision)` | exception-free form |
| `Reason` | stable string enum of every reason (below) |

`allow_flydsl` is accepted for signature parity and ignored (it selects ATOM's own page-16 scorer).
`row_starts`, when given, is zero-filled (ATOM semantics). Metadata may be int32 (zero copy) or int64.

## Enablement: off by default

Nothing is used (and neither FlyDSL nor the kernel module is imported, no device is queried)
unless a profile is named: `profile="qsa_page4_gfx1250_v1"` or
`AITER_QSA_INDEX_SCORER_PROFILE=qsa_page4_gfx1250_v1` (a name in `aiter/configs/qsa_index_scorer/`
or a path to a JSON file). Without one, every call returns `disabled`.

## Profile `aiter/configs/qsa_index_scorer/qsa_page4_gfx1250_v1.json`

Exact rows written by `python -m aiter.ops.qsa_index_scorer.gen_profile emit` and golden-checked by
`... gen_profile check` (GPU, frozen golden from `$QSA_GOLDEN_DIR`, `compare_logits(rtol=1e-4,
atol=1e-4)` + visible exact + row_starts zero + outside-mask `-inf` bit-identical + no NaN).
Top level: `schema_version`, `profile`, `api_version`, `backend`, `kernel` (module, path,
**sha256** of the kernel source, git rev), `kernel_env` (every import-time `QSA_*` switch and the
value all rows were checked with), `contract`, `golden` (revision, comparator, tolerance),
`generated_by`, `rows`. Row: `sig` (sha of the match keys), `arch`, `cu`, `cache_dtype`, `regime`,
`rows`/`requests` (`{min,max[,multiple_of]}`), `columns`, `table_width`, `vid`, `params`, `cell`,
`fixture`, `golden` (`status` pass|pending|fail + evidence), `generated_by`, `notes`.

Only rows with `golden.status == "pass"` are ever selected. The runtime also refuses the profile when
the kernel source sha differs (`profile_kernel_mismatch`), when any `QSA_*` switch differs from
`kernel_env` (`kernel_env_mismatch`), or when the kernel router would pick a different variant than
the row's `vid`/`params` (`route_mismatch`). `gen_profile verify` (CPU) asserts the committed file is
exactly what the generator writes. `--plan-file captured.json` adds captured call classes
(`{cache_dtype, rows, requests, columns, table_width}`) as `pending` rows.

Regimes (identical to the kernel router): `draft` rows ≤ requests (draft kernel, TDM, 32-column wave
slabs); `decode` requests < rows < 32·requests and rows % 4 == 0 (decode row-group kernel, w32,
barrier-free); `prefill` rows ≥ 32·requests (bf16: persistent r32 TDM for columns ≥ 512, else r32
TDM; fp8: r32 TDM, native fp8→bf16 conversion). Anything else is `rows_rule_uncovered`.

## Reasons (stable values)

`ok`, `disabled`, `profile_not_found`, `profile_invalid`, `profile_kernel_mismatch`,
`kernel_env_mismatch`, `kernel_unavailable`, `device_not_gpu`, `device_mismatch`,
`arch_unsupported`, `cu_unsupported`, `compress_ratio_unsupported`, `q_dtype_unsupported`,
`q_shape_unsupported`, `q_layout_unsupported`, `cache_format_unsupported`,
`cache_shape_unsupported`, `cache_layout_unsupported`, `cache_alignment_unsupported`,
`scale_layout_unsupported`, `metadata_unsupported`, `output_layout_unsupported`, `empty_call`,
`divisor_uncovered`, `max_columns_invalid`, `plane_too_large`, `rows_rule_uncovered`,
`columns_uncovered`, `table_width_uncovered`, `row_unverified`, `route_mismatch`.
Static (prepare-time, rank-uniform for identical caches/devices): profile/kernel/env/device/cache
reasons (`STATIC_REASONS`); the rest depend on the call's shape.

## Tests and tools

| file | runs on | what |
|---|---|---|
| `op_tests/test_qsa_index_scorer_cpu.py` | CPU, no device | profile format/loader/matcher, every reason, off-by-default (no import side effects), ATOM signature drift (AST of `$QSA_GOLDEN_DIR/atom_impl/.../qsa.py` + its sha), kernel API kept, generator drift, regime/variant vs kernel router, dispatch of every verified row through the real wrapper |
| `op_tests/test_qsa_index_scorer.py` | gfx1250 GPU | Phase-1 golden cases through the API (frozen comparator), side outputs, inputs untouched, int32≡int64, CUDA-graph≡eager, profile rows, fused vs frozen ATOM |
| `op_tests/op_benchmarks/qsa_index_scorer/isa_audit_qsa_index_scorer.py` | CPU, no device | per row × {int64, int32 positions}: API-dispatched ISA == champion kernel ISA, zero scratch, resources |
| `op_tests/op_benchmarks/qsa_index_scorer/perf_qsa_index_scorer.py` | GPU | formal protocol: `run_perftest(testGraph=True)` + graph replay, fused vs ATOM native, same process, interleaved legs; `decide()` host cost |

## Known limitations

- **gfx1250 only**, CU count as profiled (256). Other arches/CU counts → `arch_unsupported` / `cu_unsupported`.
- **Geometry fixed**: 4 index heads × 128 dims, compressed page 4 (KV block 16 / ratio 4), q bf16,
  `score_divisor = sqrt(128)` only.
- **Index-cache formats**: bf16 and fp8 E4M3 with one fp32 scale per compressed row only. **mxfp8 and
  fp4 (MXFP4) are not covered** (the golden supports them; the kernel does not) → `cache_format_unsupported`.
- **fp8 decode-group 64-column variant is disabled** (`QSA_DECODE_GROUP_COLS=64` was GPU-incorrect for
  fp8; the kernel always uses 32 columns for fp8, and any non-default value of that switch makes the
  profile refuse every call with `kernel_env_mismatch`).
- **Alignment fallback**: TDM routes need a 128-B aligned cache base. The kernel itself falls back to
  raw-load kernels for 16-B (bf16) / 8-B (fp8) aligned subviews and stays correct, but those routes are
  not profiled: by policy such calls run native (`cache_alignment_unsupported`).
- **Coverage = exact profiled rows with golden PASS** (decode 64/16, draft 16/16, prefill 512/2 at the
  e2e table width 16384 and the Phase-1 small shapes). Rows marked `pending` (e.g. decode fp8
  c32768, draft c32768, prefill 320/2) are inert until `gen_profile check` passes on GPU. Other batch
  sizes, column buckets and table widths run native until captured, added and checked.
- **r16 tail class** (requests < rows < 32·requests with rows % 4 != 0) is correct in-kernel but not
  profiled (`rows_rule_uncovered`). The opt-in kernel padding (`QSA_PAD_BUCKET=1`) is not used by the
  op (`kernel_env_mismatch` if set).
- **Prefill fp8 performance baseline**: ATOM's default fp8 prefill route aborts (LLVM PHINode
  assertion) on this toolchain, so there is no like-for-like native timing for that bucket.
- `QSA_*` kernel switches are read at kernel import time; change them only before first use.
- Eager-mode host cost: `decide()` runs per call (pure Python); callers should decide once per bucket
  (`layout_eligibility` at preparation, shape decision per bucket) and reuse it; CUDA-graph replay has
  no host cost.
