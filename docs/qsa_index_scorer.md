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

## API (`aiter.ops.qsa_index_scorer`, API_VERSION 1.1; 1.0 functions unchanged)

| name | purpose |
|---|---|
| `qsa_paged_mqa_logits(q, compressed_k_cache, page_table, token_to_request, query_positions, context_lens, compress_ratio=4, score_divisor=None, max_columns=None, row_starts=None, allow_flydsl=False, compressed_k_scale=None, *, logits=None, visible_groups=None, profile=None)` | same 12 leading parameters as ATOM's scorer; returns `(logits fp32 [rows, columns], visible_groups int32 [rows])`; raises `QSAIndexScorerUnsupported` (`.reason`) for any uncovered call |
| `eligibility(<same args>) -> (ok: bool, reason: str)` | decision only, no launch |
| `decide(<same args>) -> Decision(ok, reason, detail, profile, cell, vid, columns)` | full decision; iterable as `(ok, reason)`; `.static` = reason holds for the whole cache/bucket |
| `layout_eligibility(compressed_k_cache, compressed_k_scale=None, *, profile=None)` | prepare-time checks (profile, kernel build, device, cache layout) |
| `try_qsa_paged_mqa_logits(...) -> (out or None, Decision)` | exception-free form |
| `Reason` | stable string enum of every reason (below) |
| `prewarm(profile=None, *, cache_dtypes=("bf16",), pos_dtypes=(torch.int32, torch.int64), device=None) -> list[str]` | **1.1** build + JIT-compile every kernel variant a `pass` row/rule can select (both position dtypes) on a private dummy problem; returns tags `"<vid>/<dtype>/<i32\|i64>"`; idempotent; `[]` when off / device not covered |
| `launch(decision, <same 12 args>, *, logits=None, visible_groups=None)` | **1.1** run the variant fixed in an `ok` `Decision` without re-running `decide` (O(1) guards only); bitwise identical to `qsa_paged_mqa_logits`, which is `decide` + `launch` |

`allow_flydsl` is accepted for signature parity and ignored (it selects ATOM's own page-16 scorer).
`row_starts`, when given, is zero-filled (ATOM semantics). Metadata may be int32 (zero copy) or int64.

### Zero-overhead serving (API 1.1)

```python
import aiter.ops.qsa_index_scorer as qsa
HAS_11 = tuple(map(int, qsa.API_VERSION.split("."))) >= (1, 1)
# prepare(): model load, before warmup / CUDA-graph capture
if HAS_11:
    qsa.prewarm(profile=cfg.profile)        # first calls no longer compile inside serving
# once per call class (bucket: shapes, dtypes, strides, max_columns, outputs given or not, cache)
d = qsa.decide(*call, profile=cfg.profile)
# every step
if d.ok:
    logits, visible = qsa.launch(d, *call) if HAS_11 else qsa.qsa_paged_mqa_logits(*call, profile=cfg.profile)
else:
    logits, visible = atom_native(*call)    # log d.reason once per (site, reason)
```

- `prewarm` compiles with exactly the builder arguments the launch path uses (same
  `lru_cache` entry, same FlyDSL compiled function); building alone does not compile in FlyDSL, so
  each variant is launched once on a tiny zero problem (≤ 32 rows, ≤ 512 columns, 4 pages) on
  `device` and synchronized. Default `cache_dtypes=("bf16",)`; pass `("bf16", "fp8")` for an fp8
  index cache.
- `launch` guards: decision `ok`, bound to the currently loaded profile object and kernel module
  (a profile reload / kernel re-import makes old decisions refuse with `profile_kernel_mismatch`),
  q dtype (`q_dtype_unsupported`), and rows / page-table shape / cache dtype equal to the decided call
  (`route_mismatch`, memory safety). Everything else is the caller's contract: the decision was
  made for the identical class key (contiguity, alignment class, devices, `max_columns`,
  `score_divisor`, `row_starts`/`logits`/`visible_groups` presence and layout). `compress_ratio`,
  `score_divisor`, `max_columns`, `allow_flydsl` are taken from the decision. No tensor reads, no
  host sync, no environment reads; refusals raise `QSAIndexScorerUnsupported`.
- Profiles keep `api_version: "1.0"` (`PROFILE_API_VERSION`): 1.1 needs no profile change;
  compatibility is by major version.

## Enablement: off by default

Nothing is used (and neither FlyDSL nor the kernel module is imported, no device is queried)
unless a profile is named: `profile="qsa_page4_gfx1250_v1"` or
`AITER_QSA_INDEX_SCORER_PROFILE=qsa_page4_gfx1250_v1` (a name in `aiter/configs/qsa_index_scorer/`
or a path to a JSON file). Without one, every call returns `disabled`.

## Profile `aiter/configs/qsa_index_scorer/qsa_page4_gfx1250_v1.json`

Schema 2: exact rows plus verified **range rules**, written by
`python -m aiter.ops.qsa_index_scorer.gen_profile emit` and golden-checked by
`... gen_profile check` (GPU, frozen golden from `$QSA_GOLDEN_DIR`, `compare_logits(rtol=1e-4,
atol=1e-4)` + visible exact + row_starts zero + outside-mask `-inf` bit-identical + no NaN).
Top level: `schema_version`, `profile`, `api_version`, `backend`, `kernel` (module, path,
**sha256** of the kernel source, git rev), `kernel_env` (every import-time `QSA_*` switch and the
value all rows were checked with), `contract`, `golden` (revision, comparator, tolerance),
`generated_by`, `rows`, `rules` (schema 1 = rows only, still loadable). Row: `sig` (sha of the match keys), `arch`, `cu`, `cache_dtype`, `regime`,
`rows`/`requests` (`{min,max[,multiple_of]}`), `columns`, `table_width`, `vid`, `params`, `cell`,
`fixture`, `golden` (`status` pass|pending|fail + evidence), `generated_by`, `notes`.

Only rows with `golden.status == "pass"` are ever selected. The runtime also refuses the profile when
the kernel source sha differs (`profile_kernel_mismatch`), when any `QSA_*` switch differs from
`kernel_env` (`kernel_env_mismatch`), or when the kernel router would pick a different variant than
the row's `vid`/`params` (`route_mismatch`). `gen_profile verify` (CPU) asserts the committed file is
exactly what the generator writes (rows, rules and rule sample sets).

### Range rules

A rule admits every call of one `(arch, cu, cache_dtype, regime)` whose `rows`/`requests` lie in
inclusive ranges and satisfy the rule's `relation` (`rows<=requests`, `rows==requests`,
`rows==4*requests`, `decode_band`, `tail_band`, `rows>=32*requests`; each implies one regime),
whose `columns` is in `{"pow2": {min, max}}` (every power of two in range) or `{"set": [...]}` and
`<= table_width·4`, and whose `table_width` is in `{min, max}`; `page_size` 4 and `divisor`
`sqrt(head_dim)` are part of the rule. An optional `rows_x_columns_max` ties the rows bound to the
column count (`rows × columns ≤ rows_x_columns_max`; a call above it is `columns_uncovered`): ATOM's
`qsa_select_paged_tokens` chunks rows so one fp32 logits chunk stays ≤ 128 MiB
(`rows ≤ 2^25 / columns`), so the long-context rule only needs (and only verifies) those rows. Each
rule maps to exactly one `vid`/`params`: the loader rejects a rule whose bounds span two router
variants (e.g. bf16 prefill columns 256 and 512), whose largest admitted fp32 logits plane at any
admitted column exceeds int32 addressing, or whose `rows_x_columns_max` leaves an admitted column
without rows, and the call-time `route_mismatch` guard still compares against the kernel's own
router.

Precedence: an exact row that serves the call (golden pass) always wins; otherwise the first
**verified** rule that admits it; otherwise the most specific reason of either path
(`rows_rule_uncovered` → `table_width_uncovered` → `columns_uncovered` → `row_unverified`).
`Decision.cell` is the row cell or `rule:<id>`.

Evidence is per rule, like rows: `golden.status` pass|pending|fail, `kernel_sha256`, and
`golden.checked` = one golden record per sample. A rule is `pass` only if every sample in its
`samples` list passed on the profiled kernel (the loader enforces this). The sample set is
deterministic: rows/requests corners + midpoint (snapped onto the relation) × columns/table-width
corners + midpoint (with `rows_x_columns_max`: rows min / cap / midpoint per sampled column), plus
every built-in shape the rule admits (`STAGE3_CAPTURE`, the profile's exact bf16 rows, and the
AgentX recipe's select-loop calls `AGENTX_SHAPES`) and any `--plan-file`. Sample fixtures use the full table-width context when it fits a 2^22-page budget
(512-request samples get half), the golden's own layout for rows == requests / rows == 4·requests,
and otherwise rows split evenly over min(rows, requests) requests.

| rule | relation | rows | requests | columns | table_width | variant |
|---|---|---|---|---|---|---|
| `draft_bf16` | rows ≤ requests | 1–512 | 1–512 | pow2 64–65536 | 16–16384 | `qsa.page4.draft.v1` |
| `decode_bf16` | decode band | 4–2048 (×4) | 1–512 | pow2 64–65536 | 16–16384 | `qsa.page4.decg.v1` |
| `tail_bf16` | tail band | 2–4095 | 1–512 | pow2 64–65536 | 16–16384 | `qsa.page4.r16.v1` |
| `prefill_bf16_r32` | rows ≥ 32·requests | 32–16384 | 1–32 | pow2 64–256 | 16–16384 | `qsa.page4.r32.v1` |
| `prefill_bf16_persist` | rows ≥ 32·requests | 32–16384 | 1–32 | pow2 512–8192 | 128–16384 | `qsa.page4.r32_persist.v1` |
| `prefill_bf16_persist_wide` | rows ≥ 32·requests | 32–2048, rows·columns ≤ 2^25 | 1–32 | pow2 16384–65536 | 4096–16384 | `qsa.page4.r32_persist.v1` |

Together they cover 73 of the 75 Stage-3 captured classes (Qwen3.8-Flash-Next, no MTP): 72 by a rule
and 1 by an exact row. The two sub-tile column classes (columns 4 and 8) stay `columns_uncovered`.
For the AgentX recipe (MTP3, CONC16, contexts up to 256k, table width 16384, ≤ 16384 batched
tokens) every select-loop call with pow2 columns 64–65536 resolves to a rule: prefill chunks up to
the 2^25 / columns cap (`prefill_bf16_persist_wide` at 16384–65536), MTP3 verify (rows = 4·bs) in
`decode_bf16`, MTP draft (rows = bs) in `draft_bf16`, and a chunked prefill's short last chunk
(rows < 32·requests) in the draft / decode / tail rules.

```
# CPU: rewrite keeping existing evidence; drift guard; how a capture resolves
python -m aiter.ops.qsa_index_scorer.gen_profile emit --keep-golden [--plan-file cap.json]
python -m aiter.ops.qsa_index_scorer.gen_profile verify [--plan-file cap.json]
python -m aiter.ops.qsa_index_scorer.gen_profile coverage --plan-file cap.json
# GPU: golden-check every rule sample (+ admitted plan shapes) and restamp
QSA_GOLDEN_DIR=... python aiter/ops/qsa_index_scorer/gen_profile.py check --only rules \
    --plan-file cap.json --out restamped.json
```

`--plan-file` (a JSON list or `{"plan": [...]}` of `{cache_dtype, rows, requests, columns,
table_width}`) adds rule-admitted entries as rule samples; `--plan-rows` adds the others as
`pending` exact rows.

Regimes (identical to the kernel router): `draft` rows ≤ requests (draft kernel, TDM, 32-column wave
slabs; fp8 tile schedule `qpre`); `decode` requests < rows < 32·requests and rows % 4 == 0 (decode
row-group kernel, barrier-free, `QSA_DECODE_GROUP_COLS=auto`: bf16 32-column / fp8 64-column slabs,
fp8 schedule `base`); `prefill` rows ≥ 32·requests (bf16: persistent r32 TDM for columns ≥ 512, else
r32 TDM; fp8: r32 TDM, native fp8→bf16 conversion); `tail` requests < rows < 32·requests and
rows % 4 != 0 (r16 WMMA kernel: bf16 TDM loads, fp8 raw loads). Every fp8
variant's `params` also record the converter build (`pk8: intrin`) and, for draft / decode-group,
the fp8 schedule (`sched`), so a converter or schedule switch is a `route_mismatch` /
`kernel_env_mismatch`, never a silent reuse of evidence.

### Profiled kernel and evidence (kernel `82c4dec5`)

The profile pins kernel source sha256
`bf49226206b999f948e8b1eb0605935157b3d291b0a0b243581897d8ca661b93` (champion `82c4dec5`) and the
champion-default switch set in `kernel_env`, including the `82c4dec5` knobs `QSA_FP8_CVT_ASM=0`,
`QSA_DRAFT_FP8_SCHED=qpre`, `QSA_DECG_FP8_SCHED=base` and `QSA_DECODE_GROUP_COLS=auto`. Every `pass`
row names the kernel it was GPU-verified on (`golden.kernel_sha256`). A pass that was verified on an
older kernel is accepted only with a recorded `golden.isa_identity` justification (the loader and
`gen_profile verify` reject it otherwise):

- **bf16 rows** keep their Phase-1 evidence. That evidence is `run_gpu.py` correctness 22/22 on
  `a78c285c` (kernel sha `30c6213d…`). It carries over because every bf16 kernel is ISA
  byte-identical from `a78c285c` to `82c4dec5`.
- **fp8 rows** are `pending`. `82c4dec5` changed every fp8 kernel: the intrinsic pk8 converter,
  decode-group w64 and draft `qpre`. Their old evidence is kept only as `stale_cases` /
  `stale_kernel_sha256`. The rows serve nothing until the GPU `gen_profile check` restamps them with
  `kernel_sha256 = bf49…`.

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
| `op_tests/test_qsa_index_scorer_cpu.py` | CPU, no device | profile format/loader/matcher, every reason, off-by-default (no import side effects), ATOM signature drift (AST of `$QSA_GOLDEN_DIR/atom_impl/.../qsa.py` + its sha), kernel API kept, generator drift, range rules (validation, precedence, bounds, reasons, route guard, Stage-3 capture coverage, mocked GPU check), regime/variant vs kernel router, dispatch of every row and of rule samples (pending rows via the sweep path) through the real wrapper, evidence policy; API 1.1: `launch` ≡ kernel wrapper ≡ op launch for every pass class (stubbed builders/`_run`, int32/int64 positions and metadata, `decide` patched to raise), guards, no env reads; `prewarm` variant set = union over pass rows/rules, both position dtypes, idempotence, same builder keys as real calls |
| `op_tests/test_qsa_index_scorer.py` | gfx1250 GPU | Phase-1 golden cases through the API (frozen comparator), side outputs, inputs untouched, int32≡int64, `launch`≡op≡kernel wrapper (bitwise), CUDA-graph≡eager, profile rows, range-rule samples (corners + 4 captured per rule; `--all-rule-samples` for all), fused vs frozen ATOM. `--allow-unverified` also serves/checks `pending` rows; `QSA_TEST_PROFILE=<path>` points the test at a restamped profile |
| `op_tests/op_benchmarks/qsa_index_scorer/isa_audit_qsa_index_scorer.py` | CPU, no device | per row × {int64, int32 positions}: API-dispatched ISA == champion kernel ISA, zero scratch, resources |
| `op_tests/op_benchmarks/qsa_index_scorer/zero_overhead_qsa_index_scorer.py` | gfx1250 GPU | API 1.1: `prewarm` per-variant compile time with a fresh `FLYDSL_RUNTIME_CACHE_DIR` (`--fresh-cache`) and first real call per variant < 5 ms; host µs/call of `launch` vs 1.0 op vs `decide` vs kernel wrapper vs ATOM native; `launch` ≡ op ≡ kernel wrapper bitwise |
| `op_tests/op_benchmarks/qsa_index_scorer/perf_qsa_index_scorer.py` | GPU | formal protocol: `run_perftest(testGraph=True)` + graph replay, fused vs ATOM native, same process, interleaved legs; `decide()` host cost |

## Known limitations

- **gfx1250 only**, CU count as profiled (256). Other arches/CU counts → `arch_unsupported` / `cu_unsupported`.
- **Geometry fixed**: 4 index heads × 128 dims, compressed page 4 (KV block 16 / ratio 4), q bf16,
  `score_divisor = sqrt(128)` only.
- **Index-cache formats**: bf16 and fp8 E4M3 with one fp32 scale per compressed row only. **mxfp8 and
  fp4 (MXFP4) are not covered** (the golden supports them; the kernel does not) → `cache_format_unsupported`.
- **fp8 WMMA hazards (fixed in `82c4dec5`)**: the fp8 kernels used an inline-asm fp8→bf16 converter.
  LLVM's hazard recognizer does not pad inline asm, so its VALUs could overwrite WMMA A/B/D registers
  inside the WMMA hazard window. This made the fp8 decode-group w64 kernel GPU-incorrect. The shipped
  w32 / draft kernels had the same latent violations and were correct only because of how registers
  happened to be allocated. Every fp8 kernel now uses a compiler-visible (intrinsic) pk8 converter,
  and the WMMA-hazard scanner gate reports 0 violations on all default kernels. The fp8 decode-group
  kernel therefore defaults to 64-column slabs; the earlier "fp8 w64 disabled" limitation is gone.
  `QSA_FP8_CVT_ASM=1` restores the asm converter for debug and A/B runs only. With it, fp8 decode-group
  falls back to w32. The profile refuses it with `kernel_env_mismatch`, as it does any non-default
  `QSA_DECODE_GROUP_COLS`, `QSA_DRAFT_FP8_SCHED` or `QSA_DECG_FP8_SCHED`.
- **fp8 rows await GPU re-verification on `82c4dec5`**: until the root's `gen_profile check` restamps
  them, fp8 calls return `row_unverified` and run native.
- **Alignment fallback**: TDM routes need a 128-B aligned cache base. The kernel itself falls back to
  raw-load kernels for 16-B (bf16) / 8-B (fp8) aligned subviews and stays correct, but those routes are
  not profiled: by policy such calls run native (`cache_alignment_unsupported`).
- **Coverage = exact rows and range rules with golden PASS**. The range rules ship `pending` until
  the GPU `gen_profile check --only rules` restamps them; until then only exact rows serve. Exact
  rows: (decode 64/16, draft 16/16, prefill 512/2 at the
  e2e table width 16384 and the Phase-1 small shapes). Rows marked `pending` (every fp8 row on
  `82c4dec5`; bf16 decode c32768 is pass, bf16 draft c32768 / prefill 320/2 are pending) are inert
  until `gen_profile check` passes on GPU. Other batch
  sizes, column buckets and table widths run native until captured, added and checked.
- **Rules are bf16 only** (the Stage-3 e2e index cache); fp8 calls outside the exact rows stay
  native. Columns below one WMMA tile (< 64) are in no rule (`columns_uncovered`).
- **r16 tail class** (requests < rows < 32·requests with rows % 4 != 0) is served only through the
  `tail_bf16` rule once it is GPU-verified. The opt-in kernel padding (`QSA_PAD_BUCKET=1`) is not used
  by the op (`kernel_env_mismatch` if set).
- **Prefill fp8 performance baseline**: ATOM's default fp8 prefill route aborts (LLVM PHINode
  assertion) on this toolchain, so there is no like-for-like native timing for that bucket.
- `QSA_*` kernel switches are read at kernel import time; change them only before first use.
- Eager-mode host cost: `qsa_paged_mqa_logits` runs `decide()` per call (pure Python, ~17 µs on the
  probe host); serving should `decide` once per bucket and call `launch` (API 1.1: precomputed launch
  plan, ~6 µs CPU-side with the launcher stubbed vs ~11 µs for the kernel wrapper). CUDA-graph replay
  has no host cost.
- `launch` trusts the caller's class key beyond its O(1) guards: reusing a decision for a call with
  e.g. a different `max_columns`, a non-contiguous or differently aligned tensor, or outputs added /
  dropped is outside the contract (re-run `decide` per bucket).
- `prewarm` compiles only `pass` variants of the profile for the requested cache dtypes and position
  dtypes; int64 metadata other than positions needs no extra variant (narrowed by a torch clamp).
  The opt-in kernel padding (`QSA_PAD_BUCKET`) is never taken by the op: `launch` replays the
  default-env launch (the env is checked against the profile at `decide` time, import-time snapshot).
