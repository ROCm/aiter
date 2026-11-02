# Opt-in QSA bucket padding (ROUND-PAD2)

PAD2 base: `f7c91d5dedf206e41ec13f0daa3fa3e1c82029a8` (ROUND-PAD).
No kernel, access helper, WMMA arithmetic, E4M3 conversion, TDM descriptor or
existing builder source is changed. The host planner/dispatcher uses those same
specializations. `QSA_PAD_BUCKET=1` is required. Unset/0/other values retain the
original true-shape dispatch and byte-identical generated kernel ISA.
The parent GPU sweep verified ROUND-PAD padded and exact shapes; **PAD2 requires
fresh GPU correctness/performance verification and is not promoted**.

## Public CPU plan

`qsa_pad_plan(shape, dtype)` accepts a mapping with `rows`, `heads`, `columns`,
`requests`, `page_table_width`, `cache_pages`, optional `regime`; or a six-item
sequence with optional seventh regime. Dtype is `bf16`, `fp8`, or the matching
torch dtype. It imports no additional packages and performs no allocation,
compiler call, tensor operation, device inspection, or GPU work. It can also be
AST-extracted without importing torch/FlyDSL. The containing module retains its
existing imports; `_g2/_load.py` supports stock host-only FlyDSL stubbing.

The returned dictionary has true/padded shapes, geometry, column tile, K-load
mode, `needs_padding`, descriptive regime and true copy-back extents. Regime
never overrides the champion true-shape router. Padding requests FIRST would
incorrectly change rows37/requests1 prefill from r32 to r16: do not do that.

### Bucket selection table

| True-shape condition | Bucket geometry | Column unit | K load |
|---|---|---|---|
| BF16, below 32 rows/request or rows<32 | r16 | 32 by default | TDM |
| BF16, >=32 rows and >=32 rows/request | r32 | 64 | TDM |
| BF16 r32, true columns >=512, default persist8 | r32 | 512 | persistent TDM |
| BF16, explicit R64=1 and >=64 rows/request | r64 | 64 | TDM |
| FP8, below 32 rows/request or rows<32 | r16 | 64 | raw |
| FP8, >=32 rows and >=32 rows/request | r32 | 64 | TDM |
| Forced raw or copy-atom access | champion r16/r32 | 64 | raw |

The existing environment switches remain authoritative: R16 TDM columns32/64,
persistent2/4/8 (units128/256/512), R64 opt-in, shared/per-dtype K-load selection,
scalar override, copy-atom access and fp8 conversion. Scalar overrides do not
pad and keep the historical exact fallback. Raw disables R64 in the router.
The plan is an **alignment-independent candidate**: a TDM/persistent candidate
with cache base not divisible by128 launches r16/raw/c64, as the champion
wrapper does. Allocations remain sufficient for that fallback's row/column
tails. Raw BF16 requires16B cache alignment, raw FP8 requires8B; its16B loads
are LDS-only. TDM requires128B. Cache pages/scales are never extended.

## Padding and copy-back

- Rows round up to the selected geometry, columns to its actual candidate
  column unit; requests round up to4; table width rounds up to column-unit/4.
- Heads remain4, head dimension128, cache_pages and dtype unchanged.
- Before allocation, each padded plane is checked with Python-wide integer
  multiplication against the existing positive/int32-byte-offset budget. The
  existing kernel `_plane_bytes` still widens each clamped dimension to i64
  before multiplying. No descriptors are advertised around smaller buffers.
- Padded Q rows are zero; T2R is -1; positions and contexts are0. Original T2R
  outside TRUE requests is sanitized to-1 before request count is enlarged.
  New PT slots are-1. Existing invalid physical page IDs stay invalid; existing
  kernel guards clamp their source loads to physical page0, retaining c%4.
- Padded output storage is allocated iff rows/columns change. VIS/RST get
  temporary vectors iff rows change. True output buffers are reused otherwise.
- Existing kernels predicate and clamp all stores and write every declared
  output element exactly once, including -inf for invalid rows/pages/visibility.
  Padded columns sharing a valid partial final page are masked to-inf after the
  bucket launch; they are never copied to the caller.
- Copy back only logits[:true_rows,:true_columns], VIS[:true_rows], RST[:true_rows].
  True VIS is not clamped to padded/true columns; this preserves the contract.
  Returned tensors are the caller's original output buffers.
- **Native/exact fast path:** native row/column units are cached once at import
  from the original champion specialization settings. True-shape integer
  comparisons select row geometry; rows must be divisible by that geometry,
  columns by its candidate column tile, and PT width by tile/4. No pointer
  inspection is added: the original downstream champion still handles cache
  alignment fallback and raw tails. These calls go straight to the unchanged
  champion launch using TRUE rows/columns/requests/width and original buffers.
  Request count need not be divisible by4: req1/2 prefill buckets now bypass
  metadata padding too, because the original kernel predicates true request IDs
  and indexes true-sized metadata. No request is resurrected by this bypass.
  The public plan remains a ROUND-PAD allocation plan (so its `needs_padding`
  can still be true for a metadata-only native bucket); it is NOT consulted.
- PAD routing performs one `QSA_PAD_BUCKET` env lookup in BOTH modes. Enabling
  it adds only integer true-shape geometry/tile/extents comparisons and a false
  diagnostic flag check. No planner dict,
  opt-in device/alias validation, endpoint scan, padded tensor, copy-back or
  added tensor attribute inspection. Original champion safety checks, cached
  builder, alias-only tensor views and single kernel launch remain unchanged.
- Allocation/sync-free evidence assumes caller-supplied outputs, contiguous
  int32 metadata, and an already-built specialization, just as PAD=0. Missing
  outputs still allocate; int64 narrowing costs one device clamp+narrow launch
  per int64 tensor and no host sync since ROUND-NOSYNC (`QSA_STRICT_I32=1`
  restores the historical min/max probe, see `ROUND_NOSYNC.md`);
  noncontiguous metadata still copies; first compilation still costs. These
  are unchanged original API costs, not bucket-padding overhead.
- **Non-native extent path** retains the full ROUND-PAD plan/allocate/launch/
  copy-back and endpoint saturation. Before allocation, it validates all input
  devices, contiguous outputs, and disjoint output intervals against every
  ORIGINAL input/output (including metadata before narrowing). Disjoint views
  of one storage are okay; overlapping writes are rejected. Native buckets have
  only the original champion acceptance contract: callers must provide safe
  same-device, nonoverlapping buffers. Scalar opt-in retains slow validation.
  Default-off validation is unchanged.
- Zero rows/columns/requests/PT-width/cache-pages are explicitly rejected, never
  dispatched. Singleton dimensions are supported. Columns exceeding true PT
  capacity are rejected by the plan rather than silently truncated. The wrapper
  retains its historical max_columns capacity behavior, including when off.

### Extreme int32 position endpoint

The frozen kernel computes i32 pos+1, so INT32_MAX is a confirmed old overflow
corner. For padding-needed calls, host positions saturate this ONE endpoint to
INT32_MAX-1: because ctx<=INT32_MAX, the mathematical min((pos+1)//4,ctx//4) is
unchanged. Native/identity/scalar bypasses preserve the frozen numeric path:
by default they DO NOT scan or reject INT32_MAX, exactly like PAD=0. The caller
must exclude this old overflow endpoint when requesting mathematical visibility
semantics on an unpadded launch.

`QSA_PAD_INT32_GUARD=1` explicitly enables rejection on PAD=1 bypasses instead
of changing frozen kernel arithmetic. It is read **once at module import**,
like the specialization switches: set it before starting Python/importing this
module; changing it after import has no effect. Unset/0/other values disable it.
The enabled diagnostic uses a comparison tensor + `.any()` + Python `bool`,
and therefore allocates and synchronizes; it is excluded from the free-path
performance contract. PAD=0 never scans even when this diagnostic is enabled.
Padding-needed calls always saturate the endpoint, independent of this env, and
never add the diagnostic host scan before/after their padded dispatch.

## Boundary evidence

`test_pad_bucket_cpu.py`: pure shape routing and budgets, real wrapper capture
for both dtypes, invalid true request resurrection, invalid physical page IDs,
neutral tails, caller sentinel/copy-back bounds, exact output/input reuse,
scalar true extents, copy-atom/R64 overrides, alignment fallback and alias
rejection. PAD2 adds twenty native bucket counter subcases (bf16/fp8,
columns32768/65536, decode64/draft16/prefill512 req1/2/4): all tensor operations
must be alias-only views/detach; zero tensor storage allocations, zero scalar
synchronizations, zero explicit synchronize calls, exactly one PAD env lookup,
and fail-on-call planner/validation/padding mocks. Metadata-only req1/2 native
bypasses retain invalid request rejection and byte-identical model outputs.
Endpoint env tests cover import-time opt-in/default, scalar/exact rejection,
PAD=0 exclusion and unconditional padded saturation. Opaque BF16 K-fragment bytes and all256 FP8 encodings compare exact
versus padded PT plans and independent LDS gather/consumer models.

`test_pad_bucket_trace_cpu.py`: actual production wrapper, padded allocations
and copy-back run over the instrumented **real raw r16/r32 kernel body**. Nine
shapes include row37/req3/width17/cols67, row37/req1, singletons, draft, short
visible/negative positions, invalid requests/pages, an exact bucket and native
r32 req1/2 metadata-only bypasses (true extents and true-invalid requests). The
trace checks i32 intermediate overflow, every all-lane load/store address,
true i64 descriptors and exactly-once stores. True logits compare by int32
bit views; VIS/RST compare exactly. This is CPU evidence, not GPU correctness
or TDM execution evidence. Existing TDM/BM64/persistent/CVT CPU and ISA gates
remain required.

## Required GPU bitwise and performance sweep (root-owned)

Run only under the shared-node cooperative GPU lock, on a healthy gfx1250 GPU,
using the approved stock FlyDSL image and a NEW nonfrozen evidence directory.
No GPU execution was performed by this optimizer.

1. Record commit, kernel hash, all QSA env vars, image/compiler/FlyDSL versions,
   GPU/partition/clocks and code object hashes. Establish idle/health gate.
2. For BF16 and FP8, compare PAD=0 and PAD=1 bitwise on approved decode64,
   draft16, prefill512 and full prefill chunk sequences at columns32768/65536.
   Include exact request4-multiple buckets and native req1/2 buckets (no
   metadata padding in PAD2). Use caller-supplied outputs and verify original tensor identities.
3. Sweep rows1,3,7,13,15,17,31,33,37,63,65,70; requests1,3,5; columns1,3,5,
   31,33,63,65,67,257,1000; widths1,17,33,65; contexts negative,0,3,4,short,
   long; positions negative,0,3 and INT32_MAX/INT32_MAX-1. Include invalid request
   -1,true_requests,large and PT pages -1,num_pages,large/-large.
4. Repeat raw/TDM, default persistent8 and opt-in2/4, BF16 r16c32/c64, explicit
   R64, FP8 cvtfast/current and copy-atom safe modes. Verify mixed/full/tail row
   guards and 8B/16B-valid subviews that intentionally fall back to raw for
   non128B cache bases. Reject lower alignment before launch. Verify all256
   FP8 codes (including signed zeros, subnormals and NaN encodings) with nonunit
   per-token scales and documented existing scalar-vs-WMMA acceptance contract.
5. Sentinel-protect logits/VIS/RST backing tails, nonalias outputs and disjoint
   views; verify -inf masks, exact metadata, no OOB/error or GPU-health changes.
   Compare padded-on WMMA to padded-off SAME WMMA specialization bitwise. Do
   not introduce a new arithmetic or converter path based on tolerance alone.
6. Unprofiled randomized A/A then A/B latency + held-out rerun, warm caches and
   same stream. Keep QSA_PAD_INT32_GUARD unset/0 before import for the free-path
   A/B; measure its intentionally allocating/synchronizing opt-in separately.
   Include host allocations/copy-back for genuinely padded cases. Measure
   kernel-only and end-to-end separately. Do not infer gains from CPU timing or
   instruction counts. Target PAD=1 within <=2% of PAD=0 on all nine approved
   buckets; promote only after fresh correctness and held-out timing pass.

Expected ISA register/instruction/scratch delta for each reused specialization:
zero (checked against fresh ROUND-PAD champion no-device ISA). Genuinely padded
extents still add allocation/fill/copy/metadata launches; speedup is not claimed.
Native extents, including approved prefill req1/2, add no tensor work or device
synchronization. Expected end-to-end native delta is approximately zero (only
host env/extent predicates); <=2% remains a GPU measurement target, not a CPU
performance claim.
