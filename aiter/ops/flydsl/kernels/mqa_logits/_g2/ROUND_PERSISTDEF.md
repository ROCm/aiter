# Round PERSISTDEF — measured persistent prefill defaults

Base: `e0d85129a4b2f31f65f84843316a24524efc4a10`.

## Production change

Only module-level defaults and their adjacent explanatory comments change:
`QSA_BF16_PERSIST=1`, `QSA_PERSIST_COLS=8`. The switch still accepts 0/1;
columns still accepts 2/4/8, with the same import-time validation.
`QSA_BF16_PERSIST=0` selects the existing single-tile r32 TDM kernel for A/B.
All 119 production function source segments, including nested functions, were
compared byte-for-byte with the base using `ast.get_source_segment`: identical.
This includes persistent and every nonpersistent kernel, builder, resolver,
wrapper, and guard; historical in-function comments are deliberately preserved.

Routing remains guarded by bf16 WMMA r32 geometry, TDM K load, raw access,
128-byte aligned K (the preexisting earlier alignment guard), and columns >=
64 * P. Default P8 therefore requires >=512 columns. Eligible default dispatch:
`qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4_r32_c64_flydsl_tdm_persist8`.
Forced-off dispatch:
`qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4_r32_c64_flydsl_tdm`.
Unaligned K still chooses raw r16; narrow aligned prefill still chooses the
single-tile r32 TDM path. Raw/copy-atom access cannot select persistent.
Copy-atom's existing wrapper label can remain `tdm`; its builder resolves the
actual raw specialization. Mixed/partial row blocks retain the identical
kernel-side raw fallback. fp8, r16 decode/draft, r64 and copy-atom kernels are
unchanged.

## Test-only changes and reproducible CPU/ISA gates

Routing tests now assert default-on P8, forced-off single-tile, exact fallback
routes, accepted/invalid environment values, and supported P2/P4/P8 spans.
The compile-only helper executes the production wrapper routing AST, then
compiles the selected real FlyDSL builder; it does not launch a kernel.
The two host scripts reset the environment for truly unset `auto`, `off`,
legacy `raw`/`tdm` with persistence off, and explicit `p2`/`p4`/`p8` modes.
The routing and persistent ISA summaries cross-check the selected dispatch,
compile JSON/log and emitted ISA; they require default auto P8 to equal explicit
P8 ISA and all nonpersistent ISA dumps to equal the merge evidence already
retained by the base. The old evidence file is not overwritten.

No-device verification uses host `heliosr-1b114-a07-3.mnb.dcgpu`, staging
`/data/blyu/qsa/work/opt-persistdef-cpu-trace-20261031`, and image
`rocm/fw-bringup-mlperf:gfx1250-atom-20261015` (FlyDSL `0.3.3.dev886`).
The Docker invocation has no `--device`, is network-disabled, mounts source
read-only, and asserts that neither `/dev/kfd` nor `/dev/dri` exists.
Run `host_verify_persist_cpu.sh SOURCE_ARCHIVE` to reproduce.

Local Windows stdlib gates passed: routing 12/12, persistent 18/18;
`git diff --check` passed. Remote verification results and ISA resource counts
are retained separately in `persistdef_isa_evidence.json`.
Remote full verification completed with `ALL_GATES_RC=0`: contract, addressing,
Q LDS, trace16/32, K TDM (37 tests), routing (12), fp8 K TDM (18), BM64 (12),
persistent (18), strict TDM ISA, routing ISA, fp8 ISA, and persistent ISA all
passed. Results: `results-persist.v7oVie` under the staging directory above.
All 42 emitted nonpersistent ISA instances (9 distinct kernels) match the
base's retained evidence, and P2/P4/P8 persistent ISA also matches that base.
Default auto P8 and explicit P8 are byte-identical: ISA SHA-256
`b1e4b37c247b62a77bd9d8cf39bda0638bfb2d19c9786046e94cc9c8e9a9daf9`.
P8: 34,816 B LDS, 16 tensor loads, 16 tensor waits (8 wait(0), 8 nonstalling
SGPR-reuse waits), 8 barrier pairs, 256 LDS b128 loads, 1,280 static bf16 WMMA
instructions across guarded/raw paths, 229 VGPR / 86 SGPR, zero spills/scratch.
Single-tile r32: 17,408 B LDS, 2 tensor loads, 2 tensor waits (1 wait(0)),
1 barrier pair, 32 LDS b128 loads, 183 VGPR / 79 SGPR, zero spills/scratch.

## Expected gain and next GPU validation

Root-supplied idle-window a07-3 measurements with P8 (not rerun here):

| bf16 prefill bucket | Single-tile us | P8 us | Latency reduction |
| --- | ---: | ---: | ---: |
| prefill512_bf16_c65536 | 133.0 | 105.2 | 20.90% |
| gate prefill_bf16_bm16 | 777.1 | 606.0 | 22.02% |
| gate prefill_bf16_bmge64 | 770.7 | 597.5 | 22.48% |

Root reports prior persistent-on GPU correctness 22/22 PASS. This round performs
CPU correctness/layout/ISA checks only; no new GPU correctness or speed claim.
Next: under the shared host GPU lock, repeat the 22-case correctness suite with
truly unset defaults and forced-off A/B, explicitly exercise P2/P4/P8 and
unaligned/narrow/mixed/partial/raw/cop fallbacks, then use pinned idle windows
for default-vs-off prefill measurements and decode/draft/fp8 regression checks.
Environment defaults are read at module import; A/B must use fresh processes.
