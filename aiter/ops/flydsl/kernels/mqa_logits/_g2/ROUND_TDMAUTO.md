# Round TDMAUTO: default padded TDM for bf16 r32, raw for r16

Base: `6802aec5f74ed607ba70e4f3be57c3402af19e44` (TDM2).

Only routing changes: `QSA_BF16_K_LOAD=raw|tdm|auto`, default `auto`.
The host and builder share one geometry resolver; specialization selection is
compile-time Python, not a runtime kernel branch. The existing 128-byte cache
alignment guard still selects the actual raw r16 specialization for an
unaligned TDM candidate. All 29 other top-level definitions and all 31 nested
WMMA kernel/helper source bodies compare byte-for-byte equal to TDM2.

| Mode | bf16 r16 decode/draft | bf16 r32 prefill |
|---|---|---|
| auto (default) | raw | padded TDM |
| raw | raw | raw |
| tdm | padded TDM | padded TDM |

Geometry remains `rows >= 32 && rows >= 32 * requests` for r32. Thus
64 rows / 16 requests (decode) and 16 / 16 (draft) stay raw, regardless of
column count; 512 / 1, 512 / 2 and 2880 / 2 prefill select r32 TDM.
Mixed/partial r32 workgroups retain the existing raw in-kernel fallback.
Scalar bf16, copy-atom access and fp8 behavior are unchanged.

## CPU-only verification

Reproduce with `host_compile_tdmauto_cpu.sh <absolute-source-archive>`.
Archive only the kernel, `_g2` tree and the two read-only public golden Python
modules; exclude `__pycache__`. No frozen/sealed data is modified or consulted.

Host: `heliosr-1b114-a07-3.mnb.dcgpu`. Image:
`rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, FlyDSL `0.3.3.dev886`.
The container has **no --device, --privileged, /dev/kfd or /dev/dri mount**.
`COMPILE_ONLY=1`, `FLYDSL_GPU_ARCH=gfx1250`, IR and final ISA dumps enabled.
All six scalar/WMMA bf16/fp8 specializations compiled in each mode (18 total).
`[COMPILED]` here means compiler-only success, not a GPU dispatch.

Results: `/data/blyu/qsa/work/opt-tdmauto-cpu-trace-20261031/results-tdmauto.8YimYH`.
`check_bf16_routing_isa_cpu.py` binds each compile-log dispatch name to its
expected kernel and checks actual final assembly, including raw zero-LDS and
padded-TDM strict ISA gates. Kernel prefix below:
`qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4`.

| Specialization suffix | LDS B | tensor loads | tensor waits | ds_load_b128 | barrier pairs |
|---|---:|---:|---:|---:|---:|
| `_r16_c64_flydsl` (auto/raw) | 0 | 0 | 0 | 0 | 0 |
| `_r32_c64_flydsl` (raw) | 0 | 0 | 0 | 0 | 0 |
| `_r32_c64_flydsl_tdm` (auto/tdm) | 17408 | 2 | 2 | 32 | 1 |
| `_r16_c64_flydsl_tdm` (forced tdm) | 69632 | 32 | 32 | 128 | 8 |

Raw r16/r32 contain 136/320 `buffer_load_b128` instructions. TDM paths retain
272-byte pitch (`0x110`), descriptor `0x07510000`, explicit wait(0) counts 1/4,
and zero `v_readfirstlane_b32`, `v_max_i64`, `ds_load_b128_addtid` markers.
Auto-r16 and forced-raw-r16 final ISA SHA256 are identical:
`3634e0083e2984c33e411b1bbfeba2fb33adb139cda8a6d1562891bb4db06ee8`.
Auto-r32 and forced-tdm-r32 final ISA SHA256 are identical:
`869941f5f06e47d0868d7152e923e146836afa3eaeb11934e1df0904f396212e`.

Existing contract, address, Q-LDS, raw trace16/trace32 (16 fixtures each),
and padded-layout K suite (28 tests) pass. New routing tests: **7/7 pass**.
The initial archive was staged before the new routing test was written; its
runner stopped at that missing test after all 18 compilations and existing CPU
gates passed. Uploading the test and rerunning the routing/strict-ISA gates in a
second no-device container using the same source/results completed with exit 0.
A final complete source archive is retained in the staging directory.
Routing tests execute the real
source's resolver/builder/host dispatch with CPU mocks; they are not numerical
GPU correctness evidence. Explicitly set `QSA_BF16_K_LOAD=raw` for trace32:
the trace mock models raw WMMA, not TDM shared-memory execution.

## Expected impact and remaining GPU plan

No GPU was used in this round; no new latency or GPU correctness is claimed.
The supplied prior TDM2 prefill-bf16 bm16 result was 956.4 -> 760.2 us
(-20.5%). Auto should inherit that r32 win while r16 decode/draft retain raw
performance (avoiding the supplied +8%, +10%, +16% forced-r16 regressions).
The supplied decode-c65536 gain is **not** a promise for auto: r16 geometry
remains raw regardless of that case name or column count.

Before rollout, acquire the cooperative GPU lock, run the full contract and
alignment/mixed/partial first-touch ladder in auto/raw/tdm, confirm actual
dispatch names/LDS sizes, then unprofiled A/A and auto-vs-raw A/B with held-out
reruns. Primary risks remain shape eligibility, mixed/partial fallback,
unaligned cache fallback and inherited TDM GPU synchronization/correctness;
CPU/ISA gates alone cannot prove any latency percentage.
