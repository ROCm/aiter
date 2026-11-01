# ROUND-CVTDEF — native FP8 conversion becomes the default

Base: `4a5dff18978c3b1ee96ba980dd1691d84e4f237c`.
Branch: `qsa-index-scorer/opt-cvtdef`. No push or GPU execution in this round.

The sole executable production change is the import-time default of
`QSA_FP8_CVT`, from `current` to `fast`. Both values remain accepted after
whitespace/case normalization; invalid values still raise `ValueError`.
Set `QSA_FP8_CVT=current` before import for the byte-identical legacy A/B path.
Unset/default and explicit `fast` produce the same native specialization.

Only full-r32 single-request padded-TDM FP8 fragments use native conversion.
FP8 scalar/r16/raw, mixed/tail fallback, BF16, geometry, masks, descriptors,
i64 extents, scales, head-sum and stores are unchanged. The converter itself is
unchanged from ROUND-FP8CVT.

## Validation

No-device stock container on a07-3:
`rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, FlyDSL `0.3.3.dev886`,
`COMPILE_ONLY=1`, `gfx1250`, network none, no device flags; scripts assert
`/dev/kfd` and `/dev/dri` are absent. Staging:
`/data/blyu/qsa/work/opt-cvtdef-cpu-trace-20261101`.

The conversion compiler covers truly unset/default, explicit current and fast
in separate processes/caches, including helper micro ISA. The strict ISA gate
requires default == fast byte-for-byte and current == champion byte-for-byte.
The complete historical 15-gate CPU/ISA suite pins explicit current to preserve
its historical hash checks. Independent default compilation checks every
scalar/BF16/r16/raw file against champion. AST tests execute the real default,
validation and routing statements without importing GPU packages.

Default/fast production ISA SHA256:
`7d54e9e766bb6dfe6a7450b57ac80a2cfb1b4f93d1f431994b0a13e5e93d01e6`.
Current ISA SHA256:
`f68c8c7719649287116e8a15cb9146a2deb92e528a16a5648caaf536f182c667`.
Expected resources are unchanged from ROUND-FP8CVT: fast/current VGPR 168/193,
SGPR 83/83, LDS 9216/9216, scratch 0/0, preserve-denormal mode 3/3.
Regenerated machine-readable results are in
[fp8cvt_isa_evidence.json](<aiter/ops/flydsl/kernels/mqa_logits/_g2/fp8cvt_isa_evidence.json>)
and [fp8cvt_cpu_evidence.json](<aiter/ops/flydsl/kernels/mqa_logits/_g2/fp8cvt_cpu_evidence.json>).

## Measured prior and GPU plan

Root reported idle-window paired fresh-process current/fast A/B (three legs
per mode) on a07-3: prefill512_fp8_c65536 median 158.9 -> 147.6 us (-7.1%).
Other paths were unchanged apart from idle-window noise. Root also reported
22/22 GPU correctness PASS with explicit fast. These are supplied measurements,
not executions or new measurements by this CPU-only round.

Parent/root should rerun default correctness and an unset-vs-current fresh-
process A/B under the cooperative GPU lock, confirm the `_cvtfast` dispatch,
and retain explicit current as rollback. CPU/ISA identity predicts the already-
measured fast behavior but does not replace device validation. Import-time
selection requires a fresh process for reliable A/B; hardware FP8 subnormal
semantics and schedule behavior are unchanged from the previously tested fast
path. Historical ROUND-FP8CVT documentation describes the old opt-in round.
