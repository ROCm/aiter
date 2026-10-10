# Ticket 02 default real-GPU acceptance

The coordinator ran four independent tasks sequentially on physical GPU7:
`ROCR_VISIBLE_DEVICES=GPU-5d8a72a332aa124c`, `HIP_VISIBLE_DEVICES=0`, verified
PCI `0000:f5:00.0`, gfx950, 256 CUs. Each task retains its idle samples, ending in three consecutive idle checks, and
its runtime-device verification, exact command/environment, process start/exit,
console and final result. A busy observation before a task was waited out on
this card; no other card was selected. All four tasks exited 0.

The prescribed small shape is token4/H3072/I512/E128/routing-top-k4, BF16 caller
input/output, explicit FP4 A/W per-1x32, separated gates, Swiglu default limit7,
no stage1 route weighting. The configured output switches were both0 and tuning
limit environment empty. Random tensors/routes use Torch seed0; production
JSON retains input/ids/weights SHA256 fingerprints.

| Task | Implementation | Result |
| --- | --- | --- |
| Small staged CLI | `09c9b5ef2` | 12 G1 screens, 32 G2 screens, 8 validations, no fallback; one canonical winner; summary107.2228 seconds |
| Exact public production | `85dcad12c` | cosine_diff0.011169739046099525, finite, exact selected names/parameters, explicit FP4 input contract |
| Cold default AOT | `85dcad12c` | all4 selected-CSV jobs compiled; compile action wall8.183971 seconds |
| Precompiled public replay | `85dcad12c` | `FLYDSL_RUNTIME_RUN_ONLY=1`, cosine_diff0.011173635721687503, finite, exact names/parameters from new AOT cache |

Winner G1 is `flydsl_mxmoe_g1_a4w4_16x128x256_f16in_hpf_nt_swiglu`; G2 is
`flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16`. The canonical
28-column winner CSV writes end-to-end kernel-duration sum15.3478 us to `us1`
and `us`, `us2=0`, error1.1% in its existing representation. The winner's logged
validation error was0.011167. Independent screen timings remain in the console
and are ranking evidence, with no additive latency claim.

The actual public operation observation verifies raw BF16 inline G1 quant at
BM16, FP4 intermediate payload/native BM16 scales and FP4A/W→BF16 atomic G2,
plus parsed launch fields (including hpf, NT, tiles/waves/k_wave/XCD). Exact CSV
metadata lookup supplies both names and BM/fuse_quant, so a heuristic fallback
cannot pass this acceptance. C++ auxiliary sort is the expected BM16 prologue.

Search used an isolated external FlyDSL cache `search-default`; default AOT was
built into a new `aot-default` cache and precompiled replay used only that cache.
`small-aot.json` retains all4 jobs: separated/interleaved G1 cache variants and
no-bias/bias G2 variants. The replay consumes the separated/no-bias entries.
A shared external auxiliary JIT directory holds required C++ setup/sort modules;
those are recorded separately from FlyDSL AOT and are not tuning-cache evidence.

Public replay also retains compressed complete Chrome traces and matched
synchronized per-call diagnostics. Default production's synchronized diagnostic
kernel sum with main aggregation was19.298556 us, real-kernel trace span186.024050
us/call and median synchronized elapsed184.495933 us. These calls synchronize
after each iteration; the values describe that diagnostic contract separately
from the continuous tuner kernel-duration sum and from serving latency. Synthetic
GPU annotation events are excluded;303 real events=3 kernels×101 iterations.

GPU/toolchain/environment records accompany each task. Full27-search calibration,
I384/BM32/large-token/output-mode acceptance and Situv2 production belong to the
remaining tickets; this small M_est1 fixture does not establish scatter coverage.
