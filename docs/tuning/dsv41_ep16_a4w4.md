# DSV4.1 Flash EP16 A4W4 tuning

The DSV4.1 table covers hidden size 5120, intermediate size 2304, BF16 output,
MXFP4 activations and weights, per-32 scaling, SiLU, G1U1, and routing weights
applied in stage 2. It contains two local-expert configurations:

| Local experts | Compute top-k | Deployment interpretation |
| --- | --- | --- |
| 24 | 6 | 384 routed experts distributed across 16 EP ranks |
| 25 | 7 | 24 routed experts plus one homogeneous shared expert replicated on each rank |

For the shared configuration the router still selects six routed experts. The
shared expert is appended as a seventh, always-active slot with weight 1. This
requires a shared expert with the same intermediate size and quantization as
the routed experts. In a local-replica layout, logical routed ID `e` maps to
physical ID `(e // 24) * 25 + e % 24`; each rank's last slot is shared. There are
400 physical slots, including 16 shared replicas, and 384 logical routed experts.

These rows use generic `flydsl_moe1_*` and `flydsl_moe2_layout_*` kernels. The
previous `flydsl_mxmoe_g1_*` rows were rejected by runtime dispatch: the generated
MXFP4 auxiliary shape list did not cover `(24, 5120, 2304, 6)`, and its
`output_aux` sorting path drops `expert_mask`. The generic path supports the
global-to-local expert mapping and filtering required by EP.

## Original tuning source

The original nine rows came from the November 3, 2026 MXFP4 port tuner run in
`/home/hzm/e2e_rel/aiter`, with input and output archived under
`/home/hzm/d41tune/`. Both its input CSV and log explicitly specify **24 experts
and top-k 6**, rather than a full-model expert count. Representative raw output:

```text
[mxfp4-port] tuning 9 shapes across 8 GPUs
[mxfp4-port] shape token=1024 inter=2304 expert=24 topk=6 -> GPU2
[mxfp4-port] pruned G1: pid=778480 token=1024 model_dim=5120 inter_dim=2304 expert=24 topk=6 act=silu M_est=256 valid=84 kept=12
```

The archived input, output and log have these SHA-256 hashes:

```text
input: 3fcf27ffe41be52e17ed741537a57ce266f1b54f70ebff09a910200476fbb4f9
output: baa36a1526ffd569f3e644a39df620da789e9434d88a531f951dad85e7490ef7
log: f770258c35dc656cb6f841f9e59aaa4769e2e8c7e9435afeaf572253d5c71f07
```

Those timings measured the coupled MXMOE path; they are not performance
measurements of the EP-mask path. They are retained here as source provenance:

| Tokens | Original coupled time (us) |
| ---: | ---: |
| 256 | 133.2561 |
| 512 | 153.8829 |
| 1024 | 218.0010 |
| 2048 | 350.8814 |
| 4096 | 621.6381 |
| 8192 | 1131.0551 |
| 16384 | 2168.5379 |
| 32768 | 4390.3162 |
| 131072 | 17574.1972 |

## Generic-kernel retuning

The new run uses AITER revision `9815cf8619e373573dfe4c85323ccfb878373361`
on AMD Instinct MI355X (`gfx950`, 256 CUs), ROCm 7.2.0, Python 3.12,
PyTorch `2.9.1+rocm7.2.0.git7e1940d4`, and **FlyDSL 0.4.2**, as required by
this AITER revision. FlyDSL was installed into an isolated task environment.

The [input CSV](dsv41_ep16_a4w4_input.csv) contains all 18 shapes. Run from the
AITER repository root with its dependencies installed:

```bash
TUNE_ONLY=flydslv2 \
TUNE_MOE_KERNEL_REGEX='(^flydsl_moe1_afp4_wfp4_bf16_t(32|64|128)x(64|128|256)x256(_w[234])?(_bnt0)?_fp4$)|(^flydsl_moe2_layout_afp4_wfp4_bf16_t(32|64|128)x(128|256)x(128|256)_(atomic|reduce)(_persist)?_sbm(32|64|128)$)' \
AITER_MOE_EXPERT_BALANCE=true \
python -u csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i docs/tuning/dsv41_ep16_a4w4_input.csv \
  -o /path/on/large/storage/dsv41_tuned.csv \
  -o2 /path/on/large/storage/dsv41_profile.csv \
  --mp 8 --batch 1 --warmup 5 --iters 30 --timeout 900 \
  --errRatio 0.05 --verbose
```

The first eight 24/6 shapes used the unmodified reference function. For the
remaining shapes, the measurement worktree applied the
[reference-gather patch](dsv41_sorted_reference.patch):

```bash
git apply --unidiff-zero docs/tuning/dsv41_sorted_reference.patch
```

This changes only the untimed Python `v2_stage1_sorted_ref` gather, batching
8192 rows at a time instead of synchronizing several times per routed row.
The timed kernels are identical. The patched reference was bitwise equal to
the original on CPU and gfx950 for 97 and 8321 sorted rows, top-k 7 and 6,
block sizes 32 and 128, with packed IDs, padding, missing experts and repeated
expert IDs. The patch is a reproduction artifact; it is not applied to the
library by this PR. The unmodified reference also produces the same results,
but is much slower at the largest token bucket.

This is a bounded generic-kernel search, not an exhaustive search of every
FlyDSL variant. The tuner measures GPU kernel time and checks each candidate
against its stage reference. Its generated routes are balanced over **local**
experts, with no EP communication or mask. The 25/7 tuning workload also routes
uniformly over local experts; it does not model the shared expert's always-active
load. Separate runtime tests below exercise that behavior.

All 18 shapes produced valid selected pairs. The selected stage error columns
are 0.9% and 1.2%, respectively, within the tuner's 5% threshold. GPU stage
times (us) are below; `sum` is the two measured stage times, and excludes
initial sorting/input quantization and EP communication. Full kernel names
are recorded in the [runtime CSV](../../aiter/configs/model_configs/dsv41_fp4_tuned_fmoe.csv).

| Tokens | 24/6 stage 1 | 24/6 stage 2 | 24/6 sum | 25/7 stage 1 | 25/7 stage 2 | 25/7 sum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 54.7394 | 37.8731 | 92.6125 | 64.9955 | 43.7273 | 108.7228 |
| 512 | 71.7422 | 48.4467 | 120.1889 | 117.3389 | 67.3787 | 184.7176 |
| 1024 | 127.2939 | 83.1075 | 210.4014 | 168.4875 | 101.077 | 269.5645 |
| 2048 | 220.8013 | 149.7224 | 370.5237 | 260.4222 | 174.6339 | 435.0561 |
| 4096 | 387.8369 | 267.6865 | 655.5234 | 443.4186 | 317.7002 | 761.1188 |
| 8192 | 735.4447 | 512.8291 | 1248.2738 | 841.5472 | 595.0237 | 1436.5709 |
| 16384 | 1377.6834 | 982.1297 | 2359.8131 | 1563.0758 | 1145.4017 | 2708.4775 |
| 32768 | 2564.7096 | 2077.5136 | 4642.2232 | 3087.5726 | 2429.0975 | 5516.6701 |
| 131072 | 10076.8464 | 8239.2597 | 18316.1061 | 12139.3644 | 9770.5243 | 21909.8887 |

## Runtime validation

```bash
AITER_LOG_LEVEL=INFO python -m pytest -q -s \
  op_tests/test_moe_dsv41_fp4_ep_dispatch.py
```

The test selects each of the 18 rows with `is_ep=True` and checks the configured
stage names and sorting block size. Its numerical cases call public `fused_moe`
with a real `expert_mask`, shuffled MXFP4 weights and SiLU limit 10, for every
token bucket in both configurations. Routed IDs include both local and remote
experts. The 24/6 case also includes a masked sentinel ID; the 25/7 case appends
the last rank's shared physical ID 399 to every token with weight 1.

The independent Torch reference samples 32 tokens across all 24 routed local
experts and reproduces global-to-local mapping and both MXFP4 quantization
boundaries. The checks require finite full outputs and mean sampled row cosine
error below 0.01. Full validation tensors and profiling data stay on remote
large storage; no tensor dumps are needed to run these tests.


The gfx950 run completed **36 tests in 6.85 seconds** (18 selection cases and
18 numerical cases). All numerical calls selected their configured kernel
pairs, with no discarded rows or heuristic fallback. The maximum sampled mean
row cosine error was 0.00080180, below 0.01. Actual log excerpts from the
public `fused_moe` calls with non-null EP masks:

```text
[aiter] [fused_moe] using 2stage (kernelName1='flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w3_fp4', kernelName2='flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_reduce_sbm64') for ('gfx950', 256, 256, 5120, 2304, 24, 6, <ActivationType.Silu: 0>, 'torch.bfloat16', 'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False)
EP16 token=256 local_experts=24 topk=6 shared=False cosine_error=0.00067407
[aiter] [fused_moe] using 2stage (kernelName1='flydsl_moe1_afp4_wfp4_bf16_t128x128x256_w2_fp4', kernelName2='flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_reduce_sbm128') for ('gfx950', 256, 256, 5120, 2304, 25, 7, <ActivationType.Silu: 0>, 'torch.bfloat16', 'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False)
EP16 token=256 local_experts=25 topk=7 shared=True cosine_error=0.00066125
36 passed in 6.85s
```

| Tokens | 24/6 cosine error | 25/7 cosine error |
| ---: | ---: | ---: |
| 256 | 0.00067407 | 0.00066125 |
| 512 | 0.00060189 | 0.00064373 |
| 1024 | 0.00076020 | 0.00080180 |
| 2048 | 0.00075281 | 0.00058550 |
| 4096 | 0.00078809 | 0.00070667 |
| 8192 | 0.00046819 | 0.00063276 |
| 16384 | 0.00068164 | 0.00077248 |
| 32768 | 0.00071335 | 0.00067782 |
| 131072 | 0.00058728 | 0.00080013 |

This validates one EP rank's compute on a GPU. Full 16-GPU ATOM deployment,
inter-node dispatch/combine, and end-to-end serving latency were not measured.
