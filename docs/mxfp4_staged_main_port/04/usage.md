# MXFP4 search, compilation and replay

Run commands from the repository root with the ROCm/FlyDSL environment recorded
in [toolchain.json](../implementation/toolchain.json). The dedicated tuner is
`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py --mxfp4-flydsl`.

| Search option | Candidate universe | Pipeline-pair validation |
| --- | --- | --- |
| Omitted, or `--mxfp4-search-mode prune` | Main's M_est-pruned GEMM1 configurations and legal layout-v2 GEMM2 configurations | Exhaustive within that universe |
| `--mxfp4-search-mode full` | All statically legal main configurations | Exhaustive within that universe |
| `--mxfp4-search-mode staged` | The same pruned universe as `prune` | Independent stage screening, shortlist validation and group-local exhaustive fallback |

`M_est = ceil(token * routing_topk / expert)`. Staged fallback cannot recover
pairs excluded by M_est pruning. GEMM1 screening contexts are BM; GEMM2
screening and pipeline compatibility groups are BM plus atomic versus
non-atomic accumulation. Reduce and scatter share the non-atomic group, while
atomic retains its own shortlist. Screening scores rank stage candidates; only
complete, numerically valid pipeline-pair validation selects a winner.
An unusable reference, empty successful stage ranking or shortlist without a
valid finite pair triggers exhaustive validation of that group's admitted
pruned pairs. Returned pair outcomes are reused, so fallback does not repeat
completed validation attempts.

Screening top-K defaults to **2**. `--mxfp4-screen-topk` accepts any positive
integer and requires explicit `--mxfp4-search-mode staged`. It is separate from
the input CSV's `topk`, which is the model's routing top-k. Nonpositive K,
explicit K with prune/full or an omitted mode, and MXFP4 search options without
`--mxfp4-flydsl` are rejected before shape work. Do not combine these options
with `--grouped-gemm`.

## Inputs and result status

The supported contract is a runtime gfx950 GPU, BF16 caller input/output,
A4W4 with per-1x32 quantization, G1U1, separated gate layout and no stage1
routing weight. SiLU, SwiGLU and default Situv2 are supported. Native GEMM2
and re-tile pairs remain excluded; a non-256 inter dimension such as I384 is
eligible when legal layout-v2 pairs exist.

An input row can use this schema; this is the small SwiGLU smoke shape:

```csv
gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,gate_mode
gfx950,256,4,3072,512,128,4,ActivationType.Swiglu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,separated
```

Keep input-only `gate_mode` metadata when preparing requests. An unsupported
row is reported even if its shape already matches a tuned row. Unsupported or
empty candidate universes, failed rows and unfinished rows remain visible in
the summary and yield a nonzero final command status; later supported shapes
can still save valid winners. The production CSV retains the existing schema
and contains valid winners, with the assigned runtime GPU's gfx/CU identity.

All participating runtime devices are checked in prune, full and staged.
`GPU_ARCHS` and build metadata do not substitute for the assigned device's
properties. Staged gives each shape a fresh worker, including `--mp 1`, and
retains the result until that worker exits. Fatal worker death fails that
shape. Measurements on one logical GPU are exclusive within one tuning run;
separate tuning runs require caller coordination.

## Select the physical card and run a search

For this host's acceptance card, monitoring GPU7 has UUID
`0x5d8a72a332aa124c` and PCI `0000:f5:00.0`. Its selector is
`ROCR_VISIBLE_DEVICES=GPU-5d8a72a332aa124c` together with
`HIP_VISIBLE_DEVICES=0`. A bare HIP ordinal, including `HIP_VISIBLE_DEVICES=7`,
does not establish this physical identity.

[run_gpu_task.py](../scripts/run_gpu_task.py) checks that UUID/PCI, waits for
three idle samples, fixes visibility and verifies the child is logical GPU0 at
the same PCI with gfx950/256 CUs. Each invocation records idle samples, exact
command/environment/commit, runtime identity, console output and process
start/exit/status. Run independent tasks sequentially and wait for each to
exit. Use a fresh artifact directory for every task. The controller is fixed
to this host/card; its logical0 label is backed by those physical checks.

The following setup uses the recorded toolchain and separate example artifacts.
Choose a new `MXFP4_USAGE_ROOT` before repeating the examples:

```bash
export PATH=/root/workspace/tmp_dev/mxfp4-port-env/bin:"$PATH"
export PYTHONPATH="$PWD"
export CK_DIR=/root/workspace/tmp_dev/aiter-mxfp4-integration/3rdparty/composable_kernel
export AITER_USE_SYSTEM_TRITON=1
export GPU_ARCHS=gfx950
export FLYDSL_GPU_ARCH=gfx950
MXFP4_USAGE_ROOT=/root/workspace/tmp_dev/mxfp4-port-runtime/usage-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$MXFP4_USAGE_ROOT"
export AITER_JIT_DIR="$MXFP4_USAGE_ROOT/jit"
export FLYDSL_RUNTIME_CACHE_DIR="$MXFP4_USAGE_ROOT/tune-default"
export AITER_FLYDSL_STAGE2_FP8=0
export AITER_MXFP4_INTERMEDIATE=0
export AITER_MXFP4_TUNE_SWIGLU_LIMIT=
unset FLYDSL_RUNTIME_RUN_ONLY
```

This shell helper runs the same small input with a separate winner file and
task record for each search. The default warmup/iteration settings and
numerical gate are made explicit:

```bash
run_mxfp4_search() {
  local mxfp4_run_id="$1"
  shift
  python docs/mxfp4_staged_main_port/scripts/run_gpu_task.py \
    --artifact "$MXFP4_USAGE_ROOT/$mxfp4_run_id-task" --cwd "$PWD" \
    -- csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
    --mxfp4-flydsl --mp 1 --all --errRatio 0.1 \
    --warmup 5 --iters 101 --timeout 1800 \
    -i docs/mxfp4_staged_main_port/02/gpu/small-input.csv \
    -o "$MXFP4_USAGE_ROOT/$mxfp4_run_id.csv" "$@"
}

run_mxfp4_search small-prune
run_mxfp4_search small-full --mxfp4-search-mode full
run_mxfp4_search small-staged --mxfp4-search-mode staged --mxfp4-screen-topk 2
```

The small shape has M_est=1 and supplies a smoke check; it does not establish
BM128 scatter coverage or calibration performance. Use the prescribed shapes
and per-mode output files for calibration.

## Keep invocation settings and CSVs aligned

| Setting | Default | Enabled/configured behavior |
| --- | --- | --- |
| `AITER_FLYDSL_STAGE2_FP8` | Unset, empty or any value other than `1`: off | Exactly `1` enables FP8 route output for eligible reduce, including its final weighted reduction |
| `AITER_MXFP4_INTERMEDIATE` | Unset, empty or any value other than `1`: off | Exactly `1` enables packed FP4 scatter for eligible BM128, BN128/BN256, no-bias pairs; bias scatter retains BF16 |
| `AITER_MXFP4_TUNE_SWIGLU_LIMIT` | Unset/empty: activation defaults | A positive finite value configures the SiLU/SwiGLU clamp; Situv2 rejects a configured limit |

Default SwiGLU limit is 7; default SiLU and Situv2 are unclamped. With both
output switches enabled, reduce takes FP8 and scatter takes FP4. Atomic retains
its established output handling. An enabled switch does not make every pair
eligible for that output path.

The tuner captures these three effective settings once per invocation, before
input selection, and passes the snapshot to workers. Keep their environment
values unchanged in the parent and workers until the run ends. Production
still reads the two switches on each wrapper call. Set matching values before
starting each independent tuning, AOT or replay process; use matching explicit
activation parameters in production.

The caller associates winner files with those settings. CSV shape/device keys
and kernel names do not encode the limit or output switches. If a file already
contains a **same-key default-limit winner**, a limit-5 invocation **without
`--all` can silently skip it**. With `--all`, it retunes and **replaces that row
when a valid new winner is saved**. Changing output flags has the same
collision. `--all` forces retuning; it does not isolate configurations. Choose
the intended configuration's separate winner file. Separate files per search
mode also ensure full, prune and staged each execute their requested search.

For a limit-5, both-switches GPT-OSS search, prepare an input with its supported
A4W4 metadata and write a distinct result:

```bash
cat > "$MXFP4_USAGE_ROOT/gpt2048-input.csv" <<'CSV'
gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,gate_mode
gfx950,256,2048,3072,512,128,4,ActivationType.Swiglu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,separated
CSV
export AITER_FLYDSL_STAGE2_FP8=1
export AITER_MXFP4_INTERMEDIATE=1
export AITER_MXFP4_TUNE_SWIGLU_LIMIT=5
export FLYDSL_RUNTIME_CACHE_DIR="$MXFP4_USAGE_ROOT/tune-limit5-both"

python docs/mxfp4_staged_main_port/scripts/run_gpu_task.py \
  --artifact "$MXFP4_USAGE_ROOT/gpt-limit5-both-staged-task" --cwd "$PWD" \
  -- csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  --mxfp4-flydsl --mxfp4-search-mode staged --mxfp4-screen-topk 2 \
  --mp 1 --all --errRatio 0.1 --warmup 5 --iters 101 --timeout 1800 \
  -i "$MXFP4_USAGE_ROOT/gpt2048-input.csv" \
  -o "$MXFP4_USAGE_ROOT/gpt-limit5-both-staged.csv"
```

## Compile and verify exact production replay

AOT must use the selected CSV, the same output switches and matching effective
limit. Standalone MXFP4 AOT accepts `--swiglu-limit`; it takes precedence over
`AITER_MXFP4_TUNE_SWIGLU_LIMIT`. Omitted standalone and batch callers read that
environment variable. The default batch corpus contains Situv2 rows, so select
compatible CSVs for a custom-limit batch.

Continuing the limit-5, both-switches example, first compile into a fresh cache:

```bash
export FLYDSL_RUNTIME_CACHE_DIR="$MXFP4_USAGE_ROOT/aot-limit5-both"
unset FLYDSL_RUNTIME_RUN_ONLY

python docs/mxfp4_staged_main_port/scripts/run_gpu_task.py \
  --artifact "$MXFP4_USAGE_ROOT/gpt-limit5-both-aot-task" --cwd "$PWD" \
  -- -m aiter.aot.flydsl.mxfp4_moe \
  --csv "$MXFP4_USAGE_ROOT/gpt-limit5-both-staged.csv" --swiglu-limit 5
```

Every selected job must compile successfully. To use the existing batch worker
pool instead, substitute this task for the standalone task:

```bash
python docs/mxfp4_staged_main_port/scripts/run_gpu_task.py \
  --artifact "$MXFP4_USAGE_ROOT/gpt-limit5-both-batch-aot-task" --cwd "$PWD" \
  -- docs/mxfp4_staged_main_port/scripts/compile_selected.py \
  --csv "$MXFP4_USAGE_ROOT/gpt-limit5-both-staged.csv"
```

It reads the exported limit 5; it also accepts `--swiglu-limit 5` explicitly
and multiple selected CSV paths. See the
[AOT guide](../../../aiter/aot/flydsl/README.md) for worker controls. The C++
auxiliary sort/scatter reduction build is separate from the FlyDSL cache.

In a later independent process, use that same AOT cache in run-only mode:

```bash
export FLYDSL_RUNTIME_RUN_ONLY=1

python docs/mxfp4_staged_main_port/scripts/run_gpu_task.py \
  --artifact "$MXFP4_USAGE_ROOT/gpt-limit5-both-precompiled-task" --cwd "$PWD" \
  -- docs/mxfp4_staged_main_port/scripts/configured_replay.py \
  --csv "$MXFP4_USAGE_ROOT/gpt-limit5-both-staged.csv" \
  --result "$MXFP4_USAGE_ROOT/gpt-limit5-both-precompiled.json" \
  --action replay --swiglu-limit 5 --warmup 5 --iters 101
```

The replay harness takes a canonical one-row winner. It calls public
`aiter.fused_moe.fused_moe` with BF16 input/output, per-1x32 FP4 weights/scales,
`gate_mode="separated"`, explicit `quant_dtype_a=dtypes.fp4x2` and
`swiglu_limit=5`. It verifies the exact CSV G1/G2 names and metadata, actual
input quantization/prequantization and effective output/reduction arguments,
and final numerics against Torch. An `OK` from heuristic kernels or first-call
JIT compilation does not establish exact precompiled replay. Small calls can
otherwise select A16W4, so retain the explicit A4W4 input contract.

For a default-environment winner, reset both output switches to 0 and the
tuning-limit variable to empty before starting separate default AOT/replay
tasks; omit `--swiglu-limit`. Tuner Situv2 uses `(beta, linear_beta)=(4,25)`;
production replay of that tuning contract must pass those values. Public
Situv2's omitted `(1,1)` parameters are a separate execution contract and must
be verified separately. AOT retains both existing Situv2 parameter variants.

## Interpret measurements and timeouts

MXFP4 `--errRatio` defaults to **0.1** and requires finite error. The metric is
`1 - 2 * dot(ref, out) / (norm(ref)^2 + norm(out)^2)`, not a relative-error
percentage. CSV `err1`/`err2` format that metric as percentages. Every output
path, including lossy FP4 scatter, uses the same ordinary Torch reference and
configured gate. Only finite positive validated latency can win.

Saved `us` is the main-compatible **device kernel-duration sum** for the
complete pipeline, including applicable sorting/quantization and final output
handling. The coupled-pair tuner writes this value to `us1` and `us`, with
`us2=0`; these fields are not isolated-stage timings. Screening scores and
untimed reference bootstrap never become final CSV timings. `--profile_file`
is an ordinary fMoE profiling option: the dedicated `--mxfp4-flydsl` tuner does
not write stage-ranking or full-pair profile artifacts.

Tuning wall time includes preparation, reference/correctness work,
compilation/cache effects, warmup, profiling, screening and failed attempts.
Synchronized replay elapsed time and the span of real kernel trace events are
separate companion metrics. Exclude synthetic GPU annotations from real
kernel counts and duration diagnostics. Do not infer tuning wall time by
multiplying the winner's kernel latency by static pair count.

`--timeout` defaults to **1800 seconds** and is a best-effort per-operation
SIGALRM timeout; a nonpositive value disables it. Returned timeout/failure
outcomes can participate in staged recovery, but signal handling needs control
to return to Python. There is no parent/shape hard deadline. A still-live GPU
operation that hangs permanently may wait indefinitely, including under the
acceptance controller. A fatal worker death is a shape-level failure rather
than a recoverable pair outcome.

The [prescribed acceptance](../../../.scratch/mxfp4-staged-main-port/acceptance.md)
defines 27 initial calibration searches at K2: seven MiniMax prune/staged
pairs plus full anchors at tokens 64/2048/8192, and five GPT-OSS environments
each prune/staged. Keep inputs/routes, warmup/iterations, physical card and
cache conditions comparable and retain every paired outcome, including missing
winners. Report full-to-prune, prune-to-staged and full-to-staged only where
matching valid results exist; absent full data makes a full comparison
unavailable. Static candidate counts establish admitted proposals, not runtime
validity. The port specifies no hard regret or tuning-speedup threshold.

[run_calibration.py](run_calibration.py) is the sole coordinator for the
prescribed matrix. From the intended checkout, launch the initial 27 searches
with the recorded toolchain:

```bash
/root/workspace/tmp_dev/mxfp4-port-env/bin/python \
  docs/mxfp4_staged_main_port/04/run_calibration.py \
  --root "$PWD" --python /root/workspace/tmp_dev/mxfp4-port-env/bin/python \
  --actions search
```

`--root` defaults to the checkout containing the script; `--python` selects
the toolchain Python for child tasks. The coordinator reads the manifest,
chooses separate mode/configuration winners, fixes the settings for each task
and runs fresh independent processes sequentially through the physical-GPU
idle/identity controller. Leave `--only` omitted for the prescribed matrix;
selected IDs are for review or diagnosis. Existing successful task records
are retained rather than executed again; a failed/incomplete prior task
requires diagnosis.

After search, verify the matching winners in separate deployment tasks:

```bash
/root/workspace/tmp_dev/mxfp4-port-env/bin/python \
  docs/mxfp4_staged_main_port/04/run_calibration.py \
  --root "$PWD" --python /root/workspace/tmp_dev/mxfp4-port-env/bin/python \
  --actions replay aot precompiled
```

These replay/AOT/run-only tasks are deployment checks, separate from the 27
initial searches. Use the same runtime/cache root for AOT and precompiled
replay; override `--runtime` consistently if using a separate runtime directory.
