# GEMM tuning

Use one driver for Triton and Gluon GEMMs. Each case in `gemm_cases.py` builds
inputs and calls the public wrapper; `tune_gemm.py` searches, checks, profiles,
and writes the config returned by the wrapper's normal `get_gemm_config()`
lookup. No harness or output-name table is needed per kernel or architecture.

Run from an AITER source checkout with its test dependencies installed. Actual
tuning requires a supported AMD GPU, ROCm, Triton, and
[`rocprofv3`](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
on `PATH`.

```bash
cd aiter/ops/triton/utils/_triton/tuning
python3 tune_gemm.py --list  # lists cases and arguments without importing GPU code
HIP_VISIBLE_DEVICES=0 python3 tune_gemm.py gemm_a8w8 M=16 N=1024 K=1024
HIP_VISIBLE_DEVICES=0 python3 tune_gemm.py gemm_a8w8 M=16 N=1024 K=1024 --backend gluon
HIP_VISIBLE_DEVICES=0 python3 tune_gemm.py batched_gemm_bf16 M=32 N=128 K=512 B=4
```

The wrapper must support the requested backend on the running GPU. The same
command can tune gfx942, gfx950, gfx1250, and future architectures, wherever
the kernel and its `DEFAULT.json` exist. Run it on each target GPU; it does
not estimate one architecture's performance using another GPU or copy tuned
results between backends. See [COVERAGE.md](COVERAGE.md) for the list of kernels
that lacked harnesses, supported cases, shared families, and remaining gaps.

## What a tuning run does

1. Run the installed config and take a snapshot of all its output tensors.
   A failed baseline stops checked tuning. The comparison is against the
   installed implementation; it supplements the kernel's independent unit
   tests, rather than proving that the baseline itself is correct.
2. Try candidate configs at the same lookup, retaining the wrapper's own
   allocations, split-K handling, and backend dispatch. Compare every tensor
   with the baseline using a 5% relative L2 tolerance for floating outputs and
   exact equality for integer outputs such as quantization scale bytes. Record failed configs
   in `errors-<op>-<shape>.txt`. A resource failure rejects that candidate;
   changing stages or warps may make another config with the same tiles fit.
3. Rerun the best candidate and installed config together, with output checks.
   Write a winner only after this comparison succeeds and the measured gain
   exceeds 3%. If the baseline wins, keep an existing specialized config;
   when no specialized file exists, record the validated baseline for the
   shape. `--no-check` only reports exploratory timings and never writes.

`profile_configs.py` runs under `rocprofv3 --kernel-trace` in batches of 100
configs. A crash or timeout is isolated to a worker process. Each config is
measured over 250 runs with L2 flushed before each run. Trace markers separate
runs and configs; each run sums the GEMM kernels, including repeated launches,
split-K reduction, and fused feed-forward kernels. Warmup, cache flush, marker,
output initialization, and other wrapper work are excluded. The result is the
median kernel time, not end-to-end wrapper latency.

## Search space

The keys come from the target architecture/backend's `DEFAULT.json`. Common
Triton tile and launch parameters have shared ranges; published values for the
same family/backend are also candidates. Other keys, including Gluon buffer
counts and nested variants, start from values already in that family's JSON
files. Authors can supply new values without changing the driver:

```bash
python3 tune_gemm.py gemm_a8w8 M=16 N=1024 K=1024 \
    --space BLOCK_SIZE_M=16,32 BLOCK_SIZE_K=64,128 num_warps=4,8
```

A case can constrain its default search with `@gemm_case(space=...)`, for
example `BLOCK_SIZE_K=128` for blockscale GEMMs. Explicit `--space` values
override those choices. A complete Cartesian product can be expensive; use a
small legal search first and widen the parameters that matter. The author
owns parameter names and legal combinations; the driver does not translate
Triton keys into Gluon keys or silently repair a kernel's schema.

| Option | Meaning |
| --- | --- |
| `--list` | List registered cases, dimensions, and backend selection without a GPU |
| `--backend triton\|gluon` | Select an exposed backend; otherwise use the wrapper's default |
| `--space KEY=V1,V2 ...` | Override candidate values; numbers, booleans and null use JSON syntax |
| `--timeout SECONDS` | Timeout for each profiling batch; default 900 seconds |
| `--no-check` | Explore timings without checking outputs; never save configs |

## Where results go

The observed lookup determines the family, backend, logical shape and filename:

```
configs/<arch>/<backend>/gemm/<family>/GEMM-<name>-N=<N>-K=<K>.json
```

Batch-specific `B=` and fused custom suffixes follow the loader's precedence.
The wrapper's logical K is used, including FP4's packed-to-logical conversion.
A new file inherits the table currently resolving for the shape, including a
non-batched specialized table when appropriate; otherwise it starts from
`DEFAULT.json`.

The winner replaces the smallest `M_LEQ_<bound>` at or above M, using the
caller's bounds, file `M_BOUNDS`, or standard bounds in that order. Above the
largest bound it replaces `M_GEQ_<largest bound>`. This tunes an **M bucket**:
other M values in that bucket also receive the winner. Other entries are
preserved. Tune representative shapes at the bounds and run the affected
kernel tests before submitting the resulting JSON.

Writes lock and reload the destination before atomically replacing it, so
concurrent jobs for different buckets preserve each other's updates. Avoid
simultaneous tuning of the same bucket. Restart production Python processes
after installing results because config reads are cached. Review with
`git diff -- aiter/ops/triton/configs` from the repository root.

## Adding a kernel, backend, or architecture

1. Route the wrapper through `get_gemm_config()` without an explicit `config`.
   Keep each family tied to one compatible config contract. The generic tuner
   expects one distinct lookup per call; tune composed operations through
   their constituent GEMMs.
2. Add valid `DEFAULT.json` under the target architecture/backend, with the
   keys the kernel actually consumes, including nested variant configs. Keys
   do not need to be identical across architectures or backends. Provide the
   implementation on the target GPU. The driver does not implement backend
   fallback and rejects a lookup that differs from an explicitly requested
   backend; the wrapper's dispatch remains authoritative.
3. Add a case named for the wrapper (or its named variant). Reuse the unit
   test's input generator, return a callable that invokes the wrapper with
   no `config=`, and return all its tensor outputs. Reset accumulated outputs
   inside the callable. Forward `backend=None` through `backend_kwarg` when
   the wrapper exposes it. Include persistent, shuffled and other separately
   configurable variants. Imports stay inside the case for GPU-free listing.
4. Specify constrained or new search values in the case's `space`, existing
   family JSON, or a documented `--space` example. Timeable kernels use names
   containing `gemm` or `_ff_`, including reduction kernels. Keep input packing
   and shuffling identical to production and its correctness tests. Set
   `@gemm_case(kernel_names=("ff_a16w16_fused",))` for a family whose trace
   names do not contain `gemm`; include its reduction kernels too.
5. Update this README, [COVERAGE.md](COVERAGE.md),
   [the Triton README](../../../README.md), and
   [Copilot review instructions](../../../../../../.github/instructions/aiter-ops-triton.instructions.md)
   when the author contract changes. Follow [config rules](../../../configs/CLAUDE.md)
   for JSON placement and architecture seeding.

```python
@gemm_case(space={"BLOCK_SIZE_K": [128]})
def new_gemm(M, N, K, backend=None):
    # Import the wrapper and its test input generator here.
    inputs = generate_inputs(M, N, K)
    return lambda: op(*inputs, **backend_kwarg(backend))
```

MoE dispatch tables use a different lookup and tuning contract; they are
listed separately in the coverage inventory. They should reuse this driver's
profiling/checking concepts if unified later, but do not write GEMM bucket
configs into MoE tables.

## Testing the tooling

The CPU regression suite exercises the real config loaders with stub GPU
modules and synthetic profiler results. It does not benchmark a GPU:

```bash
python3 -m unittest discover -s op_tests/triton_tests/gemm -p test_tune_gemm.py
```

Run the relevant `op_tests/triton_tests/gemm/` tests on each target GPU after
tuning. The new explicit FP4 preshuffle backend test covers Triton and Gluon
on gfx1250.
