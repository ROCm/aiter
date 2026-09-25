# Triton and Gluon GEMM tuning scripts

Tunes any GEMM that has a case in `gemm_cases.py`, on any arch and either
backend. Run every command from this directory, on the GPU you are tuning for.

| File | What it does |
| --- | --- |
| `sweep_configs.py` | Profiles every candidate config for one GEMM at one shape and logs each runtime |
| `write_best_configs.py` | Picks the fastest config per `M` from those logs and writes the config file |
| `verify_configs.py` | Profiles a GEMM with the config the library resolves and shows which file it came from |
| `gemm_cases.py` | One function per GEMM: builds the inputs and returns a callable that runs the op |
| `_worker.py` | The process the scripts run under `rocprofv3` |
| `parse_kernel_trace.py` | Reduces a `rocprofv3 --kernel-trace` CSV to median kernel runtimes |
| `_utils.py` | Helpers shared by the three scripts |

## How it works

The scripts never pass `config=` to a wrapper. They run the op the way
production does and answer its own `get_gemm_config()` lookup with the
candidate config instead of the JSON. That lookup also tells them everything
else:

- **what to tune**: the keys of the family's `DEFAULT.json` for this arch and
  backend. Standard Triton keys (`BLOCK_SIZE_*`, `GROUP_SIZE_M`, `num_warps`,
  `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`, `cache_modifier`,
  `NUM_KSPLIT`) get the usual ranges, bounded by the shape; any other key
  (Gluon tiles, buffer counts, kernel variants, ...) gets the values the
  family's files for that backend already use, on any arch. `kpack` is only
  tuned on gfx942 and dropped elsewhere (`configs/CLAUDE.md`). The config the
  library resolves today runs as a candidate too, so a winner has to beat it.
- **where the result goes**: the config family, the backend the wrapper
  picked, the arch, and the exact file suffix the lookup tried
  (`-N=..-K=..`, `-B=..-N=..-K=..`, or a fused op's custom suffix). The K in
  AFP4WFP4 file names comes out as the logical K for free, because it is the K
  the wrapper passes.
- **which M buckets to write**: the M bounds that lookup searches (the
  caller's `bounds=`, else the file's `M_BOUNDS`, else the standard bounds).

Each candidate's output is compared with the output of the config the library
resolves (relative L2 error above 5% disqualifies it), so a config that is fast
because it is wrong never wins. A config that fails to compile, trips an
assert, or produces a wrong result is logged and skipped; one that runs out of
resources rules out every other config with the same tile sizes. If the
profiled process crashes or hangs, only the config it was running is lost and
the rest run again in a new process.

## Sweeping

    python3 sweep_configs.py <op> <M> <DIM>=<int> ... [--gpu G] [--backend triton|gluon]

`<op>` is a case name from `gemm_cases.py` (the wrapper's name), and the dims
are the case's parameters after `M`. Results go to
`sweeps/<op>/<arch>-<backend>/<dims>/M=<M>.jsonl`.

Example 1: A16W16 GEMM with the default ranges on GPU 0

    python3 sweep_configs.py gemm_a16w16 64 N=8192 K=3584 --gpu 0 > example1.out

Example 2: A8W8 blockscale GEMM in the background, one M per GPU (the case
already limits `BLOCK_SIZE_K` to the 128-wide scale blocks)

    N=2112
    K=7168
    for M_G in "8 0" "16 1" "32 2" "64 3" "128 4" "256 5" "8192 6"; do
        set -- $M_G
        nohup python3 sweep_configs.py gemm_a8w8_blockscale $1 N=$N K=$K \
            --gpu $2 > example2-M=$1-N=$N-K=$K.out &
    done

Example 3: AFP4WFP4 preshuffle GEMM with the values tried for one key
replaced, printing every skipped config and why

    python3 sweep_configs.py gemm_afp4wfp4_preshuffle 64 N=7168 K=2048 \
        --space BLOCK_SIZE_K=256,512,1024 --overwrite --verbose > example3.out

`--space KEY=V1,V2,...` replaces the values tried for a key (values are read
as JSON: `null`, `true`, numbers; anything else is a string). Pinning every
key to one value times just that config against the current one.

Backends: a case that takes `backend` can be tuned on either one with
`--backend`; without it the wrapper picks, as in production (for example
gluon on gfx1250 for `gemm_a16w16`). The Gluon kernels on gfx950 are opt-in:

    python3 sweep_configs.py gemm_a8w8 16 N=1024 K=1024 --backend gluon

Batched GEMMs take the batch as a dim:

    python3 sweep_configs.py batched_gemm_bf16 32 N=1024 K=4096 B=16

## Writing the config file

    python3 write_best_configs.py <op> <DIM>=<int> ... [--install]

Collects every `M` swept for that op and shape and writes one file, named and
placed exactly where `get_gemm_config()` looks for it. Each tuned `M` goes in
the smallest bucket that covers it and the largest tuned `M` becomes `"any"`
(`--no-any` keeps it as a bucket). Tune at the bucket bounds.

    python3 write_best_configs.py gemm_a8w8_blockscale N=2112 K=7168

writes `tuned_configs/<arch>/<backend>/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=2112-K=7168.json`,
mirroring the config tree; `--install` writes it into
`aiter/ops/triton/configs/` directly. When logs from more than one arch or
backend exist for the shape, pick one with `--arch` / `--backend`.

The family's `DEFAULT.json` has to exist for the arch and backend before
anything resolves (`configs/CLAUDE.md`, section 6); the sweep stops and says so
when it is missing.

## Verifying

    python3 verify_configs.py gemm_a8w8_blockscale 32 N=2112 K=7168

prints the family and backend the op looked up, the file its config came from
(the tuned file for the shape, or `DEFAULT.json`), the config, and the
kernels' median runtime. Each run is a fresh process, so it sees files you
just installed.

## Adding a GEMM

Every public GEMM wrapper needs a case in `gemm_cases.py`; that is all the
tuning scripts need to tune it on every arch and backend it supports. A case is
named after the wrapper, takes `M` plus the wrapper's other dims, and returns a
callable that runs the op once:

```python
@gemm_case()
def gemm_a8w8(M, N, K, backend=None):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as op
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
        generate_gemm_a8w8_inputs,
    )

    _, e4m3_type = get_fp8_dtypes()
    dtype = torch.bfloat16
    x, _, w, x_scale, w_scale, _, y = generate_gemm_a8w8_inputs(
        M, N, K, in_dtype=e4m3_type, out_dtype=dtype, layout="TN", output=True
    )
    return lambda: op(x, w, x_scale, w_scale, None, dtype, y, **_backend(backend))
```

- Build inputs with the unit test's generator, and call the wrapper exactly as
  a user would: no `config=`. Return the op's result, which the output check
  compares.
- Take `backend=None` and pass `**_backend(backend)` when the wrapper has a
  `backend` argument.
- Zero accumulated outputs inside the callable for atomic kernels.
- `@gemm_case(space={"BLOCK_SIZE_K": [128]})` limits a key to the values the
  kernel accepts, when others would only fail.
- `@gemm_case(kernels=("_ff_",))` names the kernels to time when their names do
  not contain `gemm`.

And keep the family's config files honest:

- `DEFAULT.json` for each arch and backend lists exactly the keys the kernel
  reads: the sweep tunes those keys and writes nothing else.
- The wrapper must look its config up through `get_gemm_config()` when no
  `config` is passed (thin `_get_config()`, as `aiter/ops/triton/README.md`
  asks). An op that never does, or that looks up more than one config per call
  (a composite op), cannot be tuned this way; tune the ops it is built from.

MOE GEMMs use dispatch tables instead of `get_gemm_config()` and are not
covered by these scripts.
