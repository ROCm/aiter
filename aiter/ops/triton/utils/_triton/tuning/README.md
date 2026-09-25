# GEMM tuning

`tune_gemm.py` tunes one GEMM at one shape and keeps the fastest config. It
works for every GEMM that has a case in `gemm_cases.py`, on any arch, for
Triton and Gluon kernels alike.

    cd aiter/ops/triton/utils/_triton/tuning
    HIP_VISIBLE_DEVICES=0 python3 tune_gemm.py gemm_a8w8 M=16 N=1024 K=1024

`gemm_a8w8` is the name of the wrapper being tuned; the dims are the case's
arguments (batched GEMMs also take `B=`). What the script does:

1. Runs the op with the config it gets today (the tuned file for this shape,
   or `DEFAULT.json`) and measures its kernel time.
2. Runs every candidate config and measures it. The op is called exactly as in
   production; the candidate is slipped in where the op reads its config
   (`get_gemm_config()`). A candidate that raises, or whose output differs from
   step 1's, is written to `errors-<op>-<shape>.txt` and skipped.
3. Measures the best candidate and the current config once more, back to back.
   If the candidate is faster (by more than 3%), it goes into the config file
   for this shape. A shape that has no tuned file gets one.

Measuring: the op runs in a separate process under `rocprofv3 --kernel-trace`
(`profile_configs.py`), 100 configs per process, and each config's time is the
median over 250 runs of its GEMM kernels only (kernel names containing `gemm`;
a split-K reduce kernel counts too). Nothing else that the wrapper does is
timed. A process that crashes or hangs (`--timeout`, default 900 s per batch)
costs only the config it was on; the rest of the batch runs again in a new
process.

The keys tried are the keys of the family's `DEFAULT.json` for this GPU's arch
and the backend the op runs. Standard Triton keys get the usual ranges; any
other key (Gluon tiles, buffer counts, ...) tries the values the family's
files already use. `--space KEY=V1,V2` overrides the values for one key.

Review the result with `git diff aiter/ops/triton/configs`.

## Options

| Option | Meaning |
| --- | --- |
| `--backend triton\|gluon` | for ops that have both kernels; default is whatever the wrapper picks on this arch |
| `--space KEY=V1,V2 ...` | values to try for a key, e.g. `--space BLOCK_SIZE_K=256,512,1024` |
| `--timeout SECONDS` | per batch of 100 configs (default 900) |
| `--no-check` | do not compare candidate outputs with the current config's |

Several shapes at once, one GPU each:

    i=0
    for M in 8 16 32 64 128 256 8192; do
        HIP_VISIBLE_DEVICES=$((i++)) python3 tune_gemm.py gemm_a8w8_blockscale M=$M N=2112 K=7168 > M=$M.out &
    done

## Where the config goes

The file is the one `get_gemm_config()` reads for the shape, under
`aiter/ops/triton/configs/<arch>/<backend>/gemm/<family>/`, for example
`gfx950/triton/gemm/gemm_a8w8/GEMM-A8W8-N=1024-K=1024.json`. Inside it, the
winner goes into the `M_LEQ_<bound>` bucket for the smallest standard bound at
or above `M` (`M=20` → `M_LEQ_32`), so tuning one `M` never changes another
`M`'s config. Tune at the bounds themselves (16, 32, 64, ...) for tidy files.
A new file starts as a copy of `DEFAULT.json` with the winner added, so every
other `M` keeps the config it had.

## Adding a GEMM

Add a case to `gemm_cases.py`, named after the wrapper. It takes the shape
dims, builds the inputs with the unit test's generator, and returns a function
that runs the op once, without `config=`:

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
    return lambda: op(x, w, x_scale, w_scale, None, dtype, y, **backend_kwarg(backend))
```

- Take `backend=None` and pass `**backend_kwarg(backend)` when the wrapper has
  a `backend` argument, so `--backend` can pick one.
- `@gemm_case(space={"BLOCK_SIZE_K": [128]})` pins a key the kernel
  constrains, so the sweep does not try values that can only fail.
- Zero an accumulated output inside the returned function (atomic kernels).

The kernel side has one rule: each `DEFAULT.json` of the family lists exactly
the config keys the kernel reads. Those are the keys the script tunes and
writes. The wrapper must read its config through `get_gemm_config()` when no
`config` is passed; an op that reads two configs per call cannot be tuned
this way (tune the ops it is built from). Kernel names must contain `gemm`
to be timed.

MOE GEMMs use dispatch tables instead of `get_gemm_config()` and are not
covered.
