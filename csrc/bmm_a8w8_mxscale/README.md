# MXFP8 Batched GEMM Tuning

Tune preshuffled-weight MXFP8 BMM on **gfx950** with OPUS and FlyDSL.
The tuner compares eligible kernels from both backends using the same operands,
correctness checks and timing settings, and saves the best candidate per shape.
It uses the shared GEMM tuner workflow (`-i`, `-o`, `--all`, `--run_config`,
`--compare`). Run the commands below from the repository root.

## Setup

Install AITER in a ROCm environment with a gfx950 GPU:

```bash
python3 setup.py develop
```

OPUS kernels must be available in the installed AITER module. Install the
FlyDSL dependencies when using `--libtype flydsl` or `--libtype all`.
Use `--libtype opus` to tune only OPUS. The first use of an uncached FlyDSL
specialization may compile it. Changing a tuned CSV does not force an OPUS
module rebuild; adding kernel implementations does require a build.

## CSV templates

The following files ship with **only a header, no shape or result rows**:

- [Untuned input](../../aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_untuned.csv): add the shapes you want to tune.
- [Tuned output](../../aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv): populated by the tuner.

The untuned header is:

```csv
b,m,n,k,w_scale_block
```

For example, to tune both scale formats for one shape, add these two rows to
an input CSV. These examples are documentation only; they are not prefilled
in the template.

```csv
b,m,n,k,w_scale_block
4,16,2048,8192,128x128
4,16,2048,8192,32x32
```

| Column | Meaning |
|---|---|
| `b` | Batch count. Activations use token-major `[m,b,k]` storage. |
| `m` | Rows per batch. Output has shape `[m,b,n]`. |
| `n` | Output columns. The preshuffled weight has logical shape `[b,n,k]`. |
| `k` | Reduction dimension. |
| `w_scale_block` | `128x128`, `32x32`, or `1x32`; selects weight scales and the corresponding `1x128` or `1x32` activation scales. |

For `32x32`, activation scales have shape `[m,b,k/32]` and weight scales
have shape `[b,n/32,k/32]`. Per-row weight quantization (`1x32`) uses
weight scales `[b,n,k/32]` and activation scales `[m,b,k/32]`. It is supported
by OPUS kids 9700--9710 and 9713; the joint tuner skips FlyDSL candidates for this block.
Compact OPUS kernels support `128x128` weight scales only; the existing
OPUS families provide the `32x32` candidates.

All dimensions must be positive integers. For `1x32`, N must be a multiple of
16 and K a multiple of 128; other blocks require both multiples of 128.
Individual kernels impose additional tile, layout and split-K constraints;
the tuner filters candidates for each shape. Weights use the `(16,16)`
preshuffle layout, scales use ordinary E8M0 storage, and tuning uses BF16 output.

The full result key is `gfx,b,m,n,k,w_scale_block`. An optional input `gfx`
column restricts rows to that architecture; otherwise the current GPU is used.
Uppercase shape column names and `g` as an alias for `b` are also accepted.

Keep `w_scale_block` in the input to specify the exact shape/scale combinations.
`--w_scale_block 128x128` filters explicit scale rows; it does not create
missing ones. For shape-only input, use `--w_scale_block 32x32,128x128` to
generate both square blocks (also the default). Use `--w_scale_block 1x32`
for per-row weight scales, or include it in a comma-separated list.
Adding candidates does not add shapes.

To prepare `1x32` tensors, reuse the existing quantizer with ordinary E8M0
scales (no scale shuffle), then preshuffle the quantized weight once:

```python
import torch
import aiter
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_weight

# x: [M,B,K], w: [B,N,K], contiguous BF16/FP32 inputs on gfx950.
xq, xs = per_1x32_f8_scale_f8_quant(x, scale_type=torch.uint8)
wq, ws = per_1x32_f8_scale_f8_quant(w, scale_type=torch.uint8)
xs = xs.view(*x.shape[:-1], x.shape[-1] // 32)
ws = ws.view(*w.shape[:-1], w.shape[-1] // 32)
wq = shuffle_weight(wq, layout=(16, 16))
y = aiter.batched_gemm_a8w8_mxscale_bpreshuffle(xq, wq, xs, ws)
```

The operator infers the `1x32` block from the scale tensor shapes. The raw
`opus_bmm` entry also accepts these scales with an explicit kid 9700--9710 and 9713.
Row-major B entry points do not support this block.

## Tune

After filling the untuned CSV, select an idle GPU and run:

```bash
HIP_VISIBLE_DEVICES=0 python3 csrc/bmm_a8w8_mxscale/bmm_a8w8_mxscale_bpreshuffle_tune.py \
    -i aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_untuned.csv \
    -o aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv \
    --libtype all --mp 1
```

An empty input template has no work to tune. By default, keys already in the
output CSV are skipped. Add `--all` to retune every selected input key; rows
outside that input are retained. For multiple idle GPUs, for example, use
`HIP_VISIBLE_DEVICES=0,1` with `--mp 2`.

Use explicit `-i` and `-o` as above for this template workflow. Without `-i`,
the tuner takes shapes from the shipped DSV4 model table; without `-o`, it
also writes to that model table. `--apply` explicitly selects the model table
as output and enables retuning. The empty templates do not change these defaults.

The tuned header is:

```csv
gfx,b,m,n,k,w_scale_block,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio
```

`libtype` is `opus` or `flydsl`. OPUS uses a registered `kernelId`; FlyDSL uses
`kernelId=-1` and stores its configuration in `kernelName`. `splitK` records
the selected split count. `us` is candidate kernel time in microseconds;
`tflops` and `bw` are derived throughput metrics. `errRatio` records the
fraction of output values outside the numerical tolerance.

## Verify and compare

Verify the saved backend/kernel selections without retuning:

```bash
HIP_VISIBLE_DEVICES=0 python3 csrc/bmm_a8w8_mxscale/bmm_a8w8_mxscale_bpreshuffle_tune.py \
    --run_config aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv
```

To benchmark current production routing for the input shapes, omit the config
argument after `--run_config`:

```bash
HIP_VISIBLE_DEVICES=0 python3 csrc/bmm_a8w8_mxscale/bmm_a8w8_mxscale_bpreshuffle_tune.py \
    -i aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_untuned.csv \
    --run_config
```

To compare production calls before and after tuning and update improved rows:

```bash
HIP_VISIBLE_DEVICES=0 python3 csrc/bmm_a8w8_mxscale/bmm_a8w8_mxscale_bpreshuffle_tune.py \
    -i aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_untuned.csv \
    -o aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv \
    --compare --update_improved --libtype all --mp 1
```

The default improvement threshold is 3%, configurable with
`--min_improvement_pct`. A passing candidate can also be saved when no valid
baseline exists. With `--compare` alone, the final output is not updated;
the command prints the candidate CSV and any report paths. Comparison uses the
public production operator; saved-config verification runs the named kernels.

## Use the tuned result

The preshuffled BMM runtime normally merges the generic tuned CSV with matching
files under `aiter/configs/model_configs/`. Shape keys must be unique across
those sources. For local experiments or retuning keys already in a model table,
use a separate output such as `-o /tmp/bmm_tuned.csv` and select it explicitly
before starting the application:

```bash
export AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE=/tmp/bmm_tuned.csv
```

This selects that CSV for `aiter.batched_gemm_a8w8_mxscale_bpreshuffle` calls.
The same environment override can select the pre-tune baseline for `--compare`.
When a config lookup misses, the existing FlyDSL fallback remains in effect.

## Other useful options

| Option | Purpose |
|---|---|
| `--libtype opus`, `flydsl`, or `all` | Select candidate backends; default is `all`. |
| `--w_scale_block` | Select `1x32`, `32x32`, or `128x128`; comma-separated lists are supported. |
| `--opus_candidates all` | Include all registered OPUS candidates compatible with the plain-scale preshuffled-weight pool; default is `policy`. |
| `--flydsl_candidates all` | Consider configs from all M values in the shipped table; default is nearby M values plus the heuristic choice. |
| `-o2 /tmp/bmm_profile.csv` | Save measurements for all candidates. |
| `--warmup`, `--iters` | Control warmup and timing iterations. |
| `--graph_m_max 64` | Use graph timing for M up to 64; disabled by default. |
| `--help` | Show all shared and BMM-specific options. |

See [OPUS compact BMM](../opus_gemm/README.md#compact-mxfp8-bmm-on-gfx950)
for compact tile configurations and kernel constraints.

The 1x32 fallback uses the 64x64x128 tile (kid 9713) when M is at least 64
and the batched output grid contains at least 256 such tiles. Smaller grids
retain the decode tiles. Tuned rows can also select split-K for small grids;
split-K is not enabled uniformly because its reduction cost can outweigh the
extra parallelism. Both the fallback and tuned lookup validate tile alignment.
