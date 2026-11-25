# RDNA attention

Decode and prefill attention over the paged KV cache for gfx1150..gfx1153
(RDNA3.5) and gfx1200/gfx1201 (RDNA4), ported from ROCm/vllm's
`ROCM_RDNA_ATTN` backend. The Python entry point is `aiter.ops.rdna_attn`.

| What | Where |
| --- | --- |
| Kernels (templates on one table entry) | `csrc/kernels/rdna_attn/rdna_{decode,prefill}_attn.cuh` |
| Tuned tables, one per GPU family | `csrc/kernels/rdna_attn/rdna_*_configs.h` |
| Argument checks and launches | `csrc/py_itfs_cu/rdna_attn.cu` |
| Lookup, scratch, batch plans, encoder attention | `aiter/ops/rdna_attn.py` |
| Tests | `op_tests/test_rdna_attn.py` |
| Tuning and timing tools | `op_tests/op_benchmarks/rdna_attn/` |

## What it serves

- fp16 and bf16, head dims 32, 48, 64, 72, 80, 96, 128, 256 and 512.
- The KV cache packed HND: `(pages, Hkv, page_size, 2 * D)`, K then V in a row.
- Only the shapes `(Hq, Hkv, D, window, page size)` in the tables of the GPU's
  family. A lookup returns `None` for any other shape; the caller falls back.
- Decodes of 1 to 8 query tokens per sequence (one sequence or a batch), under
  CUDA graphs: the sequence lengths and multimodal ranges are read from device
  memory at replay.
- Causal prefill of one sequence or a batch, with a sliding window and
  multimodal (PrefixLM) ranges.
- Non-causal attention for encoders, reading Q, K and V in place.

There is one table per GPU family: gfx1151's, gfx1150's (also used by gfx1152
and gfx1153) and gfx1201's (also used by gfx1200).

## Modules

Each table entry is a separate build of a kernel, so the kernels are not all
compiled at once:

- `module_rdna_attn` holds the tables and no kernel. It builds in about 10 s
  and answers every lookup.
- `module_rdna_attn_d<head dim>_<fp16|bf16>` holds the decode, batched decode
  and prefill kernels of one head dim and dtype. It is built at the first
  launch of one of them, in about half a minute on a Ryzen AI Max.

A module is built for the GPU families of `GPU_ARCHS` (by default, of the GPUs
of the machine). Each family's kernels are compiled for its archs only.

A build is looked up in `module_rdna_attn` and launched by its index in a
kernel module, which holds its own copy of the tables. Each launch carries a
fingerprint of the entry's fields, and a module whose tables hold another
entry at that index refuses it: after changing the table headers by hand,
remove the built `module_rdna_attn*` (`tune.py --apply` does). Lookups take
the `device` they are for; a process may hold GPUs of several families.

The tables were tuned on kernels compiled without two of the HIP flags aiter
adds to every module, `-mllvm --lsr-drop-solution=1` and
`-mllvm -enable-post-misched=0`. With them the 128-dim prefill kernels run 10
to 25% slower on gfx1151, so `module_rdna_attn` leaves them out
(`flags_hip_exclude` in `aiter/jit/optCompilerConfig.json`).

## Use

```python
import torch
from aiter.ops import rdna_attn

# Decode: q is (sequences * M, Hq, D).
variant = rdna_attn.variant_for(hq, hkv, d, m, page_size, 1, window, q.dtype)
if variant is not None:
    scratch = rdna_attn.make_scratch(variant, q.device)  # once, outside graphs
    rdna_attn.load(variant).decode_attn(
        q, kv_cache, block_table, out, *scratch, seq_lens, scale
    )

# Prefill of one sequence of M query tokens.
variant = rdna_attn.prefill_variant_for(
    hq, hkv, d, page_size, 1, window, q.dtype, m
)
scratch = rdna_attn.make_prefill_scratch(
    rdna_attn.prefill_variants(hq, hkv, d, page_size, window, q.dtype), q.device
)
rdna_attn.load_prefill(variant).prefill_attn(
    q, kv_cache, block_table, out, *scratch, seq_lens, max_seq_len, scale
)

# Encoder attention over segments laid one after the other.
if rdna_attn.encoder_unsupported(q, hkv) is None:
    out = rdna_attn.encoder_forward(q, k, v, segment_lengths, scale)
```

A batch of sequences takes `variant_for(..., batch=True)` for decode, and
`prefill_batch_variant_for` with `plan_prefill` for prefill; see
`_launch_prefill_batch` in `op_tests/test_rdna_attn.py`.

`block_table` and `seq_lens` are int32. `layout` is 1 (HND), the only one
built.

## Tables

The tables are tuned per GPU family with the tools of
`op_tests/op_benchmarks/rdna_attn` (see its README): `tune.py` searches the
launch knobs of a configuration and writes the headers, and `matrix.py` times
the shapes against their roofline. A table's entries must be tuned on the
kernel sources they are built from; `tune.py --apply` refuses the others.

## Test

```bash
python3 op_tests/test_rdna_attn.py                           # every build
python3 op_tests/test_rdna_attn.py -d fp16 -s 32,8,128,0 -m 1 4
pytest tests/test_rdna_attn_tables.py                        # no GPU needed
```

`op_tests/test_rdna_attn.py` times every build of the current GPU's tables,
checks it against the fp32 reference and prints one table per kind of launch,
then checks CUDA-graph replay, multimodal ranges, sliding windows and refused
inputs. It skips on other GPUs. `tests/test_rdna_attn_tables.py` checks the
rules the tables of every GPU family must keep, on the headers themselves.
