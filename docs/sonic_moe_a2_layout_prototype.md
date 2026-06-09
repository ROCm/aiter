# SonicMoE A2 Layout Prototype

This branch now has an experimental `a2_layout=current|sorted` benchmark path.
It is meant to answer one narrow question: can we validate a sorted A2 dataflow
without starting a full CK/ASM rewrite first?

## What Was Implemented

- `op_tests/op_benchmarks/sonic_moe_a2_layout.py`
  - packs current `[token, topk, inter_dim]` A2 into sorted row order
  - unpacks sorted A2 for sentinel roundtrip checks
  - runs a standalone Triton `triton_sorted_atomic_fp32` stage2 kernel
- `op_tests/test_sonic_moe_a2_layout.py`
  - validates sorted pack/unpack
  - validates sorted stage2 against a torch reference
- `op_tests/op_benchmarks/bench_sonic_moe_stage_breakdown.py`
  - adds `--a2-layout current|sorted`
  - reports `a2_pack_ms` and `sorted_stage2_backend`
- `op_tests/op_benchmarks/bench_sonic_moe_stage_sweep.py`
  - sweeps `--a2-layout current,sorted`
  - saves correctness fields in CSV/JSONL

The sorted path is intentionally a probe, not a production kernel. It keeps the
current CK stage1, explicitly packs A2 to sorted order, and then runs a generic
Triton stage2 with fp32 atomic output accumulation.

## Saved MI355X Data

Host path:

```text
/shared/amdgpu/home/hyi_qle/yifehuan_temp/data/sonic_moe_a2_layout_latest
```

Container path:

```text
/app/yifehuan_temp/data/sonic_moe_a2_layout_latest
```

Files:

- `README.md`: run summary and interpretation
- `sonic_moe_a2_layout_compare.csv`: normalized benchmark rows
- `sonic_moe_a2_layout_compare.jsonl`: full `result_json` rows
- `logs/`: per-run stdout
- `run.log`: top-level sweep log

## Result Summary

| Shape | Layout | Direct ms | Stage sum ms | Stage1 ms | Stage2 ms | A2 pack ms | Max abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `32768,4096,512,128,8` | current | 5.7405 | 5.7084 | 2.3898 | 3.2486 |  | 0 |
| `32768,4096,512,128,8` | sorted | 5.7494 | 15.9392 | 2.4008 | 12.7071 | 0.7596 | 4.54e-4 |
| `32768,4096,1024,128,8` | current | 8.5394 | 8.5051 | 4.6588 | 3.7739 |  | 0 |
| `32768,4096,1024,128,8` | sorted | 8.5850 | 18.9280 | 4.7447 | 12.8135 | 1.2968 | 6.14e-4 |
| `32768,4096,1024,256,8` | current | 8.8872 | 8.7419 | 4.6908 | 3.9770 |  | 1.91e-6 |
| `32768,4096,1024,256,8` | sorted | 8.9514 | 19.0002 | 4.8494 | 12.7948 | 1.2833 | 6.33e-4 |

Correctness passed for all rows. The sorted prototype is slower because it pays
an explicit pack and uses a generic Triton atomic-fp32 stage2. That is expected;
the point was to validate the index split and isolate the real next kernel
boundary.

## Kernel Implication

Current CK `DeviceMoeGemm` uses one packed `sorted_token_ids` value for two
purposes:

- stage2 A2 read row: `token * topk + slot`
- final output scatter row: `token`

That encoding prevents a true sorted A2 layout from being represented by Python
shape/stride changes alone. A production sorted-A2 path needs either:

- separate A-row and C-row indices in stage2, or
- equivalent in-kernel derivation that reads A2 by sorted row while scattering
  final output by original token.

The next useful implementation step is therefore not to tune the Triton
prototype. It is to add a CK/ASM stage2 variant whose A read path is sorted-row
contiguous and whose output path still atomic-adds by original token. After
that, stage1 should be changed to write sorted A2 directly, removing the pack.
