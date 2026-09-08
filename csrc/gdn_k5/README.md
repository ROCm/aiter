# FlyDSL K5 opt BV Tune

Model untuned/tuned CSVs live under `aiter/configs/model_configs/` (`qwen3_5_*_chunk_gdn_h_opt_{un,t}uned.csv`). Untuned tables are shape-only; `cu_num` lives in tuned rows. Newly tuned rows belong in the per-model tuned table, not in the canonical one. Prefill cases are in `op_tests/test_flydsl_linear_attention_prefill.py`.

Two header-only files under `aiter/configs/` back the merge that `AITER_CONFIG_GDN_K5_OPT` performs (and that the opt AOT reads):

- `chunk_gdn_h_opt_tuned.csv` -- merge anchor. Must be a readable csv: with no per-model table present, `get_config_file` returns this path as-is.
- `chunk_gdn_h_opt_untuned.csv` -- supplies the duplicate-detection keys, so two tables claiming the same lookup key fail the merge instead of silently coexisting. Its columns must stay equal to the tuner's `LOOKUP_KEYS`; using shape-only columns would mark every per-batch row a duplicate and collapse the table.

```bash
# Tune (write candidates under /tmp before merging)
python3 csrc/gdn_k5/chunk_gdn_h_opt_tune.py \
  -i aiter/configs/model_configs/qwen3_5_35b_chunk_gdn_h_opt_untuned.csv \
  -o /tmp/qwen3_5_35b_chunk_gdn_h_opt_tuned.candidate.csv \
  --case 'Qwen3.5-35B-dense-tp4-bf16snap'

# Replay tuned rows (default 5% us drift tolerance; CI test_run_config)
python3 csrc/gdn_k5/chunk_gdn_h_opt_tune.py \
  --run_config aiter/configs/model_configs/qwen3_5_397b_chunk_gdn_h_opt_tuned.csv
```

Options: `-i`/`-o`, `--run_config`, `--compare --update_improved`, `--case REGEX ...`, `--list-cases`.

Missing tuned lookup keys fall back to the CU/LDS BV rule until new rows are merged.
