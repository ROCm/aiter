# FlyDSL K5 opt BV Tune

Model untuned/tuned CSVs live under `aiter/configs/model_configs/` (`qwen3_5_*_chunk_gdn_h_opt_{un,t}uned.csv`). Runtime merge uses `AITER_CONFIG_GDN_K5_OPT` (`configs/chunk_gdn_h_opt_tuned.csv` anchor + `model_configs/*_tuned.csv`; canonical path may not exist on disk). Prefill cases are in `op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py`.

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
