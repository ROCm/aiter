# FlyDSL K5 mfma16_hip BV Tune

1. Install aiter:
```bash
cd $aiter_path
python3 setup.py develop
```

2. Model shapes live under `aiter/configs/model_configs/` (Hv=32 for 35B, Hv=64 for 397B):
   - `qwen3_5_*_chunk_gdn_h_mfma16_hip_untuned.csv` — compile/AOT + tune case filter (H/Hg = Hv/TP, Hk/TP)
   - `qwen3_5_*_chunk_gdn_h_mfma16_hip_tuned.csv` — measured BV rows (`snapshot_bf16` True=bf16 / False=fp32 snapshot)

3. Prefill case grid (P0+P1) is defined in `op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py`:
   - **P0 dense**: TP1/2/4/8 × bf16/fp32 snapshot × seq 2500/60k/128k
   - **P1 varlen ali**: TP1/2/4/8 × bf16/fp32 snapshot × mnbt 8192..65536

4. Tune BV winners (review candidate before merging into repo):

Implementation follows the Fmoe-style ``TunerCommon`` wrapper in ``k5_bv_tuner.py``;
``chunk_gdn_h_mfma16_hip_tune.py`` is the CLI entry point.

```bash
# Full sweep per model (GPU-heavy; hours on MI300X)
python3 csrc/gdn_k5/chunk_gdn_h_mfma16_hip_tune.py \
  -i aiter/configs/model_configs/qwen3_5_35b_chunk_gdn_h_mfma16_hip_untuned.csv \
  -o /tmp/qwen3_5_35b_chunk_gdn_h_mfma16_hip_tuned.candidate.csv

# Batched example: dense TP4 bf16 snapshot only
python3 csrc/gdn_k5/chunk_gdn_h_mfma16_hip_tune.py \
  -i aiter/configs/model_configs/qwen3_5_35b_chunk_gdn_h_mfma16_hip_untuned.csv \
  -o /tmp/35b_dense_tp4_bf16.candidate.csv \
  --case 'Qwen3.5-35B-dense-tp4-bf16snap'

# Batched example: 35B varlen ali TP2 fp32 snapshot
python3 csrc/gdn_k5/chunk_gdn_h_mfma16_hip_tune.py \
  -i aiter/configs/model_configs/qwen3_5_35b_chunk_gdn_h_mfma16_hip_untuned.csv \
  -o /tmp/35b_varlen_tp2_fp32.candidate.csv \
  --case 'varlen-qwen-ali-tp2-fp32snapshot'
```

5. Validate tuned table on GPU (CI `test_run_config` uses 40% ``us`` tolerance):
```bash
python3 csrc/gdn_k5/chunk_gdn_h_mfma16_hip_tune.py \
  --run_config aiter/configs/model_configs/qwen3_5_397b_chunk_gdn_h_mfma16_hip_tuned.csv
```

Runtime dispatch merges tuned CSVs via `AITER_CONFIG_GDN_K5_MFMA16_HIP`
(`configs/chunk_gdn_h_mfma16_hip_tuned.csv` stub + `model_configs/*_tuned.csv`).

**Note:** Checked-in tuned rows may lag the P0+P1 grid; missing lookup keys fall back to the CU/LDS BV rule until tune output is merged.

## Options

- `-i / --untune_file`: untuned shape list (matches cases by H/Hg/is_varlen/store_fs; legacy rows may use `model`)
- `-o / --tune_file`: tuned BV table output (use `/tmp/...` for candidates)
- `--run_config [file]`: replay tuned rows and compare live `us` against csv
- `--compare --update_improved`: merge improved rows into `-o` after a tune sweep
- `--case REGEX ...`: filter pytest case ids after untuned shape filter
- `--only-improvements`: emit rows only when measured BV beats the rule
