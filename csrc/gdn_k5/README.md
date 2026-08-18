# FlyDSL K5 mfma16_hip BV Tune

1. Install aiter:
```bash
cd $aiter_path
python3 setup.py develop
```

2. Add model shapes in `aiter/configs/chunk_gdn_h_mfma16_hip_untuned.csv`

3. Tune BV winners (writes candidate rows; review before merging into the repo):
```bash
python3 csrc/gdn_k5/chunk_gdn_h_mfma16_hip_tune.py \
  -i aiter/configs/chunk_gdn_h_mfma16_hip_untuned.csv \
  -o /tmp/chunk_gdn_h_mfma16_hip_tuned.candidate.csv
```

4. Validate the checked-in tuned table on the current GPU (default 10% ``us``
   drift tolerance; CI ``test_run_config`` passes 40% for small-shape variance):
```bash
python3 csrc/gdn_k5/chunk_gdn_h_mfma16_hip_tune.py \
  --run_config aiter/configs/chunk_gdn_h_mfma16_hip_tuned.csv
```

Runtime dispatch reads the tuned table through `AITER_CONFIG_GDN_K5_MFMA16_HIP`
(the same path family as other `aiter/configs/*_tuned*.csv` tables).

## Options

- `-i / --untune_file`: untuned shape list; `model` column selects K5 prefill cases
- `-o / --tune_file`: tuned BV output path
- `--run_config [file]`: replay tuned rows and compare live `us` against the csv
- `--compare --update_improved`: merge improved rows into `-o` after a tune sweep
- `--case REGEX ...`: optional pytest-case filters after the untuned model filter
- `--only-improvements`: emit rows only when measured BV beats the rule
