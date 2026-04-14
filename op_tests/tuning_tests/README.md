# Tuning Tests

Minimal test suite for validating the aiter tuning infrastructure, run daily.

## Structure

| File | Level | GPU | What it tests |
|------|-------|-----|---------------|
| `test_csv_validation.py` | 0 | No | Tuned CSV integrity: duplicates, invalid times, errRatio, git conflicts |
| `test_tuner_infra.py` | 1 | No | `base_tuner` utilities: CSV I/O, merge, dedup, calculate, post_process |
| `test_mp_tuner_logic.py` | 1 | No | `mp_tuner` polling: timeout, AcceleratorError, KeyError, pool restart |
| `test_tune_pipeline.py` | 2 | Yes | End-to-end: run each tuner on small shapes, verify output CSV |

## Running

```bash
# All tests (Level 0+1, no GPU, <10s)
python3 -m pytest op_tests/tuning_tests/ -v -k "not Pipeline"

# GPU pipeline tests (~10min)
python3 -m pytest op_tests/tuning_tests/test_tune_pipeline.py -v

# Everything
python3 -m pytest op_tests/tuning_tests/ -v

# With unittest (no pytest needed)
python3 -m unittest discover -s op_tests/tuning_tests -v
```
