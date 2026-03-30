## Summary
## Split Tests FILE_TIMES Update
- repo: `ROCm/aiter`
- runs_count target: `10`
- aggregate mode: `median`
- default time: `15s`
- file changed: `yes`

### Aiter
- runs used: `10`
- discovered files: `60`
- with samples: `60`
- added: `5`
- updated: `46`
- unchanged: `9`
- defaulted (no history): `0`
- removed stale entries: `0`
- defaulted files list: `none`

### Triton
- runs used: `10`
- discovered files: `69`
- with samples: `69`
- added: `31`
- updated: `34`
- unchanged: `4`
- defaulted (no history): `0`
- removed stale entries: `0`
- defaulted files list: `none`

## Test plan
- [x] bash .github/scripts/split_tests.sh --shards 5 --test-type aiter --dry-run
- [x] bash .github/scripts/split_tests.sh --shards 8 --test-type triton --dry-run
