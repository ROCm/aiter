# Attention Reproduction
## Run Tests

```bash
pytest test_attention.py
```

## Standalone run
```bash
bash run.sh
```
## Notes
Requires using: https://github.amd.com/GFX-IP-Arch/triton/tree/cagri/fma_fix to use `v_pk_fma` and `v_maximum` (nan propagating) to get rid of self max issues.
Note, maximum issue is fixed by using a vibe coded llir modifier, it might be buggy but tests pass.

Page index loading is deactivated by default (controlled by `remove_indirect_access`).
Causes perf. issues, related to https://github.com/ROCm/triton-internal/issues/1748

## MIR Basic Block Analysis
https://excalidraw.com/#json=uZp7cvP3TuMx9pJFpAGcx,hXMntmsu3TkRgPESFpnczA

