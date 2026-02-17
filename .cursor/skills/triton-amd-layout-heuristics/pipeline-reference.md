# AMD Pass Pipeline Reference

## Full Pipeline Order

From [`third_party/amd/backend/compiler.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/backend/compiler.py) (`make_ttgir` and `make_llir`):

```
═══ TTIR → TTGIR ═══

1. convert-triton-to-tritongpu(target, num_warps, threads_per_warp, num_ctas)
   → Assigns default BlockedEncoding to all tensors

═══ Optimize TTGIR ═══

2. tritongpu-coalesce
   → AxisInfo-driven coalesced BlockedEncoding for memory ops

3. tritongpu-f32-dot-tc(emuTF32=False)
   → Handle TF32 dot ops

4. tritongpu-remove-layout-conversions  [1st run]
   → Forward propagation from anchors + backward remat

5. tritongpu-optimize-thread-locality
   → Optimize thread-local data access patterns

6. tritonamdgpu-accelerate-matmul(arch, matrix_instr_nonkdim, kpack)
   → Blocked → MFMA/WMMA encoding for dot ops

7. tritongpu-remove-layout-conversions  [2nd run]
   → Cleanup after matmul acceleration

8. tritonamdgpu-optimize-epilogue
   → Optimize epilogue code after matmul

9. tritonamdgpu-optimize-dot-operands(arch)
   → LDS allocation for scaled upcast ops (CDNA4)

10. tritonamdgpu-hoist-layout-conversions
    → Hoist DotOperand converts out of loops

11. tritonamdgpu-sink-layout-conversions
    → Sink converts past LDS deallocs

12. tritongpu-fuse-nested-loops

═══ Pipelining ═══

13. tritonamdgpu-pipeline(use_async_copy, use_block_pingpong)
    → Software pipelining

14. tritonamdgpu-coalesce-async-copy(arch)  [if use_async_copy]
    → Coalesce direct-to-LDS writes (CDNA3/4 only)

15. tritonamdgpu-convert-to-tensor-ops
16. canonicalizer

17. (optional) insert-instruction-sched-hints(hint)  [per hint in schedule_hint]

18. tritongpu-remove-layout-conversions  [3rd run]
    → Final cleanup

19. tritongpu-reduce-data-duplication

20. (optional, arch-dependent) tritonamdgpu-in-thread-transpose
21. (optional) tritongpu-remove-layout-conversions  [4th run]

22. tritonamdgpu-move-up-prologue-loads
23. (optional) tritonamdgpu-block-pingpong(num_stages)

═══ TTGIR → LLIR ═══

24. convert-triton-amdgpu-to-llvm(arch)
25. (optimization passes, linking, etc.)
```

## Key Observations

### Multiple `remove-layout-conversions` runs
The pass runs 3-4 times because each transformation introduces new layout mismatches:
1. After initial coalescing (cleanup default→coalesced transitions)
2. After matmul acceleration (cleanup blocked→MMA transitions)
3. After pipelining (cleanup pipeliner-introduced transitions)
4. After in-thread transpose (if enabled)

### Pass ordering matters
- Coalescing runs **before** matmul acceleration: memory ops get coalesced layouts first, then dot ops get MMA layouts, then propagation resolves the tension
- Hoist/sink run **after** matmul acceleration but **before** pipelining: ensures the pipeliner sees the optimized layout placement
- Async copy coalescing runs **after** pipelining: only operates on `AsyncCopyGlobalToLocalOp` introduced by the pipeliner

### Configuration options

| Option | Source | Effect |
|--------|--------|--------|
| `arch` / `archGenerationName` | Backend target | Selects ISA family and available instructions |
| `matrix_instr_nonkdim` | User/autotuner | Forces MFMA tile size (0 = auto) |
| `kpack` | User/autotuner | kPack multiplier for ds_read width (1 or 2) |
| `num_stages` | User/autotuner | Pipeline stages (affects async copy + LDS usage) |
| `schedule_hint` | User | Instruction scheduling strategy |

## AMD vs NVIDIA Pipeline Comparison

| Phase | AMD | NVIDIA |
|-------|-----|--------|
| Coalescing | Same `tritongpu-coalesce` | Same |
| Layout removal | Same pass, 3-4 runs | Same pass, 3 runs |
| Matmul accel | `tritonamdgpu-accelerate-matmul` | `tritongpu-accelerate-matmul` |
| Dot operands | `tritonamdgpu-optimize-dot-operands` (LDS for scales) | `tritongpu-optimize-dot-operands` (swizzle, fuse trans) |
| Layout hoisting | `tritonamdgpu-hoist-layout-conversions` | Part of `remove-layout-conversions` |
| Layout sinking | `tritonamdgpu-sink-layout-conversions` | N/A |
| Pipelining | `tritonamdgpu-pipeline` | WGMMA/TMA-based pipeliner |
| Async coalescing | `tritonamdgpu-coalesce-async-copy` (direct-to-LDS) | `tritongpu-coalesce-async-copy` (TMA) |
| TMEM optimization | N/A | `optimize-tmem-layouts` (Blackwell) |
