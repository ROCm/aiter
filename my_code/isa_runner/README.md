# gfx1250 ISA runner

Assemble a hand-written or hand-edited gfx1250 `.s` into a loadable code object
and launch it through the HIP module API. LLVM is used **only as an assembler**
— no IR optimisation, no register allocation, no machine scheduling — so the
instruction sequence that runs is the one in the file.

```
.s ──clang++ -x assembler──> .o ──lld -shared──> .co ──hipModuleLoadData──> launch
                                          └── llvm-objdump ──> order check
```

Run inside the gfx1250 container (`hyg_fyd1`). No new Python dependencies:
the runner calls `libamdhip64.so` through `ctypes`, because `hip-python` is not
installed in that image.

## Quick start

```bash
cd /data/yanguahe/code/wk_sp1/aiter/my_code/isa_runner

python isa_runner.py inspect smoke_gfx1250.s               # assemble + verify order
python isa_runner.py run     smoke_gfx1250.s --smoke       # launch, check sentinel
python isa_runner.py bench   smoke_gfx1250.s --smoke --iters 200 --json

# reassemble the real gemm1 kernel and confirm nothing was reordered
python isa_runner.py inspect \
  ../isa_cmp/w1/gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1/21_final_isa.s \
  --verify-order --json
```

Exit codes: `0` ok, `1` build/load error, `2` instruction order changed,
`3` smoke result mismatch.

`bash selftest.sh` runs all of the above end to end. Verified on c9-3 /
`hyg_fyd1` / gfx1250: smoke order 13/13, 256/256 elements correct,
~8.4 us per launch, and the 2871-instruction production kernel reassembles in
exact order, loads and resolves its symbol.

## Assembly format

A file must be self-contained — the three blocks below are what make a code
object loadable. `21_final_isa.s` dumps from FlyDSL already have all of them,
so they can be fed in unmodified.

1. **Target + text**

   ```asm
   .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
   .amdhsa_code_object_version 6
   .text
   .globl  my_kernel
   .p2align 8
   .type   my_kernel,@function
   my_kernel:
       ...
       s_endpgm
   ```

2. **Kernel descriptor** (`.rodata`) — `.amdhsa_kernel my_kernel … .end_amdhsa_kernel`.
   The fields that bite:
   - `.amdhsa_kernarg_size` must match the packed argument buffer exactly.
   - `.amdhsa_user_sgpr_count 2` + `.amdhsa_user_sgpr_kernarg_segment_ptr 1`
     puts the kernarg pointer in `s[0:1]`.
   - `.amdhsa_system_sgpr_workgroup_id_x 1` then puts `workgroup_id_x` in `s2`.
   - `.amdhsa_next_free_vgpr` / `_sgpr` must cover every register used, or the
     dispatch is rejected or corrupts state.
   - `.amdhsa_wavefront_size32 1` for wave32.

3. **Metadata** (`.amdgpu_metadata` … `.end_amdgpu_metadata`) — YAML describing
   args, `.symbol: <name>.kd`, `.max_flat_workgroup_size` and `.wavefront_size`.
   The loader reads this, not the descriptor, when resolving the kernel symbol.

## Gotchas found the hard way

- **`workgroup_id_x` lives in `ttmp9`, not `s2`.** With
  `.amdhsa_system_sgpr_workgroup_id_x 1` you might expect `s2`; on gfx1250 it
  arrives through the trap temporaries. The compiler's own output reads `ttmp9`
  and uses `s2` as scratch. Reading `s2` gave a garbage block offset and a
  page fault (`Memory access fault ... Reason: Page not present`).
- **Guard your stores.** A bad index faults the GPU. The smoke kernel takes an
  `n_elems` argument and masks `exec` before storing; do the same in new
  kernels rather than trusting the launch geometry.
- **`vcc` must be declared.** Using `vcc_lo` requires
  `.amdhsa_reserve_vcc 1`.
- **Trailing `s_code_end` is expected.** `.p2alignl`/`.fill 96, 4, 0xBF9F0000`
  in the source becomes 99 padding instructions; `verify_order` reports them as
  `trailing_s_code_end_ignored` rather than a mismatch.

## Kernel symbol

`hipModuleGetFunction` takes the **kernel name** (`isa_smoke`), while the ELF
symbol is `isa_smoke.kd`. `list_kernels()` strips the suffix, so
`--kernel` always takes the plain name.

## Typed kernargs

Arguments are packed into one buffer with natural alignment and handed over
via HIP's `HIP_LAUNCH_PARAM_BUFFER_POINTER` protocol. Packing must match the
kernel's kernarg layout byte for byte.

| passed | packed as | align |
|---|---|---|
| `torch.Tensor` | `data_ptr()`, 8 B | 8 |
| `ctypes.c_*` | its own width | `sizeof` |
| Python `int` | **int32** | 4 |
| Python `float` | float32 | 4 |

A bare Python `int` is ambiguous, so anything that is not a 32-bit value must
be passed as an explicit ctypes object — `ctypes.c_uint64(ptr)` for a raw device
pointer, `ctypes.c_float(x)` for an fp32 scalar.

## Grid / block / dynamic LDS

`KernelLaunchSpec(grid, block, shared_mem_bytes, stream, device)` maps onto
`hipModuleLaunchKernel`. `shared_mem_bytes` is **dynamic** LDS and adds to the
descriptor's `.amdhsa_group_segment_fixed_size`.

The gemm1 TDM kernel's profile is recorded in `isa_runner.py` as
`TDM_GEMM1_BLOCK = (128, 1, 1)` and `TDM_GEMM1_LDS_BYTES = 159744`, taken from
`19_gpu_module_to_binary.mlir` (`threads in (%4,1,1)` with `%4 = 128`,
`dynamic_shared_memory_size %3` with `%3 = 159744`). That kernel takes 15
arguments and is **not** launched at this stage — argument initialisation is a
later adapter. Reassembling and loading it is covered, though, which is what
proves the toolchain path works on a real kernel.

## Verifying nothing moved

`--verify-order` disassembles the built object and compares the mnemonic
sequence with the source. Operand syntax and `_e32`/`_e64` suffixes are
normalised away; an insertion, deletion or reorder is reported with the index
and surrounding context. This is the guard that the assembler did not
reschedule anything.

## Iterating on a kernel

1. Copy the `21_final_isa.s` you want to modify.
2. Edit instructions. Keep `.amdhsa_next_free_vgpr`/`_sgpr` ≥ what you use.
3. `python isa_runner.py inspect edited.s --verify-order` — catches typos and
   any reordering.
4. `python isa_runner.py run edited.s --kernel <name>` once inputs exist.

Builds are cached in `~/.isa_runner_cache` keyed by source hash + arch; use
`--force` to rebuild. Binary `.text` patching is only a fallback for same-size
replacement and is not implemented here — editing the `.s` is the main path.

## Running a real kernel: `tdm_adapter.py`

The smoke kernel proves the mechanism; the adapter feeds a hand-edited ISA the
**real** gemm1/gemm2 inputs. Those inputs (preshuffled MXFP8 A, preshuffled
MXFP4 W, n32k4-folded scales, per-expert psum `m_tile_map`) are not practical to
construct by hand — a subtly wrong construction yields a plausible wrong answer.
So the adapter **captures** instead: it patches the FlyDSL launch, runs the
production path once, and records the device pointers the kernel really got.

```bash
cd /data/yanguahe/code/wk_sp1/aiter        # repo root: shadows /app/aiter
K=my_code/isa_cmp/w1/<kernel>/21_final_isa.s

# sanity: identical ISA on both sides must be bit-exact
python my_code/isa_runner/tdm_adapter.py replay --which gemm1 \
    --isa $K --reference-isa $K --out r.json

# edit, then compare against the unmodified dump
python my_code/isa_runner/make_variant.py $K edited.s --wait-after-wmma
python my_code/isa_runner/tdm_adapter.py replay --which gemm1 \
    --isa edited.s --reference-isa $K --iters 100 --out r.json
python my_code/isa_runner/show_report.py r.json
```

Capture and replay must be in the **same process** — the pointers are device
addresses, so a saved JSON is metadata only.

### Why the reference is another ISA run

`--reference-isa` is launched through the identical path first, and its output
is the reference. Comparing against the tensor left behind by the torch run does
not work: `i32_m` is the `align_m`-padded row count (49152 for 24576 real
routes), the kernel skips padding tiles, and those rows hold uninitialised
values near `FLT_MAX`. Both norms then overflow to `inf` and `rel_l2` is `NaN`.
With an ISA-vs-ISA reference the skipped rows hold the same poison value on both
sides and cancel exactly.

Validated on c9-3: same ISA both sides → `rel_l2 = 0.0`, `passed: true`, with
`reference_wrote_elems` exactly half the buffer. Negative control (drop one of
96 WMMA) → `rel_l2 = 7.4e-4`, `max_abs_diff = 98.0`, `passed: false`.

### Gotchas the adapter handles

- **Pin the inputs.** `ptr_arg` keeps only a raw pointer, so once the
  production call returns, torch's caching allocator reuses those buffers and
  the replay silently reads freed memory (NaN, not an error). The adapter holds
  references to every captured tensor.
- **Dynamic LDS is computed, not guessed.** `arena_bytes()` mirrors the
  frontend's arena math; it reproduces both the kernel's own `ARENA=158208` log
  and the `dynamic_shared_memory_size 159744` in the MLIR (the same arena after
  the `tile_m<=64` zero-fill rounding). The descriptor says
  `group_segment_fixed_size 0`, so the launch must supply this.
- **Capture the tuned tiles.** The adapter defaults `AITER_TDM_TILE_*` to the
  `g2_m64_nb3` config so the captured kernel matches the dumped
  `t64x256x256_b3` ISA; the CSV default would capture `16x256x256_b2`, which no
  dump matches.
- **Run from the repo root.** The container also has an older `/app/aiter`
  without the TDM kernel.

## Timing

`bench` uses HIP events around a loop of launches, so the number is
**dispatch-level** and includes launch overhead. The module stays loaded across
iterations, so load cost is excluded. For single-wave cycle counts, put
`s_get_shader_cycles_u64` in the ISA itself and write the delta to a buffer.
