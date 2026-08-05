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

## Timing

`bench` uses HIP events around a loop of launches, so the number is
**dispatch-level** and includes launch overhead. The module stays loaded across
iterations, so load cost is excluded. For single-wave cycle counts, put
`s_get_shader_cycles_u64` in the ISA itself and write the delta to a buffer.
