# gfx950 packed-FP32 / MFMA cross-wave defect — AMD report package

**Hardware:** AMD Instinct MI350X (`gfx950`, `0x75a0`) · **Software:** ROCm 7.0 · **Symptom:** silent wrong results, nondeterministic.

## Contents

| Artifact | Link |
|---|---|
| Full write-up (`REPORT.md`) | [link](https://gist.github.com/njriasan/8c8a96f87854062b86a0d7919dd590e4) |
| Standalone reproducer (`pk_mfma_crosswave.cpp`) | [link](https://gist.github.com/njriasan/e760cdc88882ab3a8d8762f349f805ea) |

## The issue in one paragraph

On `gfx950`, a packed-FP32 ALU op — `V_PK_MUL_F32`, `V_PK_ADD_F32`, or `V_PK_FMA_F32` — whose **`src1` is a VGPR read with its high dword selected for the low result lane** (`op_sel[0]` set on `src1`, clear on `src0`) reads that operand as **0** in hardware **lanes 48–63**, **when another wave resident on the same SIMD is concurrently executing a double-rate XDL MFMA**. The op then computes `src0 * 0` (or `src0 + 0`). The MFMA output and the `src1` register are both correct; only the `src1` high-dword *read* inside the packed op is lost.

## Trigger (all required)

1. Packed op with **VGPR `src1`**, `op_sel[0]` set on `src1`, clear on `src0`. (SGPR/constant `src1` is safe; commuted form — high-select on `src0` — is safe.)
2. A **co-resident wave on the same SIMD running a double-rate XDL MFMA** (`32x32x16`/`16x16x32` f16/bf16, `32x32x64`/`16x16x128` f8f6f4, `32x32x32` i8, and AGPR-accumulator forms). Legacy MFMAs, DGEMM, SGEMM, SMFMAC, DOT, transcendentals do **not** trigger it.
3. **≥2 waves co-resident on one SIMD** (a single wave never reproduces it).

## Localization (by A/B experiment)

- It is the **`src1` read**, not the MFMA output (fails with inputs disjoint from the MFMA output; with the real `qk*scale` layout the `src0`=MFMA-output read is correct, only `src1`-high is lost).
- Not a register clobber (reads back correct), not operand readiness (force-settled `src1` fails identically; a constant fails hardest), not a register-location/bank conflict (any `src1` number fails, incl. the neighbor's MFMA register numbers).
- It is specifically the **`src1` (operand-B) high-dword read port** — identical math with the high-select on `src0` is always correct. Cross-wave: one wave/SIMD is clean, two waves/SIMD maximizes it; MFMAs on other SIMDs of the CU do not trigger it.

## Verified reproduction (MI350X, ROCm 7.0, GPU 7)

Every launch, all three ops; commuted and single-wave controls clean:

| mode | per-launch result (/524288 victim lanes) |
|---|---|
| `--op mul` | ~42k–45k wrong |
| `--op add` | ~11k–12k wrong |
| `--op fma` | ~7.8k–8.4k wrong |
| `--op {mul,add,fma} --safe` (commuted) | 0 |
| `--op mul --single` (one wave) | 0 |

## Impact

Production flash-attention (Triton/LLVM): `exp2(qk*scale - m_new)` lowers to `V_PK_ADD_F32 op_sel:[0,1]` with the per-row max broadcast in a VGPR, run at 2 waves/SIMD alongside the attention MFMAs → the row max is silently not subtracted for lanes 48–63, giving nondeterministic wrong attention outputs.

## Notes

- The reproducer avoids inline asm except for the **one** packed instruction under test; the aggressor MFMA uses `__builtin_amdgcn_mfma`. The single asm line is required because the HIP-C compiler (hipcc, ROCm 7.0 LLVM) scalarizes/unpacks the pure-C++ packed "swap multiply" into scalar `v_mul_f32` in this context, so the packed instruction never reaches hardware.
- Links above are Meta-internal; attach the tarball when sending to AMD.
- Open before sending externally: confirm on a second `gfx950` host (not machine-specific).

## Questions for AMD

1. Known erratum / expected under any documented restriction?
2. Arch scope — `gfx950` only (MI350X + MI355X), or other parts?
3. Full opcode scope — other VOP3P ops reading a VGPR `src1` high dword? (F16-packed and `V_PK_MOV_B32` tested **not** affected.)
4. Is operand commutation the recommended avoidance, or is a hardware/firmware fix expected?
