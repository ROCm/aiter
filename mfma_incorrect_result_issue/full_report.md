# gfx950 defect: packed-FP32 `src1` high-dword operand read returns 0 during a concurrent MFMA

**Hardware:** AMD Instinct MI350X (`gfx950`, device `0x75a0`).
**Software:** ROCm 7.0; observed with the LLVM/clang in that toolchain.
**Severity:** silent wrong results (no fault, no hang). Nondeterministic.
**Status:** reduced to a standalone HIP reproducer; mechanism characterized by A/B experiments.

---

## 1. One-paragraph description

On `gfx950`, a packed-FP32 ALU instruction — `V_PK_MUL_F32`, `V_PK_ADD_F32`, or
`V_PK_FMA_F32` — whose **`src1` is a VGPR read with its high dword selected for the
low result lane** (`op_sel` bit 0 set on `src1`, clear on `src0`) reads that operand
as **0** in hardware lanes **48–63**, **if and only if another wave resident on the
same SIMD is concurrently executing a double-rate XDL MFMA**. The packed op then
produces `src0 * 0` (or `src0 + 0`). The MFMA’s own result is correct; the `src1`
register is intact; only the `src1` high-dword *read* inside the packed op is lost.

---

## 2. Exact trigger condition

All of the following are required:

1. **A packed-FP32 op** `V_PK_{MUL,ADD,FMA}_F32` where **`src1` is a VGPR** and
   `op_sel[0]` (the low result lane’s selector for `src1`) is **set** (reads the high
   dword), while `src0`’s `op_sel[0]` is **clear**.
   - `src1` being an SGPR or inline constant is **safe**.
   - The commuted arrangement (high-dword select on `src0` instead of `src1`) is **safe**.
   - `op_sel_hi` does not matter; when `op_sel_hi[0]` is also 0 the high result lane is corrupted too.
2. **Concurrency with a double-rate XDL MFMA** issued by **another wave co-resident
   on the same SIMD**. Triggering MFMAs: `V_MFMA_F32_{32x32x16,16x16x32}_F16`,
   `..._BF16`, `V_MFMA_F32_{32x32x64,16x16x128}_F8F6F4`, `V_MFMA_I32_32x32x32_I8`,
   and the AGPR-accumulator variants. **Not** triggered by legacy MFMAs
   (`32x32x8_f16`, `16x16x16_f16`, `4x4x4_f16`), DGEMM (`f64`), SGEMM (`32x32x2_f32`),
   `V_SMFMAC_*`, `V_DOT2_*`, or transcendentals.
3. **≥2 waves co-resident on one SIMD.** A single wave never reproduces it, at any
   instruction spacing.

The corruption is confined to **hardware lanes 48–63** (the last of the four 16-lane
passes a 64-wide wave issues), and the lost read returns exactly **0.0**.

---

## 3. Standalone reproducer

`pk_mfma_crosswave.cpp` (in this directory). No special flags.

```sh
hipcc --offload-arch=gfx950 pk_mfma_crosswave.cpp -o pk_mfma_crosswave

./pk_mfma_crosswave --op mul      # V_PK_MUL_F32  -> REPRODUCED
./pk_mfma_crosswave --op add      # V_PK_ADD_F32  -> REPRODUCED
./pk_mfma_crosswave --op fma      # V_PK_FMA_F32  -> REPRODUCED
./pk_mfma_crosswave --op mul --safe     # commuted operands -> clean
./pk_mfma_crosswave --op mul --single   # one wave only      -> clean
```

Each victim lane repeatedly computes a value that must equal `96.0` and keeps a
running minimum (so a transient wrong read cannot be masked by a later correct one).
Any stored value `!= 96.0` is a hardware miscompute.

**Observed on MI350X / ROCm 7.0** (6 launches, 2048 blocks each):

| mode | result |
|---|---|
| `--op mul` | 238,944 / 3,145,728 victim lanes wrong |
| `--op add` | 70,528 / 3,145,728 wrong |
| `--op fma` | 49,424 / 3,145,728 wrong |
| `--op {mul,add,fma} --safe` | 0 |
| `--op mul --single` | 0 |

The failing value is always `0.0` in lanes 48–63.

### A note on inline assembly

Per request, the reproducer avoids inline asm wherever possible: the **aggressor
MFMA uses the `__builtin_amdgcn_mfma` intrinsic**. The **victim packed op uses a
single inline-asm instruction**, which is genuinely necessary: the HIP-C compiler
(hipcc, ROCm 7.0 LLVM) does not reliably emit the vulnerable packed form from
plain C++. In the reproducer’s context (a loop adjacent to MFMA activity) its
instruction selection scalarizes/unpacks the packed “swap multiply”
(`r.x = a.x*b.y; r.y = a.y*b.x`) into scalar `v_mul_f32`, which is not vulnerable, so
the packed instruction never reaches the hardware. Forcing the one instruction under
test makes the probe deterministic and compiler-version-independent.

---

## 4. Localization (what it is and is not)

Established by A/B experiments (operands chosen with distinct values so each quantity
is independently checkable in a failing wave):

- **It is the `src1` read, not the MFMA output.** With the packed op’s inputs
  disjoint from the MFMA output, it still fails and the MFMA output is correct and
  unused. With the real-kernel layout where the packed op consumes the MFMA output as
  `src0`, the `src0` read is correct — only the `src1` high-dword read is lost.
- **It is not a register clobber.** Reading `src1` back after the op yields the
  correct value; the fault is transient in the read datapath.
- **It is not operand readiness.** Forcing `src1` fully settled (written, then >100
  wait-states before any packed op) fails identically; a never-recomputed constant
  `src1` fails hardest; producing `src1` more recently fails *less* (timing dilution).
  A readiness/RAW hazard would also appear on a single wave — it never does.
- **It is not a register-location / bank conflict.** The `src1` register number is
  irrelevant, including when it exactly matches the neighbor’s MFMA accumulator or
  input registers (different waves occupy physically separate register-file regions).
- **It is a `src1` (operand B) read-port issue.** The identical arithmetic with the
  high-dword select on `src0` (operand A) is always correct — hence the commute works.
- **It is cross-wave on a shared SIMD.** Forcing one wave per SIMD (via register/LDS
  pressure) eliminates it; forcing two waves per SIMD maximizes it. MFMAs on *other*
  SIMDs of the same CU do not trigger it.

The surviving explanation is contention on the SIMD’s operand-B high-dword read path
(lanes 48–63) while the double-rate XDL unit is fetching matrix operands. This is a
cross-wave hardware condition: it cannot be expressed or avoided by per-wave
instruction scheduling (one wave cannot wait on another wave’s instruction). A
workgroup barrier bracketing the packed op removes it *within a workgroup* but not
across workgroups that merely share a SIMD, confirming the cross-wave nature.

---

## 5. Impact

Real production flash-attention kernels hit this. In a Triton/LLVM-compiled
attention kernel, the softmax step `exp2(qk*scale - m_new)` lowers to
`V_PK_ADD_F32 ... op_sel:[0,1]` where `src1` is the per-row max (`m_new`) broadcast
into a VGPR — exactly the vulnerable form — running alongside the attention MFMAs at
2 waves/SIMD. The result is `qk*scale - 0` for lanes 48–63, i.e. the row max is
silently not subtracted, producing nondeterministic wrong attention outputs.

---

## 6. Software mitigation in use (pending guidance)

Commuting the operands so the high-dword select lands on `src0` (operand A) is a
reliable, precisely-targeted avoidance for all three ops (0 failures in every test).
It cannot be replaced by wait-states/barriers because the hazard is cross-wave.

**Questions for AMD:**
1. Is this a known erratum? Expected behavior under any documented restriction?
2. Correct scope — is it `gfx950`-only (MI350X and MI355X), or also other RDNA/CDNA parts?
3. Does it extend to other packed VOP3P ops that read a VGPR `src1` high dword
   (e.g. `V_PK_*_F16`, `V_PK_MOV_B32`)? Our tests found F16 packed and `V_PK_MOV_B32`
   **not** affected, but please confirm the full opcode scope.
4. Is operand commutation the recommended avoidance, or is there a hardware/firmware fix?

---

## 7. Files

- `pk_mfma_crosswave.cpp` — standalone reproducer (this report’s §3).
- `REPORT.md` — this document.
