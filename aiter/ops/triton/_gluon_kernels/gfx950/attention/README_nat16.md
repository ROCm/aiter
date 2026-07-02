# dsa_fwd_v4_gluon_nat16 — verified best (gfx950 / MI350)

nat16 = nat4 + a 4-part stack, VERIFIED BEST for the gluon fwd path.
Config: BLOCK_H=64, TILE_K=32, num_warps=4.

## Stack (on top of nat4 early-gather + int32 + deferred-rescale)
1. **16x16x32 mfma** (CDNA4): `instr_shape=[16,16,32]` on QK+PV. K=32 in the same
   16 cycles as 16x16x16 => ~2x matrix rate. (-1.7% alone)
2. **convert-fix**: load topk directly in each consumer layout (no convert_layout
   LDS round-trips). (-1.0%)
3. **topk HBM->LDS staging**: topk goes HBM->LDS->VGPR, moving its wait from vmcnt
   (shared with the KV gather) to lgkmcnt => decouples topk from the gather drain.
   "load 64 use 32" trick lowers the sub-warp async tile. (-1.7%)
4. **krope b128**: `blk_krope size_per_thread=[8,1], threads_per_warp=[8,8]` =>
   8 D_ROPE/lane = 128-bit rope load (was [2,1]=b32). (-8.8%)  [hardcoded for TILE_K=32]

## Verified (cv350 MI350 proxy, same-clock)
- **-13.4% (TOPK=1152) / -16.3% (S=8192)** vs nat4.  ~576 TF.
- Correctness: bit-exact vs nat15 (O+LSE, 5 sizes); ~3.9e-3 O / ~1e-6 LSE vs dedup
  reference (bf16-level, from the 16x16x32 accumulation reorder).

## Known gaps
- blk_krope is hardcoded [8,8] for TILE_K=32; needs a parametric fix to compile the
  TK16 autotune configs.
- Kernel is VALU-bound (~38.7%, ~1284c = softmax ~690c + gather-address ~600c).
  Remaining gluon-level levers: exp2-fold of softmax (see nat17), cross-tile
  softmax pipeline (LDS/VGPR-blocked at BH64/TK32). The unified 576 gather /
  offset-dedup needs a scalar-base gather primitive (HIP path), not available in gluon.
