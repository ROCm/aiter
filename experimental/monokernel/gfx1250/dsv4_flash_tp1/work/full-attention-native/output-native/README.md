# Native ATOM attention output reference

`native_output.py` uses the installed ATOM model methods directly: `DeepseekV4Attention.process_weights_after_loading`, `_wo_a_grouped_lora`, `_attn_post`, `LinearBase.forward` and `Block.hc_post`. TP1 metadata is stubbed during Linear construction only. No native compute kernels are replaced. The equivalent RowParallelLinear at TP1 needs no communication.

The layer0 checkpoint fixture contains original unshuffled tensors:

| Key | Shape | Type |
|---|---|---|
| OA | 8192×4096 (8 groups ×1024×4096) | FP8 E4M3 |
| OAS | 64×32 (8×8×32) | E8M0 |
| OB | 4096×8192 | FP8 E4M3 |
| OBS | 32×64 | E8M0 |

On gfx1250 the native model retains OA in FP8 and preshuffles 16×16 tiles. It does not take its BF16 OA fallback. Native output sequence is:

1. HIP inverse RoPE + per128 FP8 activation quant: input BF16[1,64,512], output FP8[1,8,4096] and uint8 scales[1,8,32]. Rotation is FP32, with no intermediate BF16 rounding. Actual layer0 BF16 RoPE table at position1024 is used.
2. AITER preshuffled grouped FP8 BMM: BF16[1,8,1024], viewed as [1,8192].
3. HIP per128 E8M0 activation quant.
4. AITER preshuffled OB GEMM: BF16[1,4096].
5. Extracted native `aiter.mhc_post`: combines OB output with residual[1,4,4096] using post[1,4] and comb[1,4,4], producing BF16[1,4,4096]. The optional `res_preshuffle=True` is passed through when the residual is in native packed layout.

**Boundary qualification:** step5 is ATOM's actual standalone `hc_post` implementation, but normal serving with fused HC usually combines this post operation with the following FFN pre-mHC. Thus output4-kernel sequence is literal serving dispatch, while standalone post is an extracted native reference for the requested attention-only boundary. Do not claim native serving has a separate post launch in that configuration.

`NativeOutput(position=1024)` returns a callable `n(o)` for steps1–4; `n.oa(o)` returns the OA result; `n.post(y,residual,post,comb,res_preshuffle=False)` runs step5; `n.full(...)` combines them. `n.rotary_emb.cos_sin_2d()` returns BF16 [2048,32] caches. The input attention output is not mutated on the selected FP8 path.

`check.py` compares native GEMM outputs against FP32 mathematical GEMM using the exact native-quantized inputs and dequantized original weights, plus mHC post against the mathematical residual combination. Checks are shape/layout validation, not a new acceptance tolerance. Relative RMS: OA0.00091022, OB0.00090183, post4.51e-9; finite outputs. The native FP8 GEMM accumulation also differs from mathematical FP32 reference; this is relevant when diagnosing prototype parity. Profiler traces and exact kernel-name JSON are included; their one-off durations are not benchmark results.

Run in the existing image:

```sh
docker exec -e HIP_VISIBLE_DEVICES=2 -e PYTHONPATH=/app/aiter:/app/ATOM vigokhal_dsv4_tp4 bash -c 'cd /work/full-attention-native/output-native; python check.py'
```

For the actual native shuffled residual entry, `post(...,res_preshuffle=True)` launches **two** kernels: `mhc_res_unshuffle_h4096_hc4_ks64_b2` followed by the HIP mHC post kernel. Output is plain [1,4,4096]; `check_shuffled.py` confirmed it is bitwise equal to the native plain-residual post. Therefore prefix10 + output4 + extracted post2 =16 native reference launches for this boundary (including the layout adapter). A fused kernel can unshuffle internally and directly produce the same plain output.
