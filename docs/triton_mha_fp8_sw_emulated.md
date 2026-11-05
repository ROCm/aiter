# Triton MHA with fp8 inputs on archs without fp8 hardware

## Scope

gfx1100, gfx1101, gfx1102, gfx1150 and gfx1151 have no fp8 WMMA and no fp8 conversion instructions. They are listed in `SW_EMULATED_FP8_ARCHS` in `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py`. The Triton MHA in `aiter.ops.triton.attention.mha_v3` accepts fp8 q, k and v on these archs, but every fp8 value is converted to fp16 before a dot product. This note applies to these archs only. Archs with fp8 hardware run the fp8 dot products directly.

## Prefer fp16 or bf16 inputs

- fp8 inputs bring no compute advantage. The dot products run in fp16 in either case, and fp8 inputs add the conversion.
- The `bshd` prefill converts k and v to fp16 once per call and q once per program. The varlen and `bhsd` layouts, decode and backward convert the fp8 operands inside the kernel, on every tile. Before the one-time conversion was added, the `bshd` prefill did the same and was 3.8x to 24.0x slower on gfx1101.
- During a `bshd` prefill call the fp16 copies of k and v exist next to the fp8 tensors. k and v then take 3 bytes per element for the duration of the call, against 2 bytes with fp16 inputs. fp8 saves memory only for tensors that are kept between calls.
- `flash_attn_fp8_func` takes fp16, bf16 or fp32 inputs and quantizes them to fp8 on every call. On these archs this adds the quantization and the conversion back to fp16, with no compute advantage in return. `flash_attn_func` with the original inputs avoids both.

On gfx1101, the `bshd` prefill with pre-quantized fp8 inputs took a median of 1.16x the time of `flash_attn_func` with fp16 inputs, over 238 shapes.

## Output dtype

For fp8 q, k and v the forward output is fp32, on every arch (`flash_attn_triton_amd/interface_v3.py`, where `out` is allocated). `flash_attn_fp8_func` therefore returns fp32 also for fp16 or bf16 inputs. `flash_attn_func` and `flash_attn_fp8_func` take no output tensor, so the caller cannot choose another dtype. With fp16 or bf16 inputs the output has the input dtype and half the size.
