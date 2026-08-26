# Qwen-Image VAE Conv Shape Report

- Observed dtype: **bf16**
- Resolutions: 7 official T2I sizes × encode/decode
- Total hook records: 469

## Parameter counts (safetensors header, no full download)

| Component | Params | Disk dtype (sample) |
|-----------|--------|---------------------|
| vae | 126,892,531 | BF16 |
| transformer | 20,430,401,088 | BF16 |
| text_encoder | 8,292,166,656 | BF16 |

## Conv layer counts

- CausalConv3d (static): 61
- Conv2d (static): 10
- time_conv dead paths: 4 (0 calls each)

## Top layers by MACs @ 1328×1328 decode

| section | path | cls | logical_in | out | MACs | dead |
|---------|------|-----|------------|-----|------|------|
| decoder | `decoder.up_blocks.1.resnets.0.conv2` | QwenImageCausalConv3d | (1, 384, 1, 332, 332) | (1, 384, 1, 332, 332) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.1.resnets.1.conv1` | QwenImageCausalConv3d | (1, 384, 1, 332, 332) | (1, 384, 1, 332, 332) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.1.resnets.1.conv2` | QwenImageCausalConv3d | (1, 384, 1, 332, 332) | (1, 384, 1, 332, 332) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.1.resnets.2.conv1` | QwenImageCausalConv3d | (1, 384, 1, 332, 332) | (1, 384, 1, 332, 332) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.1.resnets.2.conv2` | QwenImageCausalConv3d | (1, 384, 1, 332, 332) | (1, 384, 1, 332, 332) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.2.resnets.0.conv1` | QwenImageCausalConv3d | (1, 192, 1, 664, 664) | (1, 192, 1, 664, 664) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.2.resnets.0.conv2` | QwenImageCausalConv3d | (1, 192, 1, 664, 664) | (1, 192, 1, 664, 664) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.2.resnets.1.conv1` | QwenImageCausalConv3d | (1, 192, 1, 664, 664) | (1, 192, 1, 664, 664) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.2.resnets.1.conv2` | QwenImageCausalConv3d | (1, 192, 1, 664, 664) | (1, 192, 1, 664, 664) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.2.resnets.2.conv1` | QwenImageCausalConv3d | (1, 192, 1, 664, 664) | (1, 192, 1, 664, 664) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.2.resnets.2.conv2` | QwenImageCausalConv3d | (1, 192, 1, 664, 664) | (1, 192, 1, 664, 664) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.3.resnets.0.conv1` | QwenImageCausalConv3d | (1, 96, 1, 1328, 1328) | (1, 96, 1, 1328, 1328) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.3.resnets.0.conv2` | QwenImageCausalConv3d | (1, 96, 1, 1328, 1328) | (1, 96, 1, 1328, 1328) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.3.resnets.1.conv1` | QwenImageCausalConv3d | (1, 96, 1, 1328, 1328) | (1, 96, 1, 1328, 1328) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.3.resnets.1.conv2` | QwenImageCausalConv3d | (1, 96, 1, 1328, 1328) | (1, 96, 1, 1328, 1328) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.3.resnets.2.conv1` | QwenImageCausalConv3d | (1, 96, 1, 1328, 1328) | (1, 96, 1, 1328, 1328) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.3.resnets.2.conv2` | QwenImageCausalConv3d | (1, 96, 1, 1328, 1328) | (1, 96, 1, 1328, 1328) | 877,672,267,776 | False |
| decoder | `decoder.up_blocks.1.upsamplers.0.resample.1` | Conv2d | (1, 384, 664, 664) | (1, 192, 664, 664) | 585,114,845,184 | False |
| decoder | `decoder.up_blocks.2.upsamplers.0.resample.1` | Conv2d | (1, 192, 1328, 1328) | (1, 96, 1328, 1328) | 585,114,845,184 | False |
| decoder | `decoder.up_blocks.1.resnets.0.conv1` | QwenImageCausalConv3d | (1, 192, 1, 332, 332) | (1, 384, 1, 332, 332) | 438,836,133,888 | False |

## Conv3d layers @ 1328 decode

| path | logical_in | padded_in | out | MACs |
|------|------------|-----------|-----|------|
| `decoder.conv_in` | (1, 16, 1, 166, 166) | (1, 16, 3, 168, 168) | (1, 384, 1, 166, 166) | 9,142,419,456 |
| `decoder.conv_out` | (1, 96, 1, 1328, 1328) | (1, 96, 3, 1330, 1330) | (1, 3, 1, 1328, 1328) | 27,427,258,368 |
| `decoder.mid_block.resnets.0.conv1` | (1, 384, 1, 166, 166) | (1, 384, 3, 168, 168) | (1, 384, 1, 166, 166) | 219,418,066,944 |
| `decoder.mid_block.resnets.0.conv2` | (1, 384, 1, 166, 166) | (1, 384, 3, 168, 168) | (1, 384, 1, 166, 166) | 219,418,066,944 |
| `decoder.mid_block.resnets.1.conv1` | (1, 384, 1, 166, 166) | (1, 384, 3, 168, 168) | (1, 384, 1, 166, 166) | 219,418,066,944 |
| `decoder.mid_block.resnets.1.conv2` | (1, 384, 1, 166, 166) | (1, 384, 3, 168, 168) | (1, 384, 1, 166, 166) | 219,418,066,944 |
| `decoder.up_blocks.0.resnets.0.conv1` | (1, 384, 1, 166, 166) | (1, 384, 3, 168, 168) | (1, 384, 1, 166, 166) | 219,418,066,944 |
| `decoder.up_blocks.0.resnets.0.conv2` | (1, 384, 1, 166, 166) | (1, 384, 3, 168, 168) | (1, 384, 1, 166, 166) | 219,418,066,944 |
| `decoder.up_blocks.0.resnets.1.conv1` | (1, 384, 1, 166, 166) | (1, 384, 3, 168, 168) | (1, 384, 1, 166, 166) | 219,418,066,944 |
| `decoder.up_blocks.0.resnets.1.conv2` | (1, 384, 1, 166, 166) | (1, 384, 3, 168, 168) | (1, 384, 1, 166, 166) | 219,418,066,944 |
| `decoder.up_blocks.0.resnets.2.conv1` | (1, 384, 1, 166, 166) | (1, 384, 3, 168, 168) | (1, 384, 1, 166, 166) | 219,418,066,944 |
| `decoder.up_blocks.0.resnets.2.conv2` | (1, 384, 1, 166, 166) | (1, 384, 3, 168, 168) | (1, 384, 1, 166, 166) | 219,418,066,944 |
| `decoder.up_blocks.1.resnets.0.conv1` | (1, 192, 1, 332, 332) | (1, 192, 3, 334, 334) | (1, 384, 1, 332, 332) | 438,836,133,888 |
| `decoder.up_blocks.1.resnets.0.conv2` | (1, 384, 1, 332, 332) | (1, 384, 3, 334, 334) | (1, 384, 1, 332, 332) | 877,672,267,776 |
| `decoder.up_blocks.1.resnets.0.conv_shortcut` | (1, 192, 1, 332, 332) | (1, 192, 1, 332, 332) | (1, 384, 1, 332, 332) | 16,253,190,144 |
| `decoder.up_blocks.1.resnets.1.conv1` | (1, 384, 1, 332, 332) | (1, 384, 3, 334, 334) | (1, 384, 1, 332, 332) | 877,672,267,776 |
| `decoder.up_blocks.1.resnets.1.conv2` | (1, 384, 1, 332, 332) | (1, 384, 3, 334, 334) | (1, 384, 1, 332, 332) | 877,672,267,776 |
| `decoder.up_blocks.1.resnets.2.conv1` | (1, 384, 1, 332, 332) | (1, 384, 3, 334, 334) | (1, 384, 1, 332, 332) | 877,672,267,776 |
| `decoder.up_blocks.1.resnets.2.conv2` | (1, 384, 1, 332, 332) | (1, 384, 3, 334, 334) | (1, 384, 1, 332, 332) | 877,672,267,776 |
| `decoder.up_blocks.2.resnets.0.conv1` | (1, 192, 1, 664, 664) | (1, 192, 3, 666, 666) | (1, 192, 1, 664, 664) | 877,672,267,776 |
| `decoder.up_blocks.2.resnets.0.conv2` | (1, 192, 1, 664, 664) | (1, 192, 3, 666, 666) | (1, 192, 1, 664, 664) | 877,672,267,776 |
| `decoder.up_blocks.2.resnets.1.conv1` | (1, 192, 1, 664, 664) | (1, 192, 3, 666, 666) | (1, 192, 1, 664, 664) | 877,672,267,776 |
| `decoder.up_blocks.2.resnets.1.conv2` | (1, 192, 1, 664, 664) | (1, 192, 3, 666, 666) | (1, 192, 1, 664, 664) | 877,672,267,776 |
| `decoder.up_blocks.2.resnets.2.conv1` | (1, 192, 1, 664, 664) | (1, 192, 3, 666, 666) | (1, 192, 1, 664, 664) | 877,672,267,776 |
| `decoder.up_blocks.2.resnets.2.conv2` | (1, 192, 1, 664, 664) | (1, 192, 3, 666, 666) | (1, 192, 1, 664, 664) | 877,672,267,776 |
| `decoder.up_blocks.3.resnets.0.conv1` | (1, 96, 1, 1328, 1328) | (1, 96, 3, 1330, 1330) | (1, 96, 1, 1328, 1328) | 877,672,267,776 |
| `decoder.up_blocks.3.resnets.0.conv2` | (1, 96, 1, 1328, 1328) | (1, 96, 3, 1330, 1330) | (1, 96, 1, 1328, 1328) | 877,672,267,776 |
| `decoder.up_blocks.3.resnets.1.conv1` | (1, 96, 1, 1328, 1328) | (1, 96, 3, 1330, 1330) | (1, 96, 1, 1328, 1328) | 877,672,267,776 |
| `decoder.up_blocks.3.resnets.1.conv2` | (1, 96, 1, 1328, 1328) | (1, 96, 3, 1330, 1330) | (1, 96, 1, 1328, 1328) | 877,672,267,776 |
| `decoder.up_blocks.3.resnets.2.conv1` | (1, 96, 1, 1328, 1328) | (1, 96, 3, 1330, 1330) | (1, 96, 1, 1328, 1328) | 877,672,267,776 |
| `decoder.up_blocks.3.resnets.2.conv2` | (1, 96, 1, 1328, 1328) | (1, 96, 3, 1330, 1330) | (1, 96, 1, 1328, 1328) | 877,672,267,776 |
| `post_quant_conv` | (1, 16, 1, 166, 166) | (1, 16, 1, 166, 166) | (1, 16, 1, 166, 166) | 14,108,672 |

## Conv2d layers @ 1328 decode

| path | logical_in | out | MACs |
|------|------------|-----|------|
| `decoder.mid_block.attentions.0.proj` | (1, 384, 166, 166) | (1, 384, 166, 166) | 8,126,595,072 |
| `decoder.mid_block.attentions.0.to_qkv` | (1, 384, 166, 166) | (1, 1152, 166, 166) | 24,379,785,216 |
| `decoder.up_blocks.0.upsamplers.0.resample.1` | (1, 384, 332, 332) | (1, 192, 332, 332) | 146,278,711,296 |
| `decoder.up_blocks.1.upsamplers.0.resample.1` | (1, 384, 664, 664) | (1, 192, 664, 664) | 585,114,845,184 |
| `decoder.up_blocks.2.upsamplers.0.resample.1` | (1, 192, 1328, 1328) | (1, 96, 1328, 1328) | 585,114,845,184 |

## Dtype combinations (all records)

| in | weight | bias | out | count |
|----|--------|------|-----|-------|
| bfloat16 | bfloat16 | bfloat16 | bfloat16 | 469 |

## Notes

- Single-frame T=1: all 3×3×3 CausalConv3d equivalent to Conv2d with `W[:,:,-1]`.
- MIOpen/cuDNN accumulate bf16 conv in fp32 internally (not visible in module dtypes).
- RMS_norm and Upsample run normalize/interp in fp32 but cast back before conv input.
