# GEMM tuning coverage

`gemm_cases.py` is the executable inventory. `python tune_gemm.py --list` lists
the cases and their dimensions without importing Torch or GPU kernels. A case
builds representative inputs and calls the public wrapper; the tuner follows
the wrapper's config lookup to its architecture, backend, family and filename.
No architecture-specific tuning harness is needed.

When adding a public GEMM wrapper, add a case or document an explicit exception
in this inventory. Backend kernel variants selected by `kernel_type` in
the config are searched through the same case.

## Gaps in the previous 16-harness workflow

These 24 cases had no dedicated harness. They are now registered in the shared
tuner, with input generation reused from the existing correctness tests.
Registration is not a claim that GPU tuning has been run on every architecture.

| Group | Added tuning cases |
| --- | --- |
| Basic and persistent | `gemm_a16w16_persistent`, `gemm_a16wfp4_preshuffle`, `gemm_a8w8_preshuffle`, `gemm_a8w8_blockscale_group32`, `gemm_afp4wfp4_preshuffled_scales`, `gemm_afp8wfp8` |
| Batched | `batched_gemm_a8w8`, `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`, `batched_gemm_a16wfp4`, `batched_gemm_afp4wfp4`, `batched_gemm_afp4wfp4_pre_quant` |
| Fused mixed projections and quantization | `fused_gemm_a16w16_quant_x`, `fused_gemm_a8w8_blockscale_a16w16`, `fused_gemm_afp4wfp4_a16w16`, `fused_gemm_afp4wfp4_preshuffle_a16w16` |
| Fused multiply/add | `fused_gemm_a8w8_blockscale_mul_add`, `fused_gemm_afp4wfp4_mul_add`, `fused_gemm_afp4wfp4_preshuffle_add_mul` |
| Fused split/concatenate | `fused_gemm_a8w8_blockscale_split_cat`, `fused_gemm_a8w8_blockscale_preshuffle_split_cat`, `fused_gemm_afp4wfp4_split_cat`, `fused_gemm_afp4wfp4_preshuffle_split_cat` |
| Fused feed-forward | `ff_a16w16_fused_gated`, `ff_a16w16_fused_ungated` |

The original 16 cases remain: `gemm_a16w16`, `gemm_a16w16_atomic`,
`gemm_a16w16_gated`, `gemm_a16w8_blockscale`,
`gemm_a16w8_blockscale_preshuffle`, `gemm_a16wfp4`, `gemm_a8w8`,
`gemm_a8w8_blockscale`, `gemm_a8w8_blockscale_preshuffle`,
`gemm_a8w8_per_token_scale`, `gemm_a8wfp4`, `gemm_afp4wfp4`,
`gemm_afp4wfp4_preshuffle`, `gemm_afp4wfp4_pre_quant`,
`gemm_afp8wfp8_preshuffle`, and `batched_gemm_bf16`.

There are 40 registered cases for the 41 public wrapper functions in `gemm/`:
38 wrappers have a named case, and two cases cover extra modes of a wrapper
(persistent A16W16 and preshuffled mixed FP4/BF16). The three exceptions are:

| Wrapper | How it is covered |
| --- | --- |
| `ff_a16w16_nogate` | A composition of two `gemm_a16w16` calls. Tune each call's dimensions separately. |
| `ff_a16w16_gated` | A composition of `gemm_a16w16_gated` and `gemm_a16w16`. Tune each call separately. |
| `gemm_afp4wfp4_preshuffled_weight_scales` | Deprecated alias of `gemm_afp4wfp4_preshuffle`; use that case. |

`batched_gemm_afp4wfp4_pre_quant` is also a deprecated alias. Its case delegates
to the equivalent `batched_gemm_a16wfp4` case for discoverability.

## Triton, Gluon and architecture coverage

The tuner has no architecture allowlist. Run it on the target GPU; it reads
`configs/<detected-arch>/<resolved-backend>/gemm/<family>/DEFAULT.json`.
Authors must provide a working implementation and default config for that
architecture/backend. Adding a future architecture does not require another
tuner or case when the wrapper's input contract is unchanged.

Current dense/batched Gluon wrapper routes are:

| Cases | Gluon architecture(s) |
| --- | --- |
| `gemm_a8w8`, `gemm_a8w8_preshuffle` | `gfx950` |
| `gemm_a8w8_blockscale` | `gfx950`, `gfx1250` |
| `gemm_afp4wfp4` | `gfx950` |
| `gemm_a16w16`, `gemm_a16w16_persistent`, `batched_gemm_bf16` | `gfx1250` |
| `gemm_a8w8_blockscale_preshuffle`, `gemm_afp4wfp4_preshuffle`, `gemm_afp8wfp8_preshuffle` | `gfx1250` |

Pass `--backend triton`, `--backend gluon`, or `--backend both` for wrappers with
a backend argument. Both backends are tuned independently; omitting the flag
preserves the wrapper's default, including Gluon preference on supported gfx1250
routes. Triton remains selectable where supported. `gemm_a8w8_preshuffle` is
Gluon-only, so omit the flag. Other dense,
batched, fused and feed-forward cases currently use Triton. There is no dense
Gluon GEMM implementation on `gfx942` in this tree; the shared tuner does not
create one. The `gfx1250` Gluon MXFP4 route takes preshuffled weights, so use
`gemm_afp4wfp4_preshuffle`, not the unshuffled wrapper.

Hardware restrictions still apply. In particular, group32 FP8 currently
asserts `gfx950`, and MXFP4 cases require an architecture supporting their
instructions and layouts. A missing FP4 default on `gfx942` is not a missing
tuning harness. A config directory alone also does not establish that a
backend supports a wrapper; the wrapper's dispatch is authoritative.

## Representative modes and shared families

Cases normally use BF16 output and TN layout. Additional shape dimensions are:

| Case group | Dimensions beyond `M`, `N`, `K` |
| --- | --- |
| Batched GEMMs | `B` |
| Mixed FP8/FP4 + BF16 | Use `N1` and `N2` instead of `N`: low-precision and BF16 projection widths. |
| Split/concatenate | `D` and `S3`; `N` must be divisible by `D`. The case splits `N / D` into two nearly equal widths. |
| Fused gated feed-forward | `N` is the full up-projection width, so it must be even. |

The lookup key is the runtime contract. Some runtime wrappers intentionally
share a family and therefore share tuned results:

- Fused quant-X and ordinary A16W16 read `GEMM-A16W16`.
- Scale-only FP4 preshuffling and ordinary FP4 read `GEMM-AFP4WFP4`.
- Split/concatenate and several multiply/add paths read their corresponding
  unfused GEMM family; dedicated multiply/add families exist for some `gfx950`
  paths.
- Gated and ungated fused feed-forward both read `FF-A16W16-fused`.
- The Gluon A8W8 preshuffled and unshuffled wrappers share `GEMM-A8W8`.
- Atomic pre-quant FP4 and A16WFP4 share `GEMM-A16WFP4`.
- Most batched wrappers currently do not include `B` in their lookup. BF16
  batched GEMM does include it.

Tuning such cases sequentially can replace the same JSON bucket. A case name
does not create a separate runtime config namespace. If two modes need
independent winners or different config key contracts, the author should
first give the wrapper distinct lookup families or specialized filenames and
matching defaults. The tuner then follows those choices automatically.
Likewise, flags such as dtype, activation, layout, multiply/add fuse type,
group32 scale grouping and split-cat proportions need distinct lookup keys
before they can retain independent tuned results.

rocprofv3 times only selected GEMM/reduction kernels between GPU markers;
wrapper overhead and output resets are excluded. Fused feed-forward cases
select `_ff_` kernel names and zero their atomic output each invocation.
FP4 fused cases disable AOT metadata while tuning so each candidate is
compiled from its actual config.

## MoE and grouped GEMMs needing a different lookup integration

These public wrappers do not call `get_gemm_config()` and are not registered
in this GEMM tuner. No dedicated Triton/Gluon tuning harness for them is
present in this tuning directory. Other CK/assembly MoE tuners do not tune
these implementations.

| Wrappers | Existing config mechanism / integration needed |
| --- | --- |
| `moe_gemm_a16w4` | Architecture/backend-specific Python heuristics. Triton plus Gluon on `gfx950` and `gfx1250`; needs a persisted lookup contract and routing-aware inputs. |
| `moe_gemm_a4w4`, `moe_gemm_a8w4` | Mixture of Python heuristics and `get_moe_dispatch()` tables. Gluon is available on `gfx1250`; routing block size and scale/epilogue modes are part of dispatch. |
| `moe_gemm_a8w8`, `moe_gemm_a8w8_blockscale` | Python `get_kernel_config()` heuristics; needs lookup integration and expert-routing inputs. |
| `moe_gemm_int8_smoothquant` | Python config heuristics and automatic Triton/Gluon dispatch (Gluon on eligible `gfx942` shapes); needs lookup and explicit backend selection. |
| `moe_gemm_per_token` | Fixed module tile constants; needs lookup integration and grouped inputs. |
| `moe_gemm_mxfp8` | `get_tuned_kernel_config()` under `moe/mxfp8_fnuz`; needs an adapter for that config format and grouped inputs. |
| `gmm`, `ptgmm`, `nptgmm` | Separate grouped-matmul config lookup under `gmm/`; needs an adapter and group-size/transposition cases. |

For these families, reuse the benchmark loop after defining a suitable
lookup adapter. Do not flatten routing-dependent
MoE configs into an ordinary `(M, N, K)` lookup: expert distribution and
routing block size affect the kernel's launch and correctness.
