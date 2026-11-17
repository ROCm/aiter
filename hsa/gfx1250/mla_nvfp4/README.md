# gfx1250 R1 NVFP4 N128

`mla_nvfp4_n128.co` is the precompiled padding8, threshold-configurable,
four-slot N128 ASM implementation. Its ABI is ten pointers (80 bytes), in order:

`qn, qa, cn, ca, page_indptr, page_indices, last_page_lens, out, lse, params`.

Launch one workgroup per query: grid `(B,1,1)`, block `(128,1,1)`. The code
object declares **256KiB fixed LDS**; launch with zero dynamic LDS. Its
instruction entry is aligned to 16KiB. The entry address is
`kernel_object + kernel_code_entry_byte_offset`; descriptor alignment alone
does not establish instruction entry alignment.
All valid calls write BF16 output and FP32 natural-log LSE.

`manifest.json` records the shipped CO's SHA-256 hash, resource usage and
entry alignment. AITER loads this binary directly and compiles the C++ wrapper
and HIP producer kernels when needed. The ASM source and its offline build
tooling are maintained separately and are not distributed with AITER.

See `docs/mla_nvfp4.md` for the supported layout, fixed-gain numerical contract,
GPU producers and public API. Performance depends on workload and runtime;
these artifacts do not claim end-to-end model accuracy or a best schedule.
