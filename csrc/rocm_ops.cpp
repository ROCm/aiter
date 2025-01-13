#include "activation.h"
#include "attention.h"
#include "attention_ck.h"
#include "attention_asm.h"
#include "cache.h"
#include "custom_all_reduce.h"
#include "custom.h"
#include "moe_op.h"
#include "moe_sorting.h"
#include "norm.h"
#include "pos_encoding.h"
#include "rmsnorm.h"
#include "smoothquant.h"
#include "transpose_operator.h"
#include "asm_gemm_a8w8.h"
#include <torch/extension.h>
#ifdef USE_CK_A8W8
#include "gemm_a8w8.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("topk_softmax", &topk_softmax,
            "Apply topk softmax to the gating outputs.");
      m.def("moe_align_block_size", &moe_align_block_size,
            "Aligning the number of tokens to be processed by each expert such "
            "that it is divisible by the block size.");
      m.def("silu_and_mul", &silu_and_mul, "Activation function used in SwiGLU.");
      m.def("rms_norm", &rms_norm, "Apply Root Mean Square (RMS) Normalization to the input tensor.");
      m.def("fused_add_rms_norm", &fused_add_rms_norm, "In-place fused Add and RMS Normalization");
      m.def("wvSpltK", &wvSpltK, "wvSpltK(Tensor in_a, Tensor in_b, Tensor! out_c, int N_in,"
                                 "        int CuCount) -> ()");
      m.def("LLMM1", &LLMM1, "LLMM1(Tensor in_a, Tensor in_b, Tensor! out_c, int rows_per_block) -> "
                             "()");
      m.def("rotary_embedding_fwd", &rotary_embedding, "rotary_embedding");
      m.def("batched_rotary_embedding", &batched_rotary_embedding, "batched_rotary_embedding");
      m.def("moe_sum", &moe_sum, "moe_sum(Tensor! input, Tensor output) -> ()");
      m.def("paged_attention_rocm", &paged_attention,
            "paged_attention_rocm(Tensor! out, Tensor exp_sums,"
            "                Tensor max_logits, Tensor tmp_out,"
            "                Tensor query, Tensor key_cache,"
            "                Tensor value_cache, int num_kv_heads,"
            "                float scale, Tensor block_tables,"
            "                Tensor context_lens, int block_size,"
            "                int max_context_len,"
            "                Tensor? alibi_slopes,"
            "                str kv_cache_dtype,"
            "                float k_scale, float v_scale) -> ()");
#ifdef USE_CK_A8W8
      m.def("gemm_a8w8", &gemm_a8w8, "gemm_a8w8", py::arg("XQ"), py::arg("WQ"),
            py::arg("x_scale"), py::arg("w_scale"), py::arg("Out"), 
            py::arg("bias") = std::nullopt, py::arg("splitK") = 0);
#endif
      m.def("swap_blocks", &swap_blocks,
            "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
      m.def("copy_blocks", &copy_blocks,
            "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "
            "Tensor block_mapping) -> ()");

      m.def("reshape_and_cache", &reshape_and_cache,
            "reshape_and_cache(Tensor key, Tensor value,"
            "                  Tensor! key_cache, Tensor! value_cache,"
            "                  Tensor slot_mapping,"
            "                  str kv_cache_dtype,"
            "                  float k_scale, float v_scale) -> ()");
      m.def("reshape_and_cache_flash", &reshape_and_cache_flash,
            "reshape_and_cache_flash(Tensor key, Tensor value,"
            "                        Tensor! key_cache,"
            "                        Tensor! value_cache,"
            "                        Tensor slot_mapping,"
            "                        str kv_cache_dtype,"
            "                        float k_scale, float v_scale) -> ()");
      m.def("reshape_and_cache_with_pertoken_quant", &reshape_and_cache_with_pertoken_quant,
            "reshape_and_cache_with_pertoken_quant(Tensor key, Tensor value,"
            "                  Tensor! key_cache, Tensor! value_cache,"
            "                  Tensor! k_dequant_scales, Tensor! v_dequant_scales,"
            "                  Tensor slot_mapping) -> ()");
      m.def("convert_fp8", &convert_fp8,
            "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
            "str kv_cache_dtype) -> ()");

      // Custom all-reduce kernels
      m.def("init_custom_ar", &init_custom_ar,
            "init_custom_ar(Tensor meta, Tensor rank_data, "
            "str[] handles, int[] offsets, int rank, "
            "bool full_nvlink) -> int");

      m.def("all_reduce_reg", &all_reduce_reg, "all_reduce_reg(int fa, Tensor inp, Tensor! out) -> ()");

      m.def("all_reduce_unreg", &all_reduce_unreg,
            "all_reduce_unreg(int fa, Tensor inp, Tensor reg_buffer, Tensor! out) -> "
            "()");

      m.def("dispose", &dispose);
      m.def("meta_size", &meta_size);

      m.def("register_buffer", &register_buffer,
            "register_buffer(int fa, Tensor t, str[] handles, "
            "int[] offsets) -> ()");

      m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
      m.def("register_graph_buffers", &register_graph_buffers);
#ifdef USE_ROCM
      m.def("allocate_meta_buffer", &allocate_meta_buffer);
      m.def("get_meta_buffer_ipc_handle", &get_meta_buffer_ipc_handle);
#endif

#if defined(FIND_CK)
      // ck staff start
      m.def("layernorm2d_fwd", &layernorm2d);
      m.def("layernorm2d_fwd_with_add", &layernorm2d_with_add);
      m.def("layernorm2d_fwd_with_smoothquant", &layernorm2d_with_smoothquant);
      m.def("layernorm2d_fwd_with_add_smoothquant", &layernorm2d_with_add_smoothquant);
      m.def("layernorm2d_fwd_with_dynamicquant", &layernorm2d_with_dynamicquant);
      m.def("layernorm2d_fwd_with_add_dynamicquant", &layernorm2d_with_add_dynamicquant);
      m.def("smoothquant_fwd", &smoothquant_fwd);
      m.def("moe_smoothquant_fwd", &moe_smoothquant_fwd);
      m.def("moe_sorting_fwd", &moe_sorting_fwd);
      m.def("pa_fwd_naive", &pa_fwd_naive, "pa_fwd_naive",
            py::arg("Q"),
            py::arg("K"),
            py::arg("V"),
            py::arg("block_tables"),
            py::arg("context_lens"),
            py::arg("k_dequant_scales"),
            py::arg("v_dequant_scales"),
            py::arg("max_seq_len"),
            py::arg("num_kv_heads"),
            py::arg("scale_s"),
            py::arg("scale_k"),
            py::arg("scale_v"),
            py::arg("block_size"),
            py::arg("quant_algo"),
            py::arg("out_") = std::nullopt);
      // ck staff end
#endif

      m.def("fmoe", &fmoe);
      m.def("fmoe_int8_g1u0", &fmoe_int8_g1u0);
      m.def("fmoe_int8_g1u0_a16", &fmoe_int8_g1u0_a16);
      m.def("transpose_add", &transpose_add, "apply for add with transpose.");
      m.def("transpose_mul", &transpose_mul, "apply for mul with transpose.");
      m.def("transpose_sub", &transpose_sub, "apply for sub with transpose.");
      m.def("transpose_div", &transpose_div, "apply for div with transpose.");
      m.def("pa_fwd_asm", &pa_fwd, "pa_fwd",
            py::arg("Q"),
            py::arg("K"),
            py::arg("V"),
            py::arg("block_tables"),
            py::arg("context_lens"),
            py::arg("max_num_blocks"),
            py::arg("K_QScale") = std::nullopt,
            py::arg("V_QScale") = std::nullopt,
            py::arg("out_") = std::nullopt);
      m.def("gemm_a8w8_asm", &gemm_a8w8_asm,
            "Asm gemm a8w8 ,  weight should be shuffle to layout(32,16)",
            py::arg("XQ"), py::arg("WQ"),
            py::arg("x_scale"), py::arg("w_scale"),
            py::arg("Out"), py::arg("bias"),
            py::arg("sub_m") = 128, py::arg("sub_n") = 128,
            py::arg("pad_a") = 0, py::arg("pad_b") = 0,
            py::arg("pad_c") = 0, py::arg("splitK") = 0);
      m.def("all_reduce_asm", &all_reduce_asm, "");

      m.def("reshape_and_cache_with_pertoken_quant", &reshape_and_cache_with_pertoken_quant,
            "reshape_and_cache_with_pertoken_quant(Tensor key, Tensor value,"
            "                        Tensor! key_cache,"
            "                        Tensor! value_cache,"
            "                        Tensor! k_dequant_scales,"
            "                        Tensor! v_dequant_scales,"
            "                        Tensor slot_mapping,"
            "                        str kv_cache_dtype) -> ()");
      m.def("convert_fp8", &convert_fp8,
            "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
            "str kv_cache_dtype) -> ()");
}
