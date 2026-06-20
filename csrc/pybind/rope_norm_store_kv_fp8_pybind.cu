#include <torch/extension.h>

void rope_norm_store_kv_fp8_fused_hip(torch::Tensor qkv,
                                             torch::Tensor cos_sin,
                                             torch::Tensor q_index,
                                             torch::Tensor num_seqlen_per_req,
                                             torch::Tensor kvcache_indices,
                                             torch::Tensor q_norm_weight,
                                             torch::Tensor k_norm_weight,
                                             torch::Tensor hadamard,
                                             torch::Tensor k_scale,
                                             torch::Tensor v_scale,
                                             torch::Tensor out_q,
                                             torch::Tensor key_cache,
                                             torch::Tensor value_cache,
                                             torch::Tensor q_scale_out,
                                             double eps,
                                             double fp8_max,
                                             bool assume_decode_one_token,
                                             int64_t tile_hpw);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("rope_norm_store_kv_fp8_fused_hip",
        &rope_norm_store_kv_fp8_fused_hip,
        "Fused HIP rope+norm+kv-store kernel for rope_norm_store_kv_fp8",
        py::arg("qkv"), py::arg("cos_sin"), py::arg("q_index"),
        py::arg("num_seqlen_per_req"), py::arg("kvcache_indices"),
        py::arg("q_norm_weight"), py::arg("k_norm_weight"), py::arg("hadamard"),
        py::arg("k_scale"), py::arg("v_scale"), py::arg("out_q"),
        py::arg("key_cache"), py::arg("value_cache"), py::arg("q_scale_out"),
        py::arg("eps"), py::arg("fp8_max"), py::arg("assume_decode_one_token"),
        py::arg("tile_hpw") = 1);
}
