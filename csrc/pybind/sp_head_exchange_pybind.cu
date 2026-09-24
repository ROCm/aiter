#include "rocm_ops.hpp"
#include "aiter_stream.h"
namespace aiter {
void sp_head_exchange(int64_t handle, const aiter_tensor_t& input,
    const aiter_tensor_t& output, int64_t registered_buffer,
    int64_t registered_bytes, bool stage, int64_t blocks);
}
PYBIND11_MODULE(AITER_EXTENSION_NAME, m) {
    AITER_SET_STREAM_PYBIND;
    m.def("sp_head_exchange", &aiter::sp_head_exchange);
}
