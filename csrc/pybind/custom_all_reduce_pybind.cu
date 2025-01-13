#include "custom_all_reduce.h"
#include "communication_asm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("init_custom_ar", &init_custom_ar,
            "init_custom_ar(Tensor meta, Tensor rank_data, "
            "str[] handles, int[] offsets, int rank, "
            "bool full_nvlink) -> int");

      m.def("all_reduce_reg", &all_reduce_reg, "all_reduce_reg(int fa, Tensor inp, Tensor! out) -> ()");

      m.def("all_reduce_unreg", &all_reduce_unreg,
            "all_reduce_unreg(int fa, Tensor inp, Tensor reg_buffer, Tensor! out) -> "
            "()");
      m.def("all_reduce_asm_", &all_reduce_asm, "");
      m.def("all_reduce_rmsnorm_", &all_reduce_rmsnorm, "all_reduce_rmsnorm");

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
}