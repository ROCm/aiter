#include "speculative_sampling.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("tree_speculative_sampling_target_only", &tree_speculative_sampling_target_only);
}
