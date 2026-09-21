#include <opus/hip_minimal.hpp>
#include "defs.h"

#ifndef __HIP_DEVICE_COMPILE__
__global__ void get_mla_metadata_kernel(opus_mla_decode_metadata_kargs kargs) {}
#else
#include "get_mla_metadata.hpp"
#endif
