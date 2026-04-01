#pragma once
#if DISABLE_CK || defined(AITER_CK_FREE)
  #include "ck_tile_shim.h"
#else
  #include_next "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#endif
