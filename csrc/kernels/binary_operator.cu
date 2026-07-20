#include "binary_operator.cuh"
#include "binary_op_api_common.hpp"

bool aiter_add(aiter_tensor_t &input, aiter_tensor_t &other, aiter_tensor_t &output)
{
  return binary_op_dispatch("add", input, other, output);
}

bool aiter_sub(aiter_tensor_t &input, aiter_tensor_t &other, aiter_tensor_t &output)
{
  return binary_op_dispatch("sub", input, other, output);
}

bool aiter_mul(aiter_tensor_t &input, aiter_tensor_t &other, aiter_tensor_t &output)
{
  return binary_op_dispatch("mul", input, other, output);
}

bool aiter_div(aiter_tensor_t &input, aiter_tensor_t &other, aiter_tensor_t &output)
{
  return binary_op_dispatch("div", input, other, output);
}

bool aiter_add_(aiter_tensor_t &input, aiter_tensor_t &other)
{
  return binary_op_dispatch("add", input, other, input);
}

bool aiter_sub_(aiter_tensor_t &input, aiter_tensor_t &other)
{
  return binary_op_dispatch("sub", input, other, input);
}

bool aiter_mul_(aiter_tensor_t &input, aiter_tensor_t &other)
{
  return binary_op_dispatch("mul", input, other, input);
}

bool aiter_div_(aiter_tensor_t &input, aiter_tensor_t &other)
{
  return binary_op_dispatch("div", input, other, input);
}
