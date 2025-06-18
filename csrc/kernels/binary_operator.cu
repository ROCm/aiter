#include "binary_operator.cuh"

torch::Tensor aiter_add(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::AddOp, false>(input, other);
}

torch::Tensor aiter_sub(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::SubOp, false>(input, other);
}

torch::Tensor aiter_mul(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::MulOp, false>(input, other);
}

torch::Tensor aiter_div(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::DivOp, false>(input, other);
}

// inp interface
torch::Tensor aiter_add_(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::AddOp, true>(input, other);
}

torch::Tensor aiter_sub_(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::SubOp, true>(input, other);
}

torch::Tensor aiter_mul_(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::MulOp, true>(input, other);
}

torch::Tensor aiter_div_(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::DivOp, true>(input, other);
}
