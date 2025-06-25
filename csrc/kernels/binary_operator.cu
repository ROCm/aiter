#include "binary_operator.cuh"
#include "binary_op_api_common.hpp"

/*
#define DISPATCH_SECOND(pattern, Operation, _T0, scalar_type, cpp_type)                            \
  case scalar_type:                                                                                \
    binary_operation_process<pattern, Operation, _T0, cpp_type>(input, other, output, order_flag); \
    break

#define DISPATCH_FIRST(pattern, Operation, scalar_type, cpp_type)                    \
  case scalar_type:                                                                  \
    dispatch_second<pattern, Operation, cpp_type>(input, other, output, order_flag); \
    break

template <int pattern, typename Operation, typename _T0>
void dispatch_second(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, bool order_flag)
{
  printf("1111111111\n");
  switch (other.scalar_type())
  {
    // DISPATCH_SECOND(pattern, Operation, _T0, torch::kFloat32, float);
    // DISPATCH_SECOND(pattern, Operation, _T0, torch::kFloat64, double);
    // DISPATCH_SECOND(pattern, Operation, _T0, torch::kInt32, int);
    // DISPATCH_SECOND(pattern, Operation, _T0, torch::kInt64, long long);
    // DISPATCH_SECOND(pattern, Operation, _T0, torch::kBool, bool);
    // DISPATCH_SECOND(pattern, Operation, _T0, torch::kHalf, torch::Half);
    DISPATCH_SECOND(pattern, Operation, _T0, torch::kBFloat16, torch::BFloat16);
  default:
    break;
  }
}

template <int pattern, typename Operation>
void dispatch_first(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, bool order_flag)
{
  printf("2222222222\n");
  switch (input.scalar_type())
  {
    // DISPATCH_FIRST(pattern, Operation, torch::kFloat32, float);
    // DISPATCH_FIRST(pattern, Operation, torch::kFloat64, double);
    // DISPATCH_FIRST(pattern, Operation, torch::kInt32, int);
    // DISPATCH_FIRST(pattern, Operation, torch::kInt64, long long);
    // DISPATCH_FIRST(pattern, Operation, torch::kBool, bool);
    // DISPATCH_FIRST(pattern, Operation, torch::kHalf, torch::Half);
    DISPATCH_FIRST(pattern, Operation, torch::kBFloat16, torch::BFloat16);
  default:
    break;
  }
}

#undef DISPATCH_SECOND
#undef DISPATCH_FIRST

template <typename Operation, bool Inplace = false>
torch::Tensor binary_operation(torch::Tensor &input, torch::Tensor &other)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  int dim = input.dim();

  bool is_support = false;
  bool order_flag = true;
  int pattern = 0;
  constexpr uint32_t PATTERN_TRANSPOSE = 1;
  constexpr uint32_t PATTERN_BROADCAST_0 = 2;   // (m, n, k), (1, n, k)
  constexpr uint32_t PATTERN_BROADCAST_1 = 3;   // (m, n, k), (m, 1, k)
  constexpr uint32_t PATTERN_CONTIGUOUS = 4;
  constexpr uint32_t PATTERN_BROADCAST_2 = 5;   // (m, n, k), (m, n, 1)
  constexpr uint32_t PATTERN_BROADCAST_3 = 6;   // (m, n, k), (   n, 1)

  // contiguous case
  if (!is_support)
  {
    is_support = true;
    is_support &= (input.dim() == other.dim());
    is_support &= input.is_contiguous() == other.is_contiguous();
    is_support &= input.is_contiguous() == true;
    if (input.dim() == 1)
    {
      is_support &= input.numel() % 128 == 0;
    }
    for (int i = 0; i < input.dim() && is_support; ++i)
    {
      is_support &= (input.size(i) == other.size(i));
    }
    pattern = is_support ? PATTERN_CONTIGUOUS : 0;
  }

  if (!is_support && (dim == 3 || other.dim() == 3))
  {
    // transpose case
    if (input.is_contiguous() != other.is_contiguous())
    {
      auto tensor_not_conti = input.is_contiguous() ? other : input;
      order_flag = !input.is_contiguous() ? true : false;
      is_support = true;
      // avoid broadcast
      is_support &= input.dim() == other.dim();
      is_support &= input.size(0) == other.size(0);
      is_support &= input.size(1) == other.size(1);
      is_support &= input.size(2) == other.size(2);
      is_support &= tensor_not_conti.stride(1) == 1;
      pattern = is_support ? PATTERN_TRANSPOSE : 0;
    }
    // broadcast case
    else if (input.is_contiguous() && other.is_contiguous())
    {
      is_support = false;
      // input tensor dim and other tensor dim both equal to 3
      if (input.dim() == other.dim())
      {
        // broadcast at dim0 or dim1 or dim2
        auto broadcast_3d_case = [&] (int bcast_dim)
        {
          constexpr int bcast_pattern[3] = {PATTERN_BROADCAST_0, PATTERN_BROADCAST_1, PATTERN_BROADCAST_2};
          if (!is_support && (input.size(bcast_dim) == 1 || other.size(bcast_dim)) && input.size(bcast_dim) != other.size(bcast_dim))
          {
            is_support = true;
            for (int i = 0; i < 3; ++i)
            {
              if (bcast_dim != i) is_support &= input.size(i) == other.size(i);
            }
            is_support &= input.size(bcast_dim) == 1 ? other.size(bcast_dim) != 1 : true;
            pattern = is_support ? bcast_pattern[bcast_dim] : 0;
            order_flag = input.size(bcast_dim) != 1 ? true : false;
            if (bcast_dim == 1) order_flag = !order_flag;
          }
        };
        // (m, n, k), (1, n, k) or (1, n, k), (m, n, k)
        broadcast_3d_case(0);
        // (m, n, k), (m, 1, k) or (m, 1, k), (m, n, k)
        broadcast_3d_case(1);
        // (m, n, k), (m, n, 1) or (m, n, 1), (m, n, k)
        broadcast_3d_case(2);
      }
      // (m, n, k), (n, 1) or (n, 1), (m, n, k)
      else if (input.dim() == 2 || other.dim() == 2)
      {
        is_support = true;
        if (input.dim() == 2)
        {
          is_support &= input.size(0) == other.size(1);
          is_support &= input.size(1) == 1;
          pattern = is_support ? PATTERN_BROADCAST_3 : 0;
          order_flag = false;
        }
        else
        {
          is_support &= other.size(0) == input.size(1);
          is_support &= other.size(1) == 1;
          pattern = is_support ? PATTERN_BROADCAST_3 : 0;
          order_flag = true;
        }
      }
      // (m, n, k), (k) or (k), (m, n, k)
      // (m, n, k), (1) or (1), (m, n, k)
      else if (input.dim() == 1 || other.dim() == 1)
      {
        if (other.dim() == 1)
        {
          if (other.size(0) == 1 || (other.size(0) == input.size(2) && input.size(2) % (128 / input.element_size()) == 0))
          {
            if (input.numel() % (256 * 8 * 16 / input.element_size()) == 0)
            {
              is_support = true;
              pattern = PATTERN_BROADCAST_0;
              order_flag = true;
            }
          }
        }
        else
        {
          if (input.size(0) == 1 || (input.size(0) == other.size(2) && other.size(2) % (128 / other.element_size()) == 0))
          {
            if (other.numel() % (256 * 8 * 16 / other.element_size()) == 0)
            {
              is_support = true;
              pattern = PATTERN_BROADCAST_0;
              order_flag = false;
            }
          }
        }
      }
    }
  }

  if (!is_support && input.dim() != 3 && other.dim() != 3)
  {
    if (input.dim() == other.dim())
    {
      std::vector<int> bcast_dim_index = {};
      for (int i = 0; i < input.dim(); ++i)
      {
        // broadcast condition
        if (input.size(i) != other.size(i) && (input.size(i) == 1 || other.size(i) == 1))
        {
          bcast_dim_index.push_back(i);
        }
      }
      if (bcast_dim_index.size() == 1 && bcast_dim_index[0] != 0 && bcast_dim_index[0] != input.dim() - 1)
      {
        is_support = true;
        pattern = PATTERN_BROADCAST_1;
        order_flag = other.size(bcast_dim_index[0]) == 1 ? true : false;
      }
    }
  }

  // hip does not support double
  if (input.dtype() == torch::kDouble || other.dtype() == torch::kDouble)
  {
    is_support = false;
  }

  if (is_support)
  {
    auto in0_dtype = input.dtype();
    auto in1_dtype = other.dtype();
    torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
    std::vector<int64_t> out_shape = broadcastShapes(input, other);
    auto device = input.device();
    auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());

    torch::Tensor output;
    if constexpr(Inplace)
    {
      input = input.to(out_dtype);
      output = input;
    }
    else
    {
      output = torch::empty(out_shape, options);
    }

    switch (pattern)
    {
      case PATTERN_TRANSPOSE:
        dispatch_first<1, Operation>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_0:
        dispatch_first<2, Operation>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_1:
        dispatch_first<3, Operation>(input, other, output, order_flag);
        break;
      case PATTERN_CONTIGUOUS:
        dispatch_first<4, Operation>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_2:
        dispatch_first<5, Operation>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_3:
        dispatch_first<6, Operation>(input, other, output, order_flag);
        break;
      default:
        printf("[aiter/csrc/kernels/%s]: line %d break, unsupported type\n", __FILE__, __LINE__);
    }
    return output;
  }
  else
  {
    return aiter::aten_compute<Operation>(input, other);
  }
}
*/

torch::Tensor aiter_add(torch::Tensor &input, torch::Tensor &other)
{
  torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
  std::vector<int64_t> out_shape = broadcastShapes(input, other);
  auto device = input.device();
  auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());

  torch::Tensor output = torch::empty(out_shape, options);
  binary_op_dispatch("add", input, other, output);
  return output;
  // return binary_operation<aiter::AddOp, false>(input, other);
}

torch::Tensor aiter_sub(torch::Tensor &input, torch::Tensor &other)
{
  // return binary_operation<aiter::SubOp, false>(input, other);
  torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
  std::vector<int64_t> out_shape = broadcastShapes(input, other);
  auto device = input.device();
  auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());

  torch::Tensor output = torch::empty(out_shape, options);
  binary_op_dispatch("sub", input, other, output);
  return output;
}

torch::Tensor aiter_mul(torch::Tensor &input, torch::Tensor &other)
{
  // return binary_operation<aiter::MulOp, false>(input, other);
  torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
  std::vector<int64_t> out_shape = broadcastShapes(input, other);
  auto device = input.device();
  auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());

  torch::Tensor output = torch::empty(out_shape, options);
  binary_op_dispatch("mul", input, other, output);
  return output;
}

torch::Tensor aiter_div(torch::Tensor &input, torch::Tensor &other)
{
  // return binary_operation<aiter::DivOp, false>(input, other);
  torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
  std::vector<int64_t> out_shape = broadcastShapes(input, other);
  auto device = input.device();
  auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());

  torch::Tensor output = torch::empty(out_shape, options);
  binary_op_dispatch("div", input, other, output);
  return output;
}

// inp interface
torch::Tensor aiter_add_(torch::Tensor &input, torch::Tensor &other)
{
  // return binary_operation<aiter::AddOp, true>(input, other);
  torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
  std::vector<int64_t> out_shape = broadcastShapes(input, other);
  auto device = input.device();
  auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());
  input = input.to(out_dtype);
  torch::Tensor output = input;
  binary_op_dispatch("add", input, other, output);
  return output;
}

torch::Tensor aiter_sub_(torch::Tensor &input, torch::Tensor &other)
{
  // return binary_operation<aiter::SubOp, true>(input, other);
  torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
  std::vector<int64_t> out_shape = broadcastShapes(input, other);
  auto device = input.device();
  auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());
  input = input.to(out_dtype);
  torch::Tensor output = input;
  binary_op_dispatch("sub", input, other, output);
  return output;
}

torch::Tensor aiter_mul_(torch::Tensor &input, torch::Tensor &other)
{
  // return binary_operation<aiter::MulOp, true>(input, other);
  torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
  std::vector<int64_t> out_shape = broadcastShapes(input, other);
  auto device = input.device();
  auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());
  input = input.to(out_dtype);
  torch::Tensor output = input;
  binary_op_dispatch("mul", input, other, output);
  return output;
}

torch::Tensor aiter_div_(torch::Tensor &input, torch::Tensor &other)
{
  // return binary_operation<aiter::DivOp, true>(input, other);
  torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
  std::vector<int64_t> out_shape = broadcastShapes(input, other);
  auto device = input.device();
  auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());
  input = input.to(out_dtype);
  torch::Tensor output = input;
  binary_op_dispatch("div", input, other, output);
  return output;
}
