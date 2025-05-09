import torch


# TODO: this should support e5m2 as well
def quantize_fp8(tensor: torch.Tensor, dim=() ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantize_dim = [i for i in range(tensor.dim()) if i not in dim]
    max_vals = tensor.abs().amax(dim=quantize_dim, keepdim=True)
    max_repr_val = torch.finfo(torch.float8_e4m3fnuz).max
    max_vals[max_vals == 0] = 1e-8 # Avoid division by zero

    # Compute scale factors for each channel
    scale: torch.Tensor = max_repr_val / max_vals.to(torch.float32)

    # Quantize the tensor
    tensor = tensor * scale
    tensor.clamp_(-max_repr_val, max_repr_val)
    tensor_quantized = tensor.to(torch.float8_e4m3fnuz)

    scale = scale.squeeze(dim=quantize_dim)

    return tensor_quantized, scale, 1 / scale


def quantize_int8(tensor: torch.Tensor, dim=() ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantize_dim = [i for i in range(tensor.dim()) if i not in dim]
    max_vals = tensor.abs().amax(dim=quantize_dim, keepdim=True)
    max_repr_val = torch.iinfo(torch.int8).max
    max_vals[max_vals == 0] = 1e-8 # Avoid division by zero
    # Compute scale factors for each channel
    scale: torch.Tensor = max_repr_val / max_vals.to(torch.float32)
    # Quantize the tensor
    tensor = tensor * scale
    tensor.clamp_(-max_repr_val, max_repr_val)
    tensor = tensor.round_()
    tensor_quantized = tensor.to(torch.int8)

    scale = scale.squeeze(dim=quantize_dim)

    return tensor_quantized, scale, 1 / scale


def quantize_int4(tensor: torch.Tensor, group_size: int, has_zp: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Reshape tensor
    k, n = tensor.shape
    tensor = tensor.reshape(-1, group_size, n)
    tensor = tensor.permute(1, 0, 2)

    max_val = torch.max(tensor, 0, keepdim=True).values
    min_val = torch.min(tensor, 0, keepdim=True).values

    # Asymmetric quantization
    zp = None
    if has_zp:
        max_q_val = 15
        min_q_val = 0  #Min maps to 0
        scale = (max_val - min_val).clamp(min=1e-5) / (max_q_val)
        zp = torch.round(torch.abs(min_val / scale)).clamp(min_q_val, max_q_val).int()
    # Symmetric quantization
    else:
        max_q_val = 7
        min_q_val = -7
        scale = max_val / max_q_val
    # Quantize and clamp
    tensor_q = torch.round(tensor / scale).int() + (zp if has_zp else 0)
    tensor_q = torch.clamp(tensor, min_q_val, max_q_val)
    # Restore shapes
    tensor_q = tensor_q.reshape((group_size, - 1, n))
    tensor_q = tensor_q.permute(1, 0 , 2)
    tensor_q = tensor_q.reshape((k, n)).contiguous()
    # Pack scale
    scale = scale.reshape((-1, n)).contiguous()
    # Set zp if it's not None
    if zp is not None:
        zp = zp.reshape((-1, n)).contiguous()
        zp = zp.to(device=tensor.device)

    return tensor_q, scale, zp