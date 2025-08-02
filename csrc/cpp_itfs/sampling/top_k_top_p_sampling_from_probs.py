# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.


from jinja2 import Template
from csrc.cpp_itfs.utils import compile_template_op, AITER_CORE_DIR


MD_NAME = "top_k_top_p_sampling_from_probs"

with open(
    f"{AITER_CORE_DIR}/csrc/cpp_itfs/sampling/top_k_top_p_sampling_from_probs.cpp.jinja",
    "r",
) as f:
    src_template = Template(f.read())


def compile(
    d: int,
    deterministic: bool,
    folder: str = None,
):
    return compile_template_op(
        src_template,
        MD_NAME,
        [
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/utils.h",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/sampling/sampling.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/sampling/vec_dtypes.cuh",
        ],
        d=d,
        deterministic=deterministic,
        folder=folder,
    )


def top_k_top_p_sampling_from_probs(
    probs,
    indices,
    maybe_top_k_arr,
    top_k_val,
    maybe_top_p_arr,
    top_p_val,
    deterministic=False,
    generator=None,
):
    import torch
    from csrc.cpp_itfs.torch_utils import torch_to_c_types

    if generator is None:
        generator = torch.cuda.default_generators[probs.device.index]
    probs = probs.float()
    top_p_val = float(top_p_val)
    top_k_val = int(top_k_val)
    maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
    maybe_top_p_arr = maybe_top_p_arr.float() if maybe_top_p_arr is not None else None

    batch_size = indices.size(0) if indices is not None else probs.size(0)
    vocab_size = probs.size(1)
    philox_offset = generator.get_offset()
    philox_seed = generator.seed()

    output = torch.empty(batch_size, dtype=torch.int32, device=probs.device)

    func = compile(vocab_size, deterministic)
    (
        probs_ptr,
        output_ptr,
        indices_ptr,
        top_k_arr_ptr,
        top_p_arr_ptr,
        top_k_val,
        top_p_val,
        batch_size,
        philox_seed,
        philox_offset,
        stream,
    ) = torch_to_c_types(
        probs,
        output,
        indices,
        maybe_top_k_arr,
        maybe_top_p_arr,
        top_k_val,
        top_p_val,
        batch_size,
        philox_seed,
        philox_offset,
        torch.cuda.current_stream(),
    )
    func(
        probs_ptr,
        output_ptr,
        indices_ptr,
        top_k_arr_ptr,
        top_p_arr_ptr,
        batch_size,
        top_k_val,
        top_p_val,
        philox_seed,
        philox_offset,
        stream,
    )
    return output
