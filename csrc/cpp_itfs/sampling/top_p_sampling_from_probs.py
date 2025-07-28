# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.


from jinja2 import Template
from csrc.cpp_itfs.utils import compile_template_op, AITER_CORE_DIR


MD_NAME = "top_p_sampling_from_probs"

with open(
    f"{AITER_CORE_DIR}/csrc/cpp_itfs/sampling/top_p_sampling_from_probs.cpp.jinja",
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


def top_p_sampling_from_probs(
    probs,
    indices,
    maybe_top_p_arr,
    top_p_val,
    deterministic: bool = False,
    generator=None,
):
    import torch
    from csrc.cpp_itfs.torch_utils import torch_to_c_types

    if generator is None:
        generator = torch.cuda.default_generators[probs.device.index]
    philox_offset = generator.get_offset()
    philox_seed = generator.seed()

    probs = probs.float()
    maybe_top_p_arr = maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
    top_p_val = float(top_p_val)

    batch_size = probs.size(0)
    vocab_size = probs.size(1)

    samples = torch.empty(batch_size, dtype=torch.int32, device=probs.device)
    func = compile(vocab_size, deterministic)
    (
        probs_ptr,
        samples_ptr,
        indices_ptr,
        top_p_arr_ptr,
        top_p_val,
        batch_size,
        philox_seed,
        philox_offset,
        stream,
    ) = torch_to_c_types(
        probs,
        samples,
        indices,
        maybe_top_p_arr,
        top_p_val,
        batch_size,
        philox_seed,
        philox_offset,
        torch.cuda.current_stream(),
    )
    func(
        probs_ptr,
        samples_ptr,
        indices_ptr,
        top_p_arr_ptr,
        batch_size,
        top_p_val,
        philox_seed,
        philox_offset,
        stream,
    )
    return samples
