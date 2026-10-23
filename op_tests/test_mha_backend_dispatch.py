# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU routing tests; kernel numerics are covered by the GPU attention tests."""

from __future__ import annotations
import __future__

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

OPS = Path(__file__).resolve().parents[1] / "aiter" / "ops"
ENTRYPOINTS = ("flash_attn_func", "flash_attn_varlen_func")


class Tensor:
    """Shape-only tensor for executing the real autograd wrapper control flow."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.requires_grad = True

    def size(self, dim):
        return self.shape[dim]

    def __getitem__(self, index):
        last = index[-1]
        return Tensor((*self.shape[:-1], min(last.stop, self.shape[-1])))


class Context:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class AutogradFunction:
    @classmethod
    def apply(cls, *args):
        ctx = Context()
        result = cls.forward(ctx, *args)
        output = result[0] if isinstance(result, tuple) else result
        output.context = ctx
        ctx.function = cls
        ctx.input_count = len(args)
        return result


def _load_definitions(path, names, namespace):
    # Execute checkout definitions unchanged, injecting only hardware dependencies.
    # This avoids importing torch/aiter and does not alter sys.modules.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if {node.name for node in definitions} != set(names):
        raise AssertionError(f"Missing definitions in {path}: {names}")
    exec(  # noqa: S102 - only selected definitions from this checkout are executed
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


def _arguments(varlen, head_size=64):
    shape = (16, 4, head_size) if varlen else (2, 8, 4, head_size)
    args = {"q": Tensor(shape), "k": Tensor(shape), "v": Tensor(shape)}
    if varlen:
        args.update(
            cu_seqlens_q=object(),
            cu_seqlens_k=object(),
            max_seqlen_q=8,
            max_seqlen_k=8,
        )
    return args


def _triton_namespace():
    rng_state = object()
    torch = SimpleNamespace(
        autograd=SimpleNamespace(Function=AutogradFunction),
        is_grad_enabled=lambda: True,
        zeros_like=lambda tensor: Tensor(tensor.shape),
        empty_like=lambda tensor: Tensor(tensor.shape),
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                pad=lambda tensor, padding: Tensor(
                    (*tensor.shape[:-1], tensor.shape[-1] + padding[-1])
                )
            )
        ),
    )
    namespace = {
        "torch": torch,
        "_MHA_IMPL": "default",
        "_MHA_BACKENDS": ("triton", "gluon"),
        "_USE_FUSED_BWD_KERNEL": False,
        "_USE_INT64_STRIDES": True,
        "_LOGGER": Mock(),
        "_test_rng_state": rng_state,
        "_flash_attn_forward": Mock(
            side_effect=lambda q, *args, **kwargs: (
                Tensor(q.shape),
                object(),
                None,
                1,
                2,
                rng_state if kwargs["implementation"] == "dao_ai" else None,
            )
        ),
        "flash_attn_2": SimpleNamespace(bwd=Mock(), varlen_bwd=Mock()),
        "flash_attn_onekernel_backward": Mock(),
    }
    _load_definitions(
        OPS / "triton" / "attention" / "mha.py",
        (
            *ENTRYPOINTS,
            "_FlashAttnFunc",
            "_FlashAttnVarlenFunc",
            "_resolve_backend",
            "_resolve_implementation",
            "mha_set_impl",
            "_get_softmax_scale",
            "_get_sliding_window_size",
            "_pack_attn_returns",
        ),
        namespace,
    )
    return namespace


class TestTritonMhaAutograd(unittest.TestCase):
    def test_outstanding_forwards_retain_implementation_and_gradient_shape(self):
        for varlen, entrypoint in enumerate(ENTRYPOINTS):
            for head_size in (53, 64):
                with self.subTest(varlen=varlen, head_size=head_size):
                    namespace = _triton_namespace()
                    args = _arguments(varlen, head_size)
                    namespace["mha_set_impl"]("dao_ai")
                    dao = namespace[entrypoint](**args, deterministic=True)
                    default = namespace[entrypoint](**args, implementation="default")
                    self.assertEqual(namespace["_MHA_IMPL"], "dao_ai")
                    namespace["mha_set_impl"]("default")
                    for output, implementation in (
                        (dao, "dao_ai"),
                        (default, "default"),
                    ):
                        namespace["mha_set_impl"](
                            "default" if implementation == "dao_ai" else "dao_ai"
                        )
                        ctx = output.context
                        gradients = ctx.function.backward(ctx, Tensor(output.shape))
                        self.assertEqual(ctx.implementation, implementation)
                        self.assertIs(
                            ctx.rng_state,
                            (
                                namespace["_test_rng_state"]
                                if implementation == "dao_ai"
                                else None
                            ),
                        )
                        self.assertEqual(len(gradients), ctx.input_count)
                        for gradient, original in zip(gradients[:3], args.values()):
                            self.assertEqual(gradient.shape, original.shape)
                    calls = namespace["_flash_attn_forward"].call_args_list
                    self.assertEqual(
                        [call.kwargs["implementation"] for call in calls],
                        ["dao_ai", "default"],
                    )
                    dao_bwd = getattr(
                        namespace["flash_attn_2"], "varlen_bwd" if varlen else "bwd"
                    )
                    dao_bwd.assert_called_once()
                    self.assertTrue(dao_bwd.call_args.kwargs["deterministic"])
                    self.assertIs(
                        dao_bwd.call_args.kwargs["rng_state"],
                        namespace["_test_rng_state"],
                    )
                    namespace["flash_attn_onekernel_backward"].assert_called_once()

    def test_unknown_implementation_fails_before_forward(self):
        for varlen, entrypoint in enumerate(ENTRYPOINTS):
            namespace = _triton_namespace()
            with self.assertRaises(ValueError):
                namespace[entrypoint](**_arguments(varlen), implementation="unknown")
            namespace["_flash_attn_forward"].assert_not_called()
            with self.assertRaises(ValueError):
                namespace["mha_set_impl"]("unknown")
            self.assertEqual(namespace["_MHA_IMPL"], "default")

    def test_dao_varlen_rejects_ignored_buffer_arguments(self):
        for argument in ("block_table", "out"):
            with self.subTest(argument=argument):
                namespace = _triton_namespace()
                with self.assertRaises(NotImplementedError):
                    namespace["flash_attn_varlen_func"](
                        **_arguments(True),
                        implementation="dao_ai",
                        **{argument: object()},
                    )
                namespace["_flash_attn_forward"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
