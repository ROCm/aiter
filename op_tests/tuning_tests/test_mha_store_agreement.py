# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One store owns a winner, so every door into the kernel has to open on it.

The divergence these tests pin down was real: tiles measured into the runtime
CSV reached callers that went through ``aiter.ops.mha`` and were invisible to
callers that used the Triton kernel's own public entry point, which resolved
the untuned JSON feature default instead. Same kernel, same shape, same GPU,
different tiles depending on which door the caller came through.

These exercise the real lookup against a real CSV on disk rather than a
patched one, because the thing worth protecting is that both paths agree on
what the file says -- not that each of them calls a function.
"""

import csv
import os
import tempfile
import unittest
from unittest import mock

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.
import torch

from aiter.ops import mha
from aiter.ops.triton.attention import mha as triton_mha

from aiter.ops.mha_fwd_policy import MHA_FWD_RUNTIME_CSV_FIELDS

TILES = {
    "BLOCK_M": 64,
    "BLOCK_N": 32,
    "PRELOAD_V": False,
    "num_warps": 4,
    "waves_per_eu": 2,
    "num_stages": 1,
    "num_ctas": 1,
}
TILES_JSON = (
    '{"BLOCK_M":64,"BLOCK_N":32,"PRELOAD_V":false,'
    '"num_warps":4,"waves_per_eu":2,"num_stages":1,"num_ctas":1}'
)


def _config_file(path):
    """Point the runtime at one CSV.

    The real accessor is a read-only property that reroutes an environment
    override through the model_configs merge, which would rewrite the path
    out from under a test. Replacing the property leaves everything this
    module actually exercises -- the key, the table load, the backend match
    -- running for real.
    """
    return mock.patch.object(
        type(mha.AITER_CONFIGS),
        "AITER_CONFIG_MHA_FWD_FILE",
        property(lambda _self: path),
    )


class TestBothDoorsOpenOnTheSameStore(unittest.TestCase):
    MAX_SEQLEN_Q = 8
    MAX_SEQLEN_K = 16

    def _tensors(self):
        q = torch.empty((self.MAX_SEQLEN_Q, 12, 192), dtype=torch.bfloat16)
        k = torch.empty((self.MAX_SEQLEN_K, 12, 192), dtype=torch.bfloat16)
        v = torch.empty((self.MAX_SEQLEN_K, 12, 128), dtype=torch.bfloat16)
        cu_q = torch.tensor([0, self.MAX_SEQLEN_Q], dtype=torch.int32)
        cu_k = torch.tensor([0, self.MAX_SEQLEN_K], dtype=torch.int32)
        return q, k, v, cu_q, cu_k

    def _key_args(self, q, k, v):
        return {
            "mode": "varlen",
            "q": q,
            "k": k,
            "v": v,
            "batch": 1,
            "max_seqlen_q": self.MAX_SEQLEN_Q,
            "max_seqlen_k": self.MAX_SEQLEN_K,
            "min_seqlen_q": 0,
            "causal": False,
            "window_size": (-1, -1, 0),
            "dropout_p": 0.0,
            "logits_soft_cap": 0.0,
            "how_v3_bf16_cvt": 1,
            "return_lse": False,
            "return_attn_probs": False,
            "bias": None,
            "alibi_slopes": None,
            "sink_ptr": None,
            "block_table": None,
            "q_descale": None,
            "cu_seqlens_q_padded": None,
            "cu_seqlens_k_padded": None,
        }

    def _hardware(self):
        return (
            mock.patch.object(mha, "get_gfx_runtime", return_value="gfx950"),
            mock.patch.object(mha, "get_gpu_model", return_value="mi355x"),
            mock.patch.object(
                mha.torch.cuda,
                "get_device_properties",
                return_value=mock.Mock(multi_processor_count=256),
            ),
        )

    def _write_csv(self, directory, backend="triton", config_json=TILES_JSON):
        """A runtime CSV whose single row is keyed exactly to these tensors.

        The key is taken from the runtime's own key function rather than
        restated here, so the row cannot drift out of agreement with the
        lookup it is meant to be found by.
        """
        q, k, v, _, _ = self._tensors()
        gfx, model, props = self._hardware()
        with gfx, model, props:
            key = mha._mha_fwd_tuning_key(**self._key_args(q, k, v))
        row = dict(zip(mha.MHA_FWD_TUNING_KEY_FIELDS, key))
        row.update({"backend": backend, "num_splits": 0, "backend_config": config_json})
        path = os.path.join(directory, "tuned_mha_fwd.csv")
        with open(path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(MHA_FWD_RUNTIME_CSV_FIELDS))
            writer.writeheader()
            writer.writerow(row)
        return path

    def _config_through_router(self, path):
        """What the kernel receives when the caller goes through the router."""
        q, k, v, cu_q, cu_k = self._tensors()
        gfx, model, props = self._hardware()
        with (
            _config_file(path),
            gfx,
            model,
            props,
            mock.patch.object(
                triton_mha._FlashAttnVarlenFunc, "apply", return_value="ok"
            ) as apply,
        ):
            mha._load_mha_fwd_tuning_table.cache_clear()
            mha.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                self.MAX_SEQLEN_Q,
                self.MAX_SEQLEN_K,
            )
        return apply.call_args.args[-1]

    def _config_through_public_path(self, path):
        """What the kernel receives when the caller uses its own entry point."""
        q, k, v, cu_q, cu_k = self._tensors()
        gfx, model, props = self._hardware()
        with (
            _config_file(path),
            gfx,
            model,
            props,
            mock.patch.object(
                triton_mha._FlashAttnVarlenFunc, "apply", return_value="ok"
            ) as apply,
        ):
            mha._load_mha_fwd_tuning_table.cache_clear()
            triton_mha.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                self.MAX_SEQLEN_Q,
                self.MAX_SEQLEN_K,
                config=None,
                backend="triton",
            )
        return apply.call_args.args[-1]

    def test_the_public_path_launches_the_measured_tiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_csv(directory)
            self.assertEqual(self._config_through_public_path(path), TILES)

    def test_both_doors_resolve_the_same_tiles(self):
        """The regression itself: these two disagreed."""
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_csv(directory)
            self.assertEqual(
                self._config_through_public_path(path),
                self._config_through_router(path),
            )

    def test_an_explicit_config_still_wins_over_the_store(self):
        """Reading the store must not take the choice away from a caller who
        already made one."""
        chosen = dict(TILES, BLOCK_M=128)
        q, k, v, cu_q, cu_k = self._tensors()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_csv(directory)
            gfx, model, props = self._hardware()
            with (
                _config_file(path),
                gfx,
                model,
                props,
                mock.patch.object(
                    triton_mha._FlashAttnVarlenFunc, "apply", return_value="ok"
                ) as apply,
            ):
                mha._load_mha_fwd_tuning_table.cache_clear()
                triton_mha.flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_q,
                    cu_k,
                    self.MAX_SEQLEN_Q,
                    self.MAX_SEQLEN_K,
                    config=chosen,
                    backend="triton",
                )
            self.assertEqual(apply.call_args.args[-1], chosen)

    def test_a_shape_with_no_row_is_left_to_the_feature_default(self):
        """A miss must fall through rather than invent tiles, or the store
        would quietly narrow what the kernel supports."""
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_csv(directory)
            gfx, model, props = self._hardware()
            with gfx, model, props:
                key_args = self._key_args(*self._tensors()[:3])
                key_args["max_seqlen_k"] = 99999  # not the measured shape
                with _config_file(path):
                    mha._load_mha_fwd_tuning_table.cache_clear()
                    self.assertIsNone(
                        mha.lookup_mha_fwd_tile_config("triton", **key_args)
                    )

    def test_a_row_won_by_another_backend_is_not_borrowed(self):
        """Gluon tiles are not Triton tiles; a row the other backend won must
        not be handed over just because the shape matches."""
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_csv(
                directory,
                backend="gluon",
                config_json='{"BLOCK_M":64,"BLOCK_N":32,"num_warps":4,"waves_per_eu":2}',
            )
            gfx, model, props = self._hardware()
            with (
                _config_file(path),
                gfx,
                model,
                props,
            ):
                mha._load_mha_fwd_tuning_table.cache_clear()
                key_args = self._key_args(*self._tensors()[:3])
                self.assertIsNone(mha.lookup_mha_fwd_tile_config("triton", **key_args))
                self.assertIsNotNone(
                    mha.lookup_mha_fwd_tile_config("gluon", **key_args)
                )


if __name__ == "__main__":
    unittest.main()
