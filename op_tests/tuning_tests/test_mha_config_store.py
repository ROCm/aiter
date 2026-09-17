# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for MHA config persistence: the writer, the tier walk, and
the two candidate storage contracts behind the store seam."""

import collections
import json
import os
import tempfile
import unittest
from unittest import mock

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.

from aiter.ops.mha_fwd_policy import (
    MHA_FWD_TILE_CONFIG_BACKENDS,
    MHA_FWD_TILE_CONFIG_KEYS,
    MhaFwdPlan,
    MhaFwdProblem,
    enumerate_mha_fwd_candidates,
    fold_mha_fwd_selection_records,
    mha_fwd_config_matches,
)
from aiter.ops.mha_fwd_store import get_mha_fwd_store
from aiter.ops.triton.utils import attention_config_utils as attn_cfg
from aiter.ops.triton.utils.config_utils import (
    HARDWARE_ANY,
    format_hardware_key,
    load_config_json,
)
from aiter.ops.triton.utils.config_writer import (
    update_config_entry,
    validate_config_entry,
    write_config_json,
)

_TRITON_KEYS = tuple(sorted(MHA_FWD_TILE_CONFIG_KEYS["triton"]))
_SHAPE = dict(
    mode="varlen",
    hdim_q=192,
    hdim_v=128,
    nhead_q=12,
    nhead_k=12,
    dtype="bfloat16",
    causal=False,
    max_seqlen_q=4096,
    max_seqlen_k=42700,
)


def _problem_row(**overrides):
    row = {
        "gfx": "gfx942",
        "gpu_model": "mi325x",
        "cu_num": 304,
        "mode": "varlen",
        "batch": 1,
        "total_q": 4096,
        "total_k": 42700,
        "max_seqlen_q": 4096,
        "max_seqlen_k": 42700,
        "min_seqlen_q": 0,
        "nhead_q": 12,
        "nhead_k": 12,
        "hdim_q": 192,
        "hdim_v": 128,
        "dtype": "bfloat16",
        "causal": 0,
        "window_left": -1,
        "window_right": -1,
        "sink_size": 0,
        "dropout_p": 0.0,
        "logits_soft_cap": 0.0,
        "how_v3_bf16_cvt": 1,
        "return_lse": 0,
        "return_attn_probs": 0,
        "has_bias": 0,
        "has_alibi": 0,
        "has_sink": 0,
        "has_block_table": 0,
        "has_q_descale": 0,
        "has_physical_padding": 0,
        "is_grad": 0,
    }
    row.update(overrides)
    return row


class TestConfigWriter(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "DEFAULT.json")

    def test_write_is_atomic_and_leaves_no_temporary(self):
        write_config_json(self.path, {"fwd": {"default": {"BLOCK_M": 128}}})
        self.assertEqual(os.listdir(self.dir), ["DEFAULT.json"])
        with open(self.path, encoding="utf-8") as file:
            text = file.read()
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(json.loads(text)["fwd"]["default"]["BLOCK_M"], 128)

    def test_failed_write_leaves_the_previous_file_intact(self):
        write_config_json(self.path, {"fwd": {"default": {"BLOCK_M": 128}}})
        with self.assertRaises(ValueError):
            write_config_json(self.path, {"fwd": {"default": float("nan")}})
        self.assertEqual(os.listdir(self.dir), ["DEFAULT.json"])
        self.assertEqual(load_config_json(self.path)["fwd"]["default"]["BLOCK_M"], 128)

    def test_generated_container_stays_sorted_whatever_the_measurement_order(self):
        write_config_json(self.path, {"fwd": {"default": {"BLOCK_M": 128}}})
        for name in ("zzz", "aaa", "mmm"):
            update_config_entry(
                self.path, ("fwd", "shapes", "any", name), {"BLOCK_M": 1}
            )
        shapes = load_config_json(self.path)["fwd"]["shapes"]["any"]
        self.assertEqual(list(shapes), ["aaa", "mmm", "zzz"])

    def test_hand_authored_order_survives_an_update(self):
        write_config_json(
            self.path,
            {
                "fwd": {
                    "default": {"BLOCK_M": 1},
                    "pe": {"BLOCK_M": 2},
                    "fp8": {"BLOCK_M": 3},
                }
            },
        )
        update_config_entry(self.path, ("fwd", "default"), {"BLOCK_M": 99})
        fwd = load_config_json(self.path)["fwd"]
        self.assertEqual(list(fwd), ["default", "pe", "fp8"])
        self.assertEqual(fwd["default"]["BLOCK_M"], 99)

    def test_update_does_not_disturb_siblings(self):
        write_config_json(
            self.path, {"fwd": {"default": {"BLOCK_M": 1}}, "bkwd": {"x": 1}}
        )
        update_config_entry(self.path, ("fwd", "shapes", "any", "s"), {"BLOCK_M": 2})
        document = load_config_json(self.path)
        self.assertEqual(document["bkwd"], {"x": 1})
        self.assertEqual(document["fwd"]["default"], {"BLOCK_M": 1})

    def test_a_fresh_write_is_visible_immediately(self):
        write_config_json(self.path, {"fwd": {"default": {"BLOCK_M": 1}}})
        self.assertEqual(load_config_json(self.path)["fwd"]["default"]["BLOCK_M"], 1)
        update_config_entry(self.path, ("fwd", "default"), {"BLOCK_M": 2})
        self.assertEqual(load_config_json(self.path)["fwd"]["default"]["BLOCK_M"], 2)

    def test_a_missing_file_is_cached_negatively_then_read_after_writing(self):
        self.assertIsNone(load_config_json(self.path, required=False))
        update_config_entry(self.path, ("fwd", "default"), {"BLOCK_M": 7})
        self.assertEqual(load_config_json(self.path)["fwd"]["default"]["BLOCK_M"], 7)

    def test_validation_rejects_a_key_no_kernel_reads(self):
        with self.assertRaisesRegex(ValueError, "no kernel reads"):
            validate_config_entry({"BLOCK_MM": 128}, required=(), optional=_TRITON_KEYS)

    def test_validation_rejects_a_missing_required_key(self):
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            validate_config_entry(
                {"BLOCK_N": 64}, required=("BLOCK_M",), optional=_TRITON_KEYS
            )


class TestHardwareKey(unittest.TestCase):
    def test_same_arch_and_cu_count_still_separates_two_skus(self):
        self.assertNotEqual(
            format_hardware_key(304, "MI300X"), format_hardware_key(304, "MI325X")
        )

    def test_key_is_case_and_space_insensitive(self):
        self.assertEqual(
            format_hardware_key(304, "MI325X"), format_hardware_key(304, "mi325x")
        )

    def test_an_unidentifiable_gpu_is_rejected_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            format_hardware_key(304, "unknown")


class TestShapeKey(unittest.TestCase):
    def test_nearby_context_lengths_share_a_bucket(self):
        self.assertEqual(
            attn_cfg.format_mha_shape_key(**{**_SHAPE, "max_seqlen_k": 42700}),
            attn_cfg.format_mha_shape_key(**{**_SHAPE, "max_seqlen_k": 65536}),
        )

    def test_a_different_order_of_magnitude_does_not(self):
        self.assertNotEqual(
            attn_cfg.format_mha_shape_key(**{**_SHAPE, "max_seqlen_k": 4096}),
            attn_cfg.format_mha_shape_key(**{**_SHAPE, "max_seqlen_k": 65536}),
        )

    def test_torch_dtype_spelling_is_normalized(self):
        self.assertEqual(
            attn_cfg.format_mha_shape_key(**{**_SHAPE, "dtype": "torch.bfloat16"}),
            attn_cfg.format_mha_shape_key(**{**_SHAPE, "dtype": "bfloat16"}),
        )

    def test_causal_and_non_causal_are_tuned_separately(self):
        self.assertNotEqual(
            attn_cfg.format_mha_shape_key(**{**_SHAPE, "causal": True}),
            attn_cfg.format_mha_shape_key(**{**_SHAPE, "causal": False}),
        )


class TestTierWalk(unittest.TestCase):
    """The walk is exercised against a temporary config tree, so these tests
    say nothing about which tiles are shipped -- only which tier wins."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.relpath = "gfx942/triton/attention/mha/DEFAULT.json"
        self.path = os.path.join(self.root, self.relpath)
        self.shape_key = attn_cfg.format_mha_shape_key(**_SHAPE)
        self.gpu_key = format_hardware_key(304, "mi325x")
        patch = mock.patch.object(
            attn_cfg, "mha_config_relpath", return_value=self.relpath
        )
        patch.start()
        self.addCleanup(patch.stop)
        roots = mock.patch(
            "aiter.ops.triton.utils.config_utils.config_roots",
            return_value=(self.root,),
        )
        roots.start()
        self.addCleanup(roots.stop)
        self.addCleanup(attn_cfg.current_mha_hardware_key.cache_clear)
        attn_cfg.current_mha_hardware_key.cache_clear()

    def _write(self, fwd):
        write_config_json(self.path, {"fwd": fwd})

    def _get(self, hardware):
        with mock.patch.object(
            attn_cfg, "current_mha_hardware_key", return_value=hardware
        ):
            return attn_cfg.get_mha_config("triton", "default", self.shape_key)

    def test_a_file_without_a_shapes_table_behaves_exactly_as_before(self):
        self._write({"default": {"BLOCK_M": 128}})
        self.assertEqual(self._get(self.gpu_key), {"BLOCK_M": 128})

    def test_a_measured_entry_for_this_gpu_beats_the_feature_default(self):
        self._write(
            {
                "default": {"BLOCK_M": 128},
                "shapes": {self.gpu_key: {self.shape_key: {"BLOCK_M": 64}}},
            }
        )
        self.assertEqual(self._get(self.gpu_key), {"BLOCK_M": 64})

    def test_another_gpus_measurement_is_not_borrowed(self):
        other = format_hardware_key(304, "mi300x")
        self._write(
            {
                "default": {"BLOCK_M": 128},
                "shapes": {other: {self.shape_key: {"BLOCK_M": 64}}},
            }
        )
        self.assertEqual(self._get(self.gpu_key), {"BLOCK_M": 128})

    def test_this_gpu_wins_over_an_entry_published_without_a_sku(self):
        self._write(
            {
                "default": {"BLOCK_M": 128},
                "shapes": {
                    HARDWARE_ANY: {self.shape_key: {"BLOCK_M": 32}},
                    self.gpu_key: {self.shape_key: {"BLOCK_M": 64}},
                },
            }
        )
        self.assertEqual(self._get(self.gpu_key), {"BLOCK_M": 64})

    def test_an_unidentified_gpu_still_reads_the_sku_agnostic_tier(self):
        self._write(
            {
                "default": {"BLOCK_M": 128},
                "shapes": {HARDWARE_ANY: {self.shape_key: {"BLOCK_M": 32}}},
            }
        )
        self.assertEqual(self._get(HARDWARE_ANY), {"BLOCK_M": 32})

    def test_an_untuned_shape_falls_back_to_the_feature_default(self):
        self._write(
            {
                "default": {"BLOCK_M": 128},
                "shapes": {self.gpu_key: {"mode=varlen,other": {"BLOCK_M": 64}}},
            }
        )
        self.assertEqual(self._get(self.gpu_key), {"BLOCK_M": 128})

    def test_the_returned_config_is_safe_to_mutate(self):
        self._write({"default": {"BLOCK_M": 128}})
        self._get(self.gpu_key)["BLOCK_M"] = 999
        self.assertEqual(self._get(self.gpu_key), {"BLOCK_M": 128})


class TestConfigStores(unittest.TestCase):
    def setUp(self):
        self.problem = MhaFwdProblem.from_mapping(_problem_row())
        self.config = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4}

    def test_an_unknown_store_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown MHA config store"):
            get_mha_fwd_store("sqlite")

    def test_the_default_is_the_contract_the_runtime_csv_already_ships(self):
        self.assertEqual(get_mha_fwd_store().name, "csv")

    def test_the_csv_store_puts_the_tiles_in_the_column(self):
        cell = get_mha_fwd_store("csv").publish(self.problem, "triton", self.config)
        self.assertEqual(json.loads(cell), self.config)

    def test_a_name_is_config_backend_carries_an_empty_cell(self):
        for backend in ("asm_v3", "ck", "flydsl", "opus"):
            with self.subTest(backend=backend):
                self.assertEqual(
                    get_mha_fwd_store("csv").publish(self.problem, backend, None), ""
                )

    def test_the_json_store_writes_the_tree_and_empties_the_column(self):
        root = tempfile.mkdtemp()
        with mock.patch(
            "aiter.ops.triton.utils.config_utils.AITER_TRITON_CONFIGS_PATH", root
        ):
            cell = get_mha_fwd_store("json").publish(
                self.problem, "triton", self.config
            )
        self.assertEqual(cell, "")
        path = os.path.join(root, "gfx942/triton/attention/mha/DEFAULT.json")
        shapes = load_config_json(path)["fwd"]["shapes"]
        entry = shapes[format_hardware_key(304, "mi325x")]
        self.assertEqual(list(entry.values())[0], self.config)

    def test_the_json_store_keys_on_the_measured_gpu_not_the_running_one(self):
        root = tempfile.mkdtemp()
        problem = MhaFwdProblem.from_mapping(_problem_row(gpu_model="mi300x"))
        with mock.patch(
            "aiter.ops.triton.utils.config_utils.AITER_TRITON_CONFIGS_PATH", root
        ):
            get_mha_fwd_store("json").publish(problem, "triton", self.config)
        path = os.path.join(root, "gfx942/triton/attention/mha/DEFAULT.json")
        self.assertEqual(
            list(load_config_json(path)["fwd"]["shapes"]),
            [format_hardware_key(304, "mi300x")],
        )

    def test_the_json_store_refuses_a_typo_rather_than_publishing_it(self):
        root = tempfile.mkdtemp()
        with mock.patch(
            "aiter.ops.triton.utils.config_utils.AITER_TRITON_CONFIGS_PATH", root
        ):
            with self.assertRaisesRegex(ValueError, "no kernel reads"):
                get_mha_fwd_store("json").publish(
                    self.problem, "triton", {"BLOCK_MM": 1}
                )

    def test_the_json_store_refuses_a_tile_backend_winner_with_no_config(self):
        with self.assertRaisesRegex(ValueError, "no launch configuration"):
            get_mha_fwd_store("json").publish(self.problem, "triton", None)


class TestWinnerRoundTrip(unittest.TestCase):
    """The question the whole exercise turns on: after a winner is published,
    does the kernel's own loader hand that winner back on the next call?"""

    def setUp(self):
        from aiter.ops.triton.utils._triton import arch_info

        self.arch = arch_info.get_arch()
        self.root = tempfile.mkdtemp()
        self.config = {"BLOCK_M": 256, "BLOCK_N": 32, "num_warps": 8}
        # The store keys on the measured GPU, so the round trip only closes if
        # the row describes the GPU this process is running on.
        self.gpu_model = "mi325x"
        self.cu_num = 304
        self.problem = MhaFwdProblem.from_mapping(
            _problem_row(gfx=self.arch, gpu_model=self.gpu_model, cu_num=self.cu_num)
        )
        self.addCleanup(attn_cfg.current_mha_hardware_key.cache_clear)
        attn_cfg.current_mha_hardware_key.cache_clear()

    def _staged(self):
        """Publish to a scratch overlay so the shipped tree is never written.

        This is also the deployment shape of the JSON contract: the overlay is
        the highest-priority root, so it is both where a winner is published
        and the first place the loader looks.
        """
        return (
            mock.patch(
                "aiter.ops.triton.utils.config_utils.AITER_TRITON_CONFIGS_OVERLAY_PATH",
                (self.root,),
            ),
            mock.patch.object(
                attn_cfg,
                "current_mha_hardware_key",
                return_value=format_hardware_key(self.cu_num, self.gpu_model),
            ),
        )

    def test_a_published_winner_is_what_the_kernel_loader_returns(self):
        import torch

        from aiter.ops.triton._triton_kernels.attention import mha as triton_mha

        shape_key = attn_cfg.format_mha_shape_key(**_SHAPE)
        overlay, hardware = self._staged()
        with overlay, hardware:
            untuned = triton_mha._get_config(
                False, torch.bfloat16, head_dim_v=128, shape_key=shape_key
            )
            self.assertNotEqual(untuned, self.config)

            get_mha_fwd_store("json").publish(self.problem, "triton", self.config)
            triton_mha._get_config.cache_clear()

            tuned = triton_mha._get_config(
                False, torch.bfloat16, head_dim_v=128, shape_key=shape_key
            )
        self.assertEqual(tuned, self.config)

    def test_an_unmeasured_shape_is_unaffected_by_the_publication(self):
        import torch

        from aiter.ops.triton._triton_kernels.attention import mha as triton_mha

        other_key = attn_cfg.format_mha_shape_key(**{**_SHAPE, "nhead_q": 64})
        overlay, hardware = self._staged()
        with overlay, hardware:
            baseline = triton_mha._get_config(
                False, torch.bfloat16, head_dim_v=128, shape_key=other_key
            )
            get_mha_fwd_store("json").publish(self.problem, "triton", self.config)
            triton_mha._get_config.cache_clear()
            after = triton_mha._get_config(
                False, torch.bfloat16, head_dim_v=128, shape_key=other_key
            )
        self.assertEqual(after, baseline)


class TestTileConfigVocabulary(unittest.TestCase):
    """A loaded backend_config is the only free-form field in the runtime CSV,
    so it is the one field a closed vocabulary has to be enforced on."""

    def test_the_vocabulary_matches_what_enumeration_actually_emits(self):
        for gfx in ("gfx942", "gfx950", "gfx1250"):
            emitted = collections.defaultdict(set)
            for candidate in enumerate_mha_fwd_candidates(gfx):
                if candidate.backend_config:
                    emitted[candidate.backend].update(candidate.backend_config)
            for backend, keys in emitted.items():
                with self.subTest(gfx=gfx, backend=backend):
                    self.assertEqual(keys, set(MHA_FWD_TILE_CONFIG_KEYS[backend]))

    def test_a_misspelled_key_is_rejected_at_load_rather_than_at_launch(self):
        with self.assertRaisesRegex(ValueError, "never chooses"):
            MhaFwdPlan(backend="triton", backend_config={"BLOCK_MM": 128})

    def test_a_gluon_plan_rejects_a_triton_only_key(self):
        with self.assertRaisesRegex(ValueError, "never chooses"):
            MhaFwdPlan(backend="gluon", backend_config={"num_stages": 2})

    def test_a_legal_config_still_loads(self):
        plan = MhaFwdPlan(backend="triton", backend_config={"BLOCK_M": 128})
        self.assertEqual(plan.backend_config, {"BLOCK_M": 128})

    def test_a_name_is_config_backend_still_refuses_any_config(self):
        with self.assertRaisesRegex(ValueError, "does not accept backend_config"):
            MhaFwdPlan(backend="opus", backend_config={"BLOCK_M": 128})


class TestSmokeStrategy(unittest.TestCase):
    def test_smoke_is_a_strict_subset_of_the_exhaustive_catalogue(self):
        for gfx in ("gfx942", "gfx950", "gfx1250"):
            with self.subTest(gfx=gfx):
                full = {c.identity for c in enumerate_mha_fwd_candidates(gfx)}
                smoke = {c.identity for c in enumerate_mha_fwd_candidates(gfx, "smoke")}
                self.assertTrue(smoke <= full)

    def test_smoke_keeps_every_name_is_config_backend(self):
        full = collections.Counter(
            c.backend for c in enumerate_mha_fwd_candidates("gfx950")
        )
        smoke = collections.Counter(
            c.backend for c in enumerate_mha_fwd_candidates("gfx950", "smoke")
        )
        for backend, count in full.items():
            if backend not in MHA_FWD_TILE_CONFIG_BACKENDS:
                with self.subTest(backend=backend):
                    self.assertEqual(smoke[backend], count)

    def test_a_sampled_grid_still_varies_every_tuning_axis(self):
        """A stride over the flattened product pins the inner axes; if that
        regresses, a smoke run would only ever sample block sizes."""
        for backend in ("triton", "gluon"):
            values = collections.defaultdict(set)
            for candidate in enumerate_mha_fwd_candidates("gfx950", "smoke"):
                if candidate.backend == backend:
                    for key, value in candidate.backend_config.items():
                        values[key].add(value)
            full_axes = collections.defaultdict(set)
            for candidate in enumerate_mha_fwd_candidates("gfx950"):
                if candidate.backend == backend:
                    for key, value in candidate.backend_config.items():
                        full_axes[key].add(value)
            for key, sampled in values.items():
                if len(full_axes[key]) > 1:
                    with self.subTest(backend=backend, axis=key):
                        self.assertGreater(len(sampled), 1)

    def test_smoke_is_deterministic(self):
        first = [c.identity for c in enumerate_mha_fwd_candidates("gfx950", "smoke")]
        second = [c.identity for c in enumerate_mha_fwd_candidates("gfx950", "smoke")]
        self.assertEqual(first, second)

    def test_an_unknown_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown MHA search strategy"):
            enumerate_mha_fwd_candidates("gfx950", "random")


class TestSelectionProof(unittest.TestCase):
    def test_layered_records_fold_into_one_selection(self):
        observed = fold_mha_fwd_selection_records(
            [
                {"backend": "triton", "num_splits": 0, "pid": 7},
                {"config": {"BLOCK_M": 64}, "config_backend": "triton", "pid": 7},
            ]
        )
        self.assertEqual(observed["backend"], "triton")
        self.assertEqual(observed["config"], {"BLOCK_M": 64})
        self.assertNotIn("pid", observed)

    def test_a_launch_key_the_tuner_never_varied_does_not_fail_the_proof(self):
        self.assertTrue(
            mha_fwd_config_matches({"BLOCK_M": 128}, {"BLOCK_M": 128, "num_warps": 4})
        )

    def test_routing_to_the_right_backend_with_the_wrong_tiles_fails(self):
        self.assertFalse(mha_fwd_config_matches({"BLOCK_M": 128}, {"BLOCK_M": 64}))

    def test_a_backend_that_reported_no_config_cannot_satisfy_one(self):
        self.assertFalse(mha_fwd_config_matches({"BLOCK_M": 128}, None))

    def test_a_name_is_config_winner_needs_no_config_evidence(self):
        self.assertTrue(mha_fwd_config_matches(None, None))


if __name__ == "__main__":
    unittest.main()
