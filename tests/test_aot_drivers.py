"""AOT driver coverage: MLA config wiring and PA-Gluon's GPU worker policy."""

import contextlib
import inspect
import pathlib
import sys
import unittest
from unittest.mock import MagicMock, call, patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from aiter.aot import asm_mla_decode_fwd as driver
from csrc.cpp_itfs import utils as cpp_itfs_utils
from csrc.cpp_itfs.mla import asm_mla_decode_fwd as compile_module
from csrc.cpp_itfs.pa_gluon_aot import pa_decode_gluon_aot_prebuild as pa_gluon


class AsmMlaDecodeAotTest(unittest.TestCase):
    def test_config_fields_match_compile_api(self):
        compile_parameters = tuple(
            inspect.signature(compile_module.compile).parameters
        )[:6]

        self.assertEqual(driver.MLAConfig._fields, compile_parameters)


class PaGluonProcessLimitTest(unittest.TestCase):
    def run_one_config(self):
        return pa_gluon.run_multi_pa_gluon_test(
            block_sizes=[16],
            head_configs=[(1, 1)],
            context_length=[1],
            batch_sizes=[1],
            query_lengths=[1],
            quant_mode=["per_tensor"],
            trans_v=[False],
            kv_varlen=[False],
            compute_types_quant_q_and_kv=[[torch.bfloat16, False, False]],
            use_torch_flash_ref_options=[True],
            use_aot_impl_options=[True],
            context_partition_size_options=[256],
            sinks_options=[False],
            sliding_window_options=[0],
        )

    @patch.object(pa_gluon.torch.cuda, "device_count", return_value=3)
    @patch.object(pa_gluon, "get_gpu_worker_count", return_value=1)
    def test_pool_uses_gpu_worker_policy(self, worker_count, device_count):
        executor = MagicMock()
        executor.__enter__.return_value = executor
        executor.__exit__.return_value = False
        executor.map.return_value = []

        with patch.object(
            pa_gluon.concurrent.futures,
            "ProcessPoolExecutor",
            return_value=executor,
        ) as pool:
            result = self.run_one_config()

        device_count.assert_called_once_with()
        worker_count.assert_called_once_with(1, 3)
        self.assertEqual(pool.call_args.kwargs["max_workers"], 1)
        self.assertIs(
            pool.call_args.kwargs["initializer"],
            pa_gluon._init_gpu_worker,
        )
        counter, devices = pool.call_args.kwargs["initargs"]
        self.assertEqual(counter.value, 0)
        self.assertEqual(devices, 3)
        self.assertEqual(len(result), 0)

    def test_prebuild_requires_a_visible_gpu_before_starting_workers(self):
        with (
            patch.object(pa_gluon.torch.cuda, "device_count", return_value=0),
            patch.object(pa_gluon.concurrent.futures, "ProcessPoolExecutor") as pool,
            self.assertRaisesRegex(RuntimeError, "requires at least one visible GPU"),
        ):
            self.run_one_config()
        pool.assert_not_called()

    def test_workers_are_bound_evenly_to_visible_devices(self):
        counter = pa_gluon.multiprocessing.get_context("spawn").Value("i", 0)
        with (
            patch.object(pa_gluon, "configure_worker_subprocesses") as configure,
            patch.object(pa_gluon.torch.cuda, "set_device") as set_device,
        ):
            for _ in range(7):
                pa_gluon._init_gpu_worker(counter, 3)

        self.assertEqual(configure.call_count, 7)
        self.assertEqual(
            set_device.call_args_list,
            [call(device) for device in [0, 1, 2, 0, 1, 2, 0]],
        )

    def test_kernel_allocations_use_the_workers_assigned_device(self):
        with (
            patch.object(pa_gluon.torch.cuda, "current_device", return_value=2),
            patch.object(pa_gluon.torch, "set_default_device") as default_device,
            patch.object(
                pa_gluon.torch, "tensor", side_effect=RuntimeError("allocation reached")
            ) as tensor,
            self.assertRaisesRegex(RuntimeError, "allocation reached"),
        ):
            pa_gluon.run_pa_gluon_test(
                context_length=1,
                batch_size=1,
                num_heads=(1, 1),
                head_size=128,
                block_size=16,
                compute_type=torch.bfloat16,
                query_length=1,
                quant_mode="per_tensor",
                context_partition_size=256,
                trans_v=False,
                kv_varlen=False,
                use_aot_impl=True,
                quant_q=False,
                quant_kv=False,
            )

        default_device.assert_called_once_with("cuda:2")
        self.assertEqual(tensor.call_args.kwargs["device"], "cuda:2")

    @patch.object(cpp_itfs_utils, "get_worker_count_for", return_value=1)
    def test_nested_build_uses_one_job_on_both_platforms(self, worker_count):
        for windows, command in ((False, ["make", "build"]), (True, ["ninja"])):
            with (
                self.subTest(windows=windows),
                patch.object(cpp_itfs_utils, "IS_WINDOWS", windows),
            ):
                self.assertEqual(cpp_itfs_utils._build_command(3), [*command, "-j1"])
        self.assertEqual(worker_count.call_args_list, [call(3), call(3)])


class FlyDSLMoeDeploymentTest(unittest.TestCase):
    def test_buckets_follow_get_padded_m(self):
        from aiter.aot.flydsl import moe
        from aiter.fused_moe import get_padded_M

        for max_token in (1, 3, 1000, 16384, 16385, 40000, 131071, 131072, 200000):
            buckets = moe._buckets(max_token)
            starts = [low for _, low, _ in buckets[1:]] + [max_token + 1]
            self.assertEqual(buckets[0][1], 1)
            for (bucket, low, high), next_low in zip(buckets, starts):
                with self.subTest(max_token=max_token, bucket=bucket):
                    self.assertEqual(next_low, high + 1)
                    self.assertEqual(get_padded_M(low), bucket)
                    self.assertEqual(get_padded_M(high), bucket)

    @staticmethod
    @contextlib.contextmanager
    def _edited(edit):
        """The path of a copy of the default deployment file after ``edit``."""
        import json
        import tempfile

        from aiter.aot.flydsl import deployments

        with open(deployments.DEFAULT_DEPLOYMENTS[0]) as f:
            document = json.load(f)
        edit(document)
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(document, f)
            f.flush()
            yield f.name

    def _load_edited(self, edit):
        """Load a copy of the default deployment file after ``edit``."""
        from aiter.aot.flydsl import deployments

        with self._edited(edit) as path:
            return deployments.load([path])

    @staticmethod
    def _rank_call(deployment):
        """(local experts, routing columns, mask length, intermediate dim,
        FHMoE shared expert slot) of one rank's fused_moe."""
        return (
            deployment.local_experts,
            deployment.topk_cols,
            deployment.mask_size,
            deployment.inter_dim,
            deployment.shared_expert_id,
        )

    def test_default_deployments_derive_rank_calls(self):
        from aiter.aot.flydsl import deployments

        by_id = {d.id: d for d in deployments.load().fused_moe}
        expected = {
            "deepseek-v4-pro/gfx950/a8w4/ep8-r32": (52, 6, 417, 3072, -1),
            "glm-5.3-flash/gfx950/a4w4/ep8-fused": (37, 10, 290, 2048, -1),
            "glm-5.3-flash/gfx950/a4w4/ep4": (72, 8, 289, 2048, -1),
            "kimi-k3/gfx1250/a4w4/ep16": (56, 16, 897, 3072, -1),
            "deepseek-v4-pro/gfx950/a8w4/tp8": (384, 6, 0, 384, -1),
            "deepseek-v4-pro/gfx950/a8w4/tp8-fhmoe": (385, 7, 0, 384, 384),
        }
        for name, shape in expected.items():
            with self.subTest(name):
                self.assertEqual(self._rank_call(by_id[name]), shape)

    def test_default_deployments_replay_their_callers(self):
        from aiter.aot.flydsl import deployments, moe

        fused_moe = deployments.load().fused_moe
        self.assertTrue(any(d.is_ep for d in fused_moe))
        self.assertTrue(any(not d.is_ep for d in fused_moe))
        callers = {}
        for job in moe.parse_deployments(deployments.DEFAULT_DEPLOYMENTS):
            caller = (
                job["expert"],
                job["expert_mask_size"],
                job["shared_expert_id"],
                job["local_tokens"],
            )
            callers.setdefault(job["deployment"], set()).add(caller)
        for deployment in fused_moe:
            with self.subTest(deployment.id):
                mask = str(deployment.mask_size or "")
                slot = deployment.shared_expert_id
                shared = str(slot) if slot >= 0 else ""
                # Only a caller behind an EP dispatch passes num_local_tokens.
                local_tokens = (False, True) if deployment.is_ep else (False,)
                expected = {
                    (str(deployment.local_experts), mask, shared, passed)
                    for passed in local_tokens
                }
                self.assertEqual(callers[deployment.id], expected)

    def test_deployment_jobs_pass_model_activation(self):
        from aiter.aot.flydsl import moe

        def edit(document):
            models = document["models"]
            models["kimi-k3-betas"] = dict(
                models["kimi-k3"], situ_beta=2.0, situ_linear_beta=17.0
            )
            for deployment in document["fused_moe"]:
                if deployment["model"] == "kimi-k3":
                    deployment["model"] = "kimi-k3-betas"

        with self._edited(edit) as path:
            jobs = moe.parse_deployments([path])
        activations = {}
        for job in jobs:
            activations.setdefault(job["deployment"].split("/")[0], set()).add(
                (job["swiglu_limit"], job["beta"], job["linear_beta"])
            )
        # DeepSeek-V4-Pro's clamp reaches every call; SiLU takes no betas.
        self.assertEqual(activations["deepseek-v4-pro"], {(10.0, None, None)})
        self.assertEqual(activations["glm-5.3-flash"], {(None, None, None)})
        # A model's own SiTUv2 betas join the generic ones.
        generic = {(None, None), (1.0, 1.0), (4.0, 25.0)}
        self.assertEqual(
            activations["kimi-k3-betas"],
            {(None, *betas) for betas in generic | {(2.0, 17.0)}},
        )

        first = {}
        for job in jobs:
            first.setdefault(job["deployment"], job)
        with (
            patch.object(moe, "_caller_inputs", return_value={}),
            patch("aiter.fused_moe._fused_moe_impl") as fused,
            patch("aiter.fhmoe._fhmoe_impl") as fhmoe,
        ):
            moe._replay(first["deepseek-v4-pro/gfx950/a8w4/tp8"])
            moe._replay(first["deepseek-v4-pro/gfx950/a8w4/tp8-fhmoe"])
        self.assertEqual(fused.call_args.kwargs["swiglu_limit"], 10.0)
        self.assertEqual(fhmoe.call_args.kwargs["swiglu_limit"], 10.0)

    def test_tensor_parallel_derivation(self):
        from aiter.aot.flydsl import moe

        base = {
            "model": "deepseek-v4-pro",
            "gfx": "gfx950",
            "cu_num": 256,
            "quant": "a8w4",
            "max_tokens": 1024,
            "source": "test",
        }

        def edit(document):
            document.clear()
            document.update(
                schema_version=1,
                fused_moe=[
                    dict(base, tp=[8, 4], shared_expert_mode="fused"),
                    dict(base),
                ],
            )

        by_id = {d.id: d for d in self._load_edited(edit).fused_moe}
        # A shared expert fused under TP takes a column but no sentinel.
        expected = {
            "deepseek-v4-pro/gfx950/a8w4/tp8-fused": (385, 7, 0, 384, -1),
            "deepseek-v4-pro/gfx950/a8w4/tp4-fused": (385, 7, 0, 768, -1),
            "deepseek-v4-pro/gfx950/a8w4/tp1": (384, 6, 0, 3072, -1),
        }
        self.assertEqual(set(by_id), set(expected))
        for name, shape in expected.items():
            with self.subTest(name):
                self.assertEqual(self._rank_call(by_id[name]), shape)
        row = moe.deployment_row(by_id["deepseek-v4-pro/gfx950/a8w4/tp4-fused"])
        self.assertEqual(
            (row["model_dim"], row["inter_dim"], row["expert"], row["topk"]),
            ("7168", "768", "385", "7"),
        )
        self.assertEqual((row["expert_mask_size"], row["intermediate_pad"]), ("", ""))

    def test_default_mega_moe_jobs(self):
        from aiter.aot.flydsl import mega_moe

        jobs = mega_moe.deployment_jobs()
        # DeepSeek-V4-Pro r0/r32/r64 and Kimi-K3, by two quants, three max
        # tokens per rank and eight ranks.
        self.assertEqual(len(jobs), 4 * 2 * 3 * 8)
        self.assertEqual({job["experts_per_rank"] for job in jobs}, {48, 52, 56, 112})
        for job in jobs:
            situ = job["experts_per_rank"] == 112
            activation = (
                ("situv2", 4.0, 25.0, 0.0) if situ else ("silu", 1.0, 1.0, 10.0)
            )
            key = ("act", "situ_beta", "situ_linear_beta", "swiglu_limit")
            self.assertEqual(tuple(job[k] for k in key), activation)
        narrowed = mega_moe.deployment_jobs(mtprs=[8192], experts_per_ranks=[48])
        self.assertEqual(len(narrowed), 2 * 8)

    def test_own_file_names_default_models(self):
        deployment = {
            "model": "deepseek-v4-pro",
            "gfx": "gfx950",
            "cu_num": 256,
            "quant": "a8w4",
            "ep": 32,
            "max_tokens": 1024,
            "source": "test",
        }

        def edit(document):
            document.clear()
            document.update(schema_version=1, fused_moe=[deployment])

        (loaded,) = self._load_edited(edit).fused_moe
        self.assertEqual(loaded.id, "deepseek-v4-pro/gfx950/a8w4/ep32")
        self.assertEqual((loaded.local_experts, loaded.mask_size), (12, 385))

    def test_deployment_validation(self):
        def unfusable(document):
            model = dict(document["models"]["glm-5.3-flash"], shared_experts=0)
            document["models"]["glm-no-shared"] = model
            fused = next(
                d for d in document["fused_moe"] if d.get("shared_expert_mode")
            )
            fused["model"] = "glm-no-shared"

        def tp8(document):
            return next(d for d in document["fused_moe"] if d.get("tp"))

        def two_shared(document):
            model = dict(document["models"]["deepseek-v4-pro"], shared_experts=2)
            document["models"]["dsv4-two-shared"] = model
            fhmoe = next(
                d
                for d in document["fused_moe"]
                if d.get("shared_expert_mode") == "fhmoe"
            )
            fhmoe["model"] = "dsv4-two-shared"

        cases = {
            "over TP or over EP, not both": lambda d: d["fused_moe"][1].update(tp=2),
            "need an EP group": lambda d: tp8(d).update(redundant_experts=32),
            "does not split over TP7": lambda d: tp8(d).update(tp=7),
            "FHMoE runs without EP": lambda d: d["fused_moe"][1].update(
                shared_expert_mode="fhmoe"
            ),
            "FHMoE runs exactly one shared expert": two_shared,
            "unknown keys": lambda d: d["fused_moe"][0].update(redundant=[32]),
            "do not split": lambda d: d["fused_moe"][1].update(
                redundant_experts=[0, 36]
            ),
            "no shared experts to fuse": unfusable,
            "unknown model": lambda d: d["fused_moe"][0].update(model="kimi-k4"),
            "listed twice": lambda d: d["fused_moe"].append(dict(d["fused_moe"][0])),
            "exactly for Situv2": lambda d: d["models"]["deepseek-v4-pro"].update(
                situ_beta=4.0
            ),
            "missing keys": lambda d: d["mega_moe"][0].pop("max_tokens_per_rank"),
            "cu_num must be one of": lambda d: d["mega_moe"][0].update(cu_num=304),
            "differs from its other definition": lambda d: d["models"][
                "kimi-k3"
            ].update(top_k=8),
        }
        for message, edit in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                self._load_edited(edit)
