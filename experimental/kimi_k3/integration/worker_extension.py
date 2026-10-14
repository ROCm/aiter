"""Benchmark-only RPC: read sticky timeout flags after graph replay."""


class BenchmarkWorkerExtension:
    def inspect_k3_monokernel(self):
        from vllm.models.kimi_k3.amd.monokernel import inspect_adapters

        return inspect_adapters()

    def validate_k3_batch(self, batch):
        from vllm.forward_context import set_forward_context
        from vllm.models.kimi_k3.amd.monokernel import validate_batch

        with set_forward_context(None, self.vllm_config, num_tokens=batch):
            return validate_batch(batch)

    def start_k3_profile(self):
        import torch
        from vllm.distributed import get_tp_group

        rank = get_tp_group().rank_in_group
        if rank != 0:
            return {"rank": rank, "profiling": False}
        assert getattr(self, "_k3_profile", None) is None
        torch.cuda.synchronize()
        self._k3_profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        self._k3_profile.__enter__()
        return {"rank": rank, "profiling": True}

    def stop_k3_profile(self, label):
        import torch
        from pathlib import Path
        import os
        from vllm.distributed import get_tp_group

        rank = get_tp_group().rank_in_group
        if rank != 0:
            return {"rank": rank, "profiling": False}
        assert label in ("b1", "b2", "b4", "b8")
        torch.cuda.synchronize()
        profiler = self._k3_profile
        profiler.__exit__(None, None, None)
        path = Path(os.environ.get("KIMI_K3_RESULTS", "results")) / "profiles"
        path.mkdir(parents=True, exist_ok=True)
        destination = path / f"{label}-torch.json"
        profiler.export_chrome_trace(str(destination))
        self._k3_profile = None
        return {"rank": rank, "trace": str(destination)}
