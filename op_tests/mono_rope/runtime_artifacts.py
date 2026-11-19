"""Archive the exact A05 code objects returned by real Triton launches.

Use as a pytest plugin, or install() before a standalone benchmark. This only
observes completed launches; it does not change kernel arguments or binaries.
"""

import functools
import hashlib
import json
import os
from pathlib import Path


def install():
    directory = os.environ.get("MONO_ARTIFACT_DIR")
    if not directory:
        return
    from triton.runtime.jit import JITFunction

    if getattr(JITFunction.run, "_mono_artifacts", False):
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    original = JITFunction.run
    seen = set()

    @functools.wraps(original)
    def run(self, *args, **kwargs):
        compiled = original(self, *args, **kwargs)
        if not self.fn.__name__.startswith("_rope_window") or compiled is None:
            return compiled
        blob = compiled.asm["hsaco"]
        digest = hashlib.sha256(blob).hexdigest()
        variant = "persistent" if "TOKENS" in self.arg_names else "native"
        with (root / "dispatches.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(
                variant=variant, sha256=digest, grid=kwargs["grid"],
                num_warps=compiled.metadata.num_warps,
                warp_size=compiled.metadata.warp_size,
                warmup=kwargs.get("warmup", False),
            ), default=str) + "\n")
        if digest in seen:
            return compiled
        seen.add(digest)
        variant = "persistent" if "TOKENS" in self.arg_names else "native"
        target = root / f"A05_{variant}_{digest}"
        target.mkdir(exist_ok=True)
        (target / "kernel.hsaco").write_bytes(blob)
        for kind in ("amdgcn", "llir"):
            (target / f"kernel.{kind}").write_text(compiled.asm[kind])
        (target / "source.py").write_text(self.src)
        module_path = Path(self.fn.__code__.co_filename)
        module_source = module_path.read_bytes() if module_path.is_file() else None
        if module_source is not None:
            (target / "module.py").write_bytes(module_source)
        report = dict(
            module_source_path=str(module_path),
            module_source_sha256=(
                hashlib.sha256(module_source).hexdigest()
                if module_source is not None
                else None
            ),
            operator="A05",
            variant=variant,
            sha256=digest,
            triton_hash=compiled.hash,
            metadata=compiled.metadata._asdict(),
            grid=kwargs["grid"],
            warmup=kwargs.get("warmup"),
            source_module=self.fn.__module__,
            compile_options=kwargs,
        )
        (target / "launch.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        return compiled

    run._mono_artifacts = True
    JITFunction.run = run


def pytest_sessionstart(session):
    install()
