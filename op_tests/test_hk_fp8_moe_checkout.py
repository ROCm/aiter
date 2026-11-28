# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU regressions for ownership, cleanliness, and shared dependency locks."""

from __future__ import annotations

import ast
import importlib.util
import multiprocessing
import os
import subprocess
import sys
import tempfile
import types
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
_UTILS = _ROOT / "aiter/jit/utils"


def load(name):
    spec = importlib.util.spec_from_file_location(name, _UTILS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


third_party = load("third_party")


def checkout_caller(checkout, jit_dir):
    # Exercise the real callers without importing Torch or probing a GPU.
    core = _ROOT / "aiter/jit/core.py"
    definitions = [
        node
        for node in ast.parse(core.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"mp_lock", "clone_3rdparty"}
    ]
    namespace = {
        "os": os,
        "Callable": Callable,
        "FileBaton": load("file_baton").FileBaton,
        "HIP_KITTENS_DIR": str(checkout),
        "bd_dir": str(jit_dir),
    }
    exec(  # noqa: S102 - execute only the two trusted repository definitions above
        compile(ast.Module(body=definitions, type_ignores=[]), str(core), "exec"),
        namespace,
    )
    return namespace["clone_3rdparty"]


def locked_checkout(checkout, jit_dir, remote, revision, started, entered, release):
    """A separate JIT process using the production lock and a local Git remote."""
    os.environ.pop("HIP_KITTENS_DIR", None)
    module = types.ModuleType("third_party")
    module.checkout_lock_path = third_party.checkout_lock_path

    def ensure(path, _url, _revision, **kwargs):
        entered.set()
        if not release.wait(15):
            raise RuntimeError("Timed out waiting for checkout test release")
        third_party.ensure_pinned_checkout(path, remote, revision, **kwargs)

    module.ensure_pinned_checkout = ensure
    with patch.dict(sys.modules, {"third_party": module}):
        caller = checkout_caller(checkout, jit_dir)
        started.set()
        caller("HipKittens")


class TestDependencyCheckoutOwnership(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote, self.cache = self.root / "remote", self.root / "cache"
        self.git("init", "-q", str(self.remote))
        self.git("-C", str(self.remote), "config", "user.name", "test")
        self.git("-C", str(self.remote), "config", "user.email", "test@example.invalid")
        self.old = self.commit("old")
        self.new = self.commit("new")

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.STDOUT
        ).strip()

    def commit(self, content):
        (self.remote / "header").write_text(content)
        self.git("-C", str(self.remote), "add", "header")
        self.git("-C", str(self.remote), "commit", "-qm", content)
        return self.git("-C", str(self.remote), "rev-parse", "HEAD")

    def ensure(self, revision):
        third_party.ensure_pinned_checkout(self.cache, str(self.remote), revision)

    def test_dubious_ownership_is_trusted_only_for_the_managed_command(self):
        self.ensure(self.old)
        config = (self.cache / ".git/config").read_text()
        bundle = self.root / "remote.bundle"
        self.git("-C", str(self.remote), "bundle", "create", str(bundle), "--all")
        # Git's own test switch simulates a host-owned bind mount without chown
        # or root. A bundle avoids applying the switch to a remote Git server.
        owner_env = {
            "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1",
            "GIT_CONFIG_GLOBAL": str(self.root / "unused-global-config"),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "0",
        }
        with patch.dict(os.environ, owner_env):
            try:
                self.git("-C", str(self.cache), "rev-parse", "HEAD")
            except subprocess.CalledProcessError as error:
                self.assertIn("dubious ownership", error.output)
            else:
                self.skipTest("Git does not support the ownership test switch")
            third_party.ensure_pinned_checkout(self.cache, str(bundle), self.new)
            alias = self.root / "cache-alias"
            alias.symlink_to(self.cache, target_is_directory=True)
            # A clean pin, including a symlink alias, must also work offline.
            third_party.ensure_pinned_checkout(alias, "/nonexistent", self.new)
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                self.git("-C", str(self.cache), "rev-parse", "HEAD")
            self.assertIn("dubious ownership", raised.exception.output)
        self.assertEqual((self.cache / "header").read_text(), "new")
        self.assertEqual((self.cache / ".git/config").read_text(), config)
        self.assertFalse((self.root / "unused-global-config").exists())

    def test_dirty_checkout_at_the_pin_is_rejected_and_preserved(self):
        self.ensure(self.old)
        for change in ("tracked", "staged", "untracked"):
            with self.subTest(change=change):
                header = self.cache / (
                    "custom-header" if change == "untracked" else "header"
                )
                header.write_text("custom headers")
                if change == "staged":
                    self.git("-C", str(self.cache), "add", "header")
                with self.assertRaisesRegex(
                    RuntimeError, "modified dependency.*HIP_KITTENS_DIR"
                ):
                    self.ensure(self.old)
                self.assertEqual(header.read_text(), "custom headers")
                self.assertEqual(
                    self.git("-C", str(self.cache), "rev-parse", "HEAD"), self.old
                )
                if change == "untracked":
                    header.unlink()
                else:
                    self.git("-C", str(self.cache), "reset", "--hard", self.old)

    def test_override_keeps_custom_headers_without_git_or_checkout_lock(self):
        self.cache.mkdir()
        (self.cache / "header").write_text("custom headers")
        module = types.ModuleType("third_party")
        module.ensure_pinned_checkout = third_party.ensure_pinned_checkout
        module.checkout_lock_path = lambda _: self.fail(
            "Override requested a mutation lock"
        )
        with (
            patch.dict(os.environ, {"HIP_KITTENS_DIR": str(self.cache)}),
            patch.dict(sys.modules, {"third_party": module}),
            patch.object(
                third_party, "_git", side_effect=AssertionError("Override invoked Git")
            ),
        ):
            checkout_caller(self.cache, self.root / "jit")("HipKittens")
        self.assertEqual((self.cache / "header").read_text(), "custom headers")

    def test_offline_archive_fetch_error_names_the_override_and_keeps_sources(self):
        archive = self.remote / "HipKittens"
        archive.mkdir()
        (archive / "header").write_text("bundled headers")
        managed = Path(third_party.checkout_path(archive, self.root / "jit"))
        with self.assertRaisesRegex(RuntimeError, "HIP_KITTENS_DIR") as raised:
            third_party.ensure_pinned_checkout(managed, "/nonexistent", self.old)
        self.assertIn(str(managed), str(raised.exception))
        self.assertIn(self.old, str(raised.exception))
        self.assertIn("offline", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, subprocess.CalledProcessError)
        self.assertFalse(managed.exists())
        self.assertEqual(list(managed.parent.iterdir()), [])
        self.assertEqual((archive / "header").read_text(), "bundled headers")
        self.assertEqual(
            self.git("-C", str(self.remote), "rev-parse", "HEAD"), self.new
        )
        with self.assertRaisesRegex(RuntimeError, "HIP_KITTENS_DIR"):
            third_party.ensure_pinned_checkout(archive, "/nonexistent", self.old)

    def test_missing_git_error_names_the_override(self):
        with (
            patch.dict(os.environ, {"PATH": ""}),
            self.assertRaisesRegex(RuntimeError, "HIP_KITTENS_DIR"),
        ):
            self.ensure(self.old)
        self.assertFalse(self.cache.exists())

    def test_clean_managed_checkout_does_not_require_a_writable_parent(self):
        self.ensure(self.old)
        module = types.ModuleType("third_party")
        module.checkout_lock_path = third_party.checkout_lock_path
        module.ensure_pinned_checkout = lambda path, _url, _revision, **kwargs: (
            third_party.ensure_pinned_checkout(path, "/nonexistent", self.old, **kwargs)
        )
        self.root.chmod(0o555)
        try:
            with (
                patch.dict(sys.modules, {"third_party": module}),
                patch.dict(os.environ),
            ):
                os.environ.pop("HIP_KITTENS_DIR", None)
                checkout_caller(self.cache, self.root / "jit")("HipKittens")
        finally:
            self.root.chmod(0o755)
        self.assertEqual((self.cache / "header").read_text(), "old")

    def test_different_jit_processes_share_the_checkout_lock(self):
        self.ensure(self.old)
        alias = self.root / "alias"
        alias.symlink_to(self.cache, target_is_directory=True)
        context = multiprocessing.get_context("spawn")
        release = context.Event()
        started = [context.Event(), context.Event()]
        entered = [context.Event(), context.Event()]
        processes = [
            context.Process(
                target=locked_checkout,
                args=(
                    str(checkout),
                    str(self.root / f"jit-{index}"),
                    str(self.remote),
                    self.new,
                    started[index],
                    entered[index],
                    release,
                ),
            )
            for index, checkout in enumerate((self.cache, alias))
        ]
        try:
            processes[0].start()
            self.assertTrue(
                entered[0].wait(10), "First process did not acquire the lock"
            )
            processes[1].start()
            self.assertTrue(started[1].wait(10), "Second process did not start")
            self.assertFalse(
                entered[1].wait(0.5), "Second JIT cache bypassed the shared lock"
            )
            release.set()
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            # A waiter must validate its own request after the holder completes.
            self.assertTrue(entered[1].is_set())
            self.assertEqual(
                self.git("-C", str(self.cache), "rev-parse", "HEAD"), self.new
            )
            self.assertFalse(Path(third_party.checkout_lock_path(self.cache)).exists())
        finally:
            release.set()
            for process in processes:
                if process.pid is not None:
                    if process.is_alive():
                        process.terminate()
                    process.join(10)


if __name__ == "__main__":
    unittest.main()
