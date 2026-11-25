# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""IPCBufferPool exchanges IPC metadata through the default store with plain
set/get, so any backend-free KV store works. Both tcp:// (TCPStore) and
file:// (FileStore) init methods must be accepted, and callers such as vLLM
should not have to force tcp:// to use custom all-reduce. No GPU or process
group is needed: the stores are local and the pool is a stub.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
import torch.distributed as dist

from aiter.dist.device_communicators.custom_all_reduce import IPCBufferPool


def _file_store(tmp_path, world_size=1):
    store = dist.FileStore(str(tmp_path / "store"), world_size)
    # A missing peer key fails the test instead of blocking for the 300 s default.
    store.set_timeout(timedelta(seconds=30))
    return store


def _tcp_store():
    return dist.TCPStore("127.0.0.1", 0, 1, True, wait_for_workers=False)


@pytest.mark.parametrize("kind", ["tcp", "file"])
def test_backend_free_stores_are_accepted(kind, tmp_path):
    store = _tcp_store() if kind == "tcp" else _file_store(tmp_path)
    IPCBufferPool._assert_backend_free_store(store)
    IPCBufferPool._assert_backend_free_store(dist.PrefixStore("a", store))
    IPCBufferPool._assert_backend_free_store(
        dist.PrefixStore("b", dist.PrefixStore("a", store))
    )


def test_other_stores_are_rejected():
    with pytest.raises(AssertionError, match="HashStore"):
        IPCBufferPool._assert_backend_free_store(dist.HashStore())
    with pytest.raises(AssertionError, match="HashStore"):
        IPCBufferPool._assert_backend_free_store(
            dist.PrefixStore("a", dist.HashStore())
        )


def test_all_gather_object_over_file_store(tmp_path):
    """Each rank sees every rank's object, in rank order, over a FileStore."""
    world_size = 2

    def rank_fn(rank):
        pool = object.__new__(IPCBufferPool)
        pool._rank = rank
        pool._world_size = world_size
        pool._store = dist.PrefixStore("default_pg", _file_store(tmp_path, world_size))
        pool._store_key_prefix = "aiter_ipc/test"
        pool._ipc_seq = 0
        return [
            pool.all_gather_object({"rank": rank, "step": step}) for step in range(3)
        ]

    with ThreadPoolExecutor(max_workers=world_size) as ex:
        futures = [ex.submit(rank_fn, r) for r in range(world_size)]
        results = [f.result() for f in futures]

    expected = [
        [{"rank": r, "step": step} for r in range(world_size)] for step in range(3)
    ]
    assert results == [expected] * world_size


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
