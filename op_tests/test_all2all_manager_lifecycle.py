# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from aiter.dist.device_communicators.communicator_cuda import CudaCommunicator


@pytest.fixture(autouse=True)
def avoid_distributed_rank_lookup(monkeypatch):
    monkeypatch.setattr(
        "aiter.dist.device_communicators.communicator_cuda.is_global_first_rank",
        lambda: False,
    )


def make_communicator(*, use_all2all: bool = True) -> CudaCommunicator:
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    communicator.use_all2all = use_all2all
    communicator.unique_name = "ep" if use_all2all else "tp"
    communicator.all2all_backend = "mori"
    communicator._all2all_manager = None
    communicator._active_all2all_backend = None
    return communicator


def test_manager_is_created_only_when_requested(monkeypatch):
    communicator = make_communicator()
    created = []

    def create(backend):
        created.append(backend)
        return SimpleNamespace(backend=backend)

    monkeypatch.setattr(communicator, "_create_all2all_manager", create)

    assert communicator._all2all_manager is None
    manager = communicator.get_or_create_all2all_manager("flydsl")

    assert manager.backend == "flydsl"
    assert created == ["flydsl"]
    assert communicator.all2all_backend == "mori"
    assert communicator._active_all2all_backend == "flydsl"
    assert communicator.all2all_manager is manager


def test_compatibility_property_requests_configured_backend(monkeypatch):
    communicator = make_communicator()
    created = []

    def create(backend):
        created.append(backend)
        return SimpleNamespace(backend=backend)

    monkeypatch.setattr(communicator, "_create_all2all_manager", create)

    manager = communicator.all2all_manager

    assert manager.backend == "mori"
    assert created == ["mori"]


def test_same_backend_reuses_manager(monkeypatch):
    communicator = make_communicator()
    manager = object()
    created = []

    def create(backend):
        created.append(backend)
        return manager

    monkeypatch.setattr(communicator, "_create_all2all_manager", create)

    assert communicator.get_or_create_all2all_manager("mori") is manager
    assert communicator.get_or_create_all2all_manager("mori") is manager
    assert created == ["mori"]


def test_initialized_manager_cannot_switch_backend(monkeypatch):
    communicator = make_communicator()
    monkeypatch.setattr(
        communicator, "_create_all2all_manager", lambda backend: object()
    )

    communicator.get_or_create_all2all_manager("mori")

    with pytest.raises(RuntimeError, match="cannot switch"):
        communicator.get_or_create_all2all_manager("flydsl")


def test_failed_creation_can_be_retried(monkeypatch):
    communicator = make_communicator()
    manager = object()
    attempts = 0

    def create(_backend):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("initialization failed")
        return manager

    monkeypatch.setattr(communicator, "_create_all2all_manager", create)

    with pytest.raises(RuntimeError, match="initialization failed"):
        communicator.get_or_create_all2all_manager("mori")

    assert communicator._all2all_manager is None
    assert communicator._active_all2all_backend is None
    assert communicator.get_or_create_all2all_manager("mori") is manager


def test_unsupported_backend_is_rejected_without_changing_state():
    communicator = make_communicator()

    with pytest.raises(ValueError, match="Unsupported all2all manager backend"):
        communicator.get_or_create_all2all_manager("rccl")

    assert communicator._all2all_manager is None
    assert communicator._active_all2all_backend is None


def test_non_ep_communicator_cannot_create_manager():
    communicator = make_communicator(use_all2all=False)

    assert communicator.all2all_manager is None
    with pytest.raises(RuntimeError, match="not an expert-parallel communicator"):
        communicator.get_or_create_all2all_manager("mori")
