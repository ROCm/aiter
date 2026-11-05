# SPDX-License-Identifier: MIT
"""AITER_CAR_POOL_BYTES parsing, and the cutoff staying inside the pool."""

from aiter.dist.device_communicators.custom_all_reduce import (
    _DEFAULT_CAR_MAX_SIZE,
    _DEFAULT_CAR_POOL_BYTES,
    _resolve_car_max_size,
    _resolve_car_pool_bytes,
)


def test_pool_default_is_one_gib(monkeypatch):
    monkeypatch.delenv("AITER_CAR_POOL_BYTES", raising=False)
    assert _resolve_car_pool_bytes(True) == _DEFAULT_CAR_POOL_BYTES
    assert _DEFAULT_CAR_POOL_BYTES == 1 << 30


def test_pool_strips_and_accepts_a_positive_size_on_gfx1250(monkeypatch):
    monkeypatch.setenv("AITER_CAR_POOL_BYTES", "  17179869184 ")
    assert _resolve_car_pool_bytes(True) == 17179869184


def test_pool_rejects_non_integer_and_non_positive(monkeypatch, capfd):
    # The aiter logger does not propagate, so the warning is on stderr.
    for raw, expected in (
        ("nope", "not an integer"),
        ("0", "not positive"),
        ("-4", "not positive"),
    ):
        monkeypatch.setenv("AITER_CAR_POOL_BYTES", raw)
        assert _resolve_car_pool_bytes(True) == _DEFAULT_CAR_POOL_BYTES
        assert expected in capfd.readouterr().err


def test_pool_env_is_ignored_off_gfx1250(monkeypatch, capfd):
    monkeypatch.setenv("AITER_CAR_POOL_BYTES", str(16 << 30))
    assert _resolve_car_pool_bytes(False) == _DEFAULT_CAR_POOL_BYTES
    assert "ignored" in capfd.readouterr().err


def test_unset_cutoff_is_capped_at_a_small_pool(monkeypatch):
    monkeypatch.delenv("AITER_CUSTOM_AR_MAX_SIZE", raising=False)
    small = 1 << 20
    assert _resolve_car_max_size(small) == small
    assert _resolve_car_max_size(1 << 30) == _DEFAULT_CAR_MAX_SIZE


def test_cutoff_above_the_pool_falls_back_inside_it(monkeypatch, capfd):
    monkeypatch.setenv("AITER_CUSTOM_AR_MAX_SIZE", str(1 << 30))
    small = 1 << 20
    assert _resolve_car_max_size(small) == small
    assert "exceeds" in capfd.readouterr().err
