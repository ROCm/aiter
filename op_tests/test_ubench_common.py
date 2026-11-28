# SPDX-License-Identifier: MIT
import json

import pytest
import torch

from op_tests.ubench_common import fill, make_generator, print_json_table


@pytest.mark.parametrize("dist", ["zero", "constant", "uniform", "norm"])
def test_repeatable_input(dist):
    a = fill(
        (3, 17), dist, make_generator(42, "cpu"), dtype=torch.bfloat16, device="cpu"
    )
    b = fill(
        (3, 17), dist, make_generator(42, "cpu"), dtype=torch.bfloat16, device="cpu"
    )
    assert a.dtype == torch.bfloat16
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_invalid_input_distribution():
    with pytest.raises(ValueError, match="dist"):
        fill((1, 2), "invalid", None, device="cpu")


def test_json_records(capsys):
    print_json_table("softmax", [{"shape": [1, 17], "latency_us": 2.5}])
    assert json.loads(capsys.readouterr().out) == {
        "name": "softmax",
        "rows": [{"shape": [1, 17], "latency_us": 2.5}],
    }
