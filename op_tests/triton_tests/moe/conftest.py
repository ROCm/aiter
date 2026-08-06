import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--golden-cpu",
        action="store_true",
        default=False,
        help="Compute reference (golden) outputs on CPU instead of GPU. Frees GPU "
        "memory for the kernel under test (e.g. large MiniMax-M3 shapes); the "
        "reference matmul is slower. Currently used by test_moe_gemm_a16w4.",
    )


@pytest.fixture
def golden_cpu(request):
    return request.config.getoption("--golden-cpu")
