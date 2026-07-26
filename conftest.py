from importlib.metadata import version

EXPECTED_FLYDSL_VERSION = "0.3.0.dev20260725+7f363ef"


def pytest_configure(config):
    v = version("flydsl")
    if v != EXPECTED_FLYDSL_VERSION:
        raise RuntimeError(
            f"flydsl version mismatch: installed {v}, expected {EXPECTED_FLYDSL_VERSION}"
        )
