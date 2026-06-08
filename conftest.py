from importlib.metadata import version

EXPECTED_FLYDSL_VERSION = "0.2.0.dev20260608+c957349"


def pytest_configure(config):
    v = version("flydsl")
    if v != EXPECTED_FLYDSL_VERSION:
        raise RuntimeError(
            f"flydsl version mismatch: installed {v}, expected {EXPECTED_FLYDSL_VERSION}"
        )
