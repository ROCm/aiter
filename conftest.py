from importlib.metadata import version

EXPECTED_FLYDSL_VERSION = "0.1.9.dev20260529+d7ae22d"


def pytest_configure(config):
    v = version("flydsl")
    if v != EXPECTED_FLYDSL_VERSION:
        raise RuntimeError(
            f"flydsl version mismatch: installed {v}, expected {EXPECTED_FLYDSL_VERSION}"
        )
