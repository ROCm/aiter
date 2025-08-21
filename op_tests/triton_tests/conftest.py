import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skipextended", action="store_true", default=False, help="skip extended tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "extended: mark test as extended")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skipextended"):
        # --skipextended given in cli: skip extended tests
        skip_extended = pytest.mark.skip(reason="--skipextended set")
        for item in items:
            if "extended" in item.keywords:
                item.add_marker(skip_extended)
    return
