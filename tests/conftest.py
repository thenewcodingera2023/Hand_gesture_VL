"""Pytest configuration: register custom marks."""

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: heavy tests that rebuild splits end-to-end (deselect with -m 'not slow')",
    )
