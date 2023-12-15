import pytest


@pytest.fixture(scope="module", autouse=True)
def auto_clean_ser_types():
    """Automatically cleans all new types added to pyoak.serialize.TYPES by a test."""
    import pyoak.serialize

    cached = pyoak.serialize.TYPES.copy()
    yield
    pyoak.serialize.TYPES = cached


@pytest.fixture
def clean_ser_types():
    """Cleans all new types added to pyoak.serialize.TYPES by a single test."""
    import pyoak.serialize

    cached = pyoak.serialize.TYPES.copy()
    yield
    pyoak.serialize.TYPES = cached
