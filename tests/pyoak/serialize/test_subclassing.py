from dataclasses import dataclass

import pytest
from pyoak.serialize import DataClassSerializeMixin


def test_subclass_error() -> None:
    with pytest.raises(ValueError):

        @dataclass(frozen=True)
        class SerializeTest(DataClassSerializeMixin):
            attr: str

        from . import subclass  # noqa: F401
