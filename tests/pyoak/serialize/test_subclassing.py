from dataclasses import dataclass

import pytest
from mashumaro.types import SerializableType
from pyoak.serialize import DataClassSerializeMixin


def test_subclass_error() -> None:
    with pytest.raises(ValueError):

        @dataclass(frozen=True)
        class SerializeTest(DataClassSerializeMixin):
            attr: str

        from . import subclass  # noqa: F401


@pytest.mark.skipif(
    not hasattr(SerializableType, "__slots__"),
    reason="Mashumaro version doesn't support slots",
)
def test_slotted(clean_ser_types: None) -> None:
    @dataclass(frozen=True, slots=True)
    class SlottedSubclass(DataClassSerializeMixin):
        attr: str

    assert not hasattr(SlottedSubclass("test"), "__dict__")
