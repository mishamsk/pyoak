from __future__ import annotations

from dataclasses import dataclass

import pytest
from mashumaro.exceptions import InvalidFieldValue
from pyoak.serialize import (
    TYPE_KEY,
    DataClassSerializeMixin,
    SerializationOption,
    unwrap_invalid_field_exception,
)


@dataclass
class SerializeTestClass(DataClassSerializeMixin):
    foo: int
    bar: int = -1
    nested: SerializeTestClass | None = None


def test_nested_error_validation() -> None:
    # Create a nested class with 10 levels of nesting
    leaf = SerializeTestClass(11, nested=None)
    path = "foo"

    obj = leaf
    for i in range(10, 0, -1):
        obj = SerializeTestClass(i, nested=obj)
        path = f"nested.{path}"

    # Serialize to dict
    full_d = d = obj.as_dict()

    # Break the value in the leaf node
    while d["nested"] is not None:
        d = d["nested"]

    d["foo"] = "not an int"

    with pytest.raises(InvalidFieldValue) as excinfo:
        SerializeTestClass.from_dict(full_d)

    exc_path, exc = unwrap_invalid_field_exception(excinfo.value)
    assert exc_path == path
    assert isinstance(exc, ValueError)
    assert str(exc) == "invalid literal for int() with base 10: 'not an int'"


def test_serialization_options() -> None:
    m = SerializeTestClass(1)

    # Test default options
    d = m.as_dict()

    assert d == {TYPE_KEY: "SerializeTestClass", "foo": 1, "bar": -1, "nested": None}

    # Test skip class
    d = m.as_dict(serialization_options={SerializationOption.SKIP_CLASS: True})

    assert d == {"foo": 1, "bar": -1, "nested": None}

    # Test sort keys
    d = m.as_dict(serialization_options={SerializationOption.SORT_KEYS: True})

    # Class always goes first even with sorted keys
    assert d == {TYPE_KEY: "SerializeTestClass", "bar": -1, "foo": 1, "nested": None}

    # Test skip class and sort keys
    d = m.as_dict(
        serialization_options={
            SerializationOption.SKIP_CLASS: True,
            SerializationOption.SORT_KEYS: True,
        }
    )

    assert d == {"bar": -1, "foo": 1, "nested": None}
