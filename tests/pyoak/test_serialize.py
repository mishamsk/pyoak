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


@pytest.mark.dep_orjson
def test_json_serialize() -> None:
    """Test roundtrip (de)serialization from/to JSON with all serialization options variations."""

    pytest.importorskip("orjson")

    m = SerializeTestClass(1)

    # Test default options
    d = m.to_json()

    assert d == f'{{"{TYPE_KEY}":"SerializeTestClass","foo":1,"bar":-1,"nested":null}}'
    assert SerializeTestClass.from_json(d) == m
    assert d == m.to_jsonb().decode(encoding="utf-8")

    # Test indent
    d = m.to_json(indent=True)

    assert (
        d
        == f'{{\n  "{TYPE_KEY}": "SerializeTestClass",\n  "foo": 1,\n  "bar": -1,\n  "nested": null\n}}'
    )
    assert d == m.to_jsonb(indent=True).decode(encoding="utf-8")

    # Test skip class
    d = m.to_json(serialization_options={SerializationOption.SKIP_CLASS: True})

    assert d == '{"foo":1,"bar":-1,"nested":null}'
    assert SerializeTestClass.from_json(d) == m
    assert d == m.to_jsonb(serialization_options={SerializationOption.SKIP_CLASS: True}).decode(
        encoding="utf-8"
    )

    # Test sort keys
    d = m.to_json(serialization_options={SerializationOption.SORT_KEYS: True})

    # Class always goes first even with sorted keys

    assert d == f'{{"{TYPE_KEY}":"SerializeTestClass","bar":-1,"foo":1,"nested":null}}'
    assert SerializeTestClass.from_json(d) == m
    assert d == m.to_jsonb(serialization_options={SerializationOption.SORT_KEYS: True}).decode(
        encoding="utf-8"
    )

    # Test skip class and sort keys
    d = m.to_json(
        serialization_options={
            SerializationOption.SKIP_CLASS: True,
            SerializationOption.SORT_KEYS: True,
        }
    )

    assert d == '{"bar":-1,"foo":1,"nested":null}'
    assert SerializeTestClass.from_json(d) == m
    assert d == m.to_jsonb(
        serialization_options={
            SerializationOption.SKIP_CLASS: True,
            SerializationOption.SORT_KEYS: True,
        }
    ).decode(encoding="utf-8")


def test_no_extras_serialize_funcs() -> None:
    assert not hasattr(SerializeTestClass, "from_yaml")
    assert not hasattr(SerializeTestClass, "to_yaml")
    assert not hasattr(SerializeTestClass, "from_msgpck")
    assert not hasattr(SerializeTestClass, "to_msgpck")
    assert hasattr(SerializeTestClass, "from_json")
    assert hasattr(SerializeTestClass, "to_json")
    assert hasattr(SerializeTestClass, "to_jsonb")


@pytest.mark.dep_pyyaml
def test_yaml_serialize_funcs() -> None:
    pytest.importorskip("yaml")

    assert hasattr(SerializeTestClass, "from_yaml")
    assert hasattr(SerializeTestClass, "to_yaml")

    m = SerializeTestClass(1)

    # Test default options
    d = m.to_yaml()

    assert (
        d
        == f"""\
{TYPE_KEY}: SerializeTestClass
bar: -1
foo: 1
nested: null
"""
    )
    assert SerializeTestClass.from_yaml(d) == m

    m = SerializeTestClass(1, nested=m)

    # Test skip class
    d = m.to_yaml(serialization_options={SerializationOption.SKIP_CLASS: True})

    assert (
        d
        == """\
bar: -1
foo: 1
nested:
  bar: -1
  foo: 1
  nested: null
"""
    )
    assert SerializeTestClass.from_yaml(d) == m


@pytest.mark.dep_ruamel
def test_ruamel_serialize_funcs() -> None:
    """Compared to the default pyyaml:
    - ruamel.yaml uses the same order as the object, this means that the SerializationOption.SORT_KEYS is working
    - ruamel.yaml treats None in different ways, see tests below
    """
    pytest.importorskip("ruamel.yaml")

    assert hasattr(SerializeTestClass, "from_yaml")
    assert hasattr(SerializeTestClass, "to_yaml")

    m = SerializeTestClass(1)

    # Test default options
    d = m.to_yaml()

    assert (
        d
        == f"""\
{TYPE_KEY}: SerializeTestClass
foo: 1
bar: -1
nested:
"""
    )
    assert SerializeTestClass.from_yaml(d) == m

    m = SerializeTestClass(1, nested=m)

    # Test skip class
    d = m.to_yaml(
        serialization_options={
            SerializationOption.SKIP_CLASS: True,
            SerializationOption.SORT_KEYS: True,
        }
    )

    assert (
        d
        == """\
bar: -1
foo: 1
nested:
  bar: -1
  foo: 1
  nested:
"""
    )
    assert SerializeTestClass.from_yaml(d) == m


@pytest.mark.dep_msgpack
def test_msgpack_serialize_funcs() -> None:
    pytest.importorskip("msgpack")

    assert hasattr(SerializeTestClass, "from_msgpck")
    assert hasattr(SerializeTestClass, "to_msgpck")


@pytest.mark.dep_all
def test_all_serialize_funcs() -> None:
    pytest.importorskip("ruamel.yaml")
    pytest.importorskip("msgpack")
    pytest.importorskip("orjson")

    assert hasattr(SerializeTestClass, "from_yaml")
    assert hasattr(SerializeTestClass, "to_yaml")
    assert hasattr(SerializeTestClass, "from_msgpck")
    assert hasattr(SerializeTestClass, "to_msgpck")
    assert hasattr(SerializeTestClass, "from_json")
    assert hasattr(SerializeTestClass, "to_json")
    assert hasattr(SerializeTestClass, "to_jsonb")
