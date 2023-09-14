from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from pyoak.typing import process_node_fields

if TYPE_CHECKING:
    from dataclasses import Field as DataClassField

    from .node import ASTNode
    from .typing import FieldTypeInfo

    Field = DataClassField[Any]

# Cached mappings of ASTNode subclasses to their properties and child fields
_TYPE_TO_ALL_FIELDS: dict[type[ASTNode], Mapping[Field, FieldTypeInfo]] = {}
_TYPE_TO_CHILD_FIELDS: dict[type[ASTNode], Mapping[Field, FieldTypeInfo]] = {}
_TYPE_TO_PROPS: dict[type[ASTNode], Mapping[Field, FieldTypeInfo]] = {}


def _populate_type_dicts(cls: type[ASTNode]) -> None:
    """Returns a tuple of all fields of the given class."""

    # Dynamic import to avoid circular imports
    from .node import ASTNode

    _TYPE_TO_CHILD_FIELDS[cls], _TYPE_TO_PROPS[cls] = process_node_fields(cls, ASTNode)

    _TYPE_TO_ALL_FIELDS[cls] = {
        **_TYPE_TO_CHILD_FIELDS[cls],
        **_TYPE_TO_PROPS[cls],
    }


def get_cls_all_fields(cls: type[ASTNode]) -> Mapping[Field, FieldTypeInfo]:
    """Returns a tuple of all fields of the given class."""
    if cls not in _TYPE_TO_ALL_FIELDS:
        _populate_type_dicts(cls)

    return _TYPE_TO_ALL_FIELDS[cls]


def get_cls_child_fields(cls: type[ASTNode]) -> Mapping[Field, FieldTypeInfo]:
    """Returns a tuple of all child fields of the given class.

    Raises:
        InvalidFieldTypes: If the class has fields with mixed ASTNode subclasses and regular types or
            unsupported child fileds types (e.g. mutable collections of ASTNode's).
    """
    if cls not in _TYPE_TO_CHILD_FIELDS:
        _populate_type_dicts(cls)

    return _TYPE_TO_CHILD_FIELDS[cls]


def get_cls_props(cls: type[ASTNode]) -> Mapping[Field, FieldTypeInfo]:
    """Returns a tuple of all properties of the given class.

    Raises:
        InvalidFieldTypes: If the class has fields with mixed ASTNode subclasses and regular types or
            unsupported child fileds types (e.g. mutable collections of ASTNode's).
    """
    if cls not in _TYPE_TO_PROPS:
        _populate_type_dicts(cls)

    return _TYPE_TO_PROPS[cls]
