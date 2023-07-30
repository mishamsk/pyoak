from __future__ import annotations

import logging
import types
from dataclasses import Field as DataClassField
from dataclasses import fields
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

if TYPE_CHECKING:
    from pyoak.node import ASTNode

logger = logging.getLogger(__name__)

# to make mypy happy
Field = DataClassField[Any]


def is_union(type_: Any) -> bool:
    orig = get_origin(type_)
    if orig is Union:
        return True
    try:
        return isinstance(type_, types.UnionType)
    except TypeError:
        return False


def is_list(type_: Any) -> bool:
    orig = get_origin(type_)
    if orig is list or orig is List:
        return True
    try:
        return issubclass(type_, (List, list))
    except TypeError:
        return False


def is_tuple(type_: Any) -> bool:
    orig = get_origin(type_)
    if orig is tuple or orig is Tuple:
        return True
    try:
        return issubclass(type_, (Tuple, tuple))  # type: ignore[arg-type]
    except TypeError:
        return False


def is_optional(type_: Any) -> bool:
    orig = get_origin(type_)
    if orig is Optional:
        return True

    if not is_union(type_):
        return False

    args = get_args(type_)
    return any(a is type(None) for a in args)


def is_sequence(type_: Any) -> bool:
    """Return True if the type is a supported sequence (list or tuple)."""
    return is_list(type_) or is_tuple(type_)


def has_node_in_type(type_: Any) -> bool:
    """Return True if the type include any subclasses of ASTNode."""

    # Dynamically import to avoid circular imports
    from pyoak.node import ASTNode

    try:
        if issubclass(type_, ASTNode):
            return True
    except TypeError:
        pass

    return any(has_node_in_type(t) for t in get_args(type_))


def is_child_node(type_: Any, allow_sequence: bool = True, strict: bool = False) -> bool:
    """Return True if the type is a child node."""

    # Dynamically import to avoid circular imports
    from pyoak.node import ASTNode

    arg_check_func = any if not strict else all

    if is_optional(type_):
        orig = get_origin(type_)
        args = get_args(type_)

        if orig is Optional:
            return arg_check_func(is_child_node(t) for t in args)
        else:
            return arg_check_func(is_child_node(t) for t in args if t is not type(None))

    if is_union(type_):
        return arg_check_func(is_child_node(t) for t in get_args(type_))

    if allow_sequence and is_list(type_):
        args = get_args(type_)
        if len(args) == 0:
            return False

        return arg_check_func(is_child_node(t, allow_sequence=False) for t in args)

    if allow_sequence and is_tuple(type_):
        args = get_args(type_)
        if len(args) == 0:
            return False

        if len(args) == 2 and args[1] is Ellipsis:
            return is_child_node(args[0], allow_sequence=False)
        else:
            return arg_check_func(is_child_node(t, allow_sequence=False) for t in args)

    try:
        return issubclass(type_, ASTNode)
    except TypeError:
        return False


class ChildFieldTypeInfo(NamedTuple):
    is_optional: bool
    sequence_type: Type[list[Any]] | Type[tuple[Any, ...]] | None
    types: tuple[Type[ASTNode], ...]


def get_node_type_info(type_: Any, allow_sequence: bool = True) -> ChildFieldTypeInfo:
    """Get a tuple of ASTNode subclasses from a type annotation.

    For lists and tuples, the types of the elements are returned.

    """

    # Dynamically import to avoid circular imports
    from pyoak.node import ASTNode

    args = get_args(type_)
    if is_optional(type_):
        return ChildFieldTypeInfo(True, None, tuple(t for t in args if issubclass(t, ASTNode)))

    if is_union(type_):
        return ChildFieldTypeInfo(False, None, tuple(t for t in args if issubclass(t, ASTNode)))

    if is_list(type_) and allow_sequence:
        if len(args) == 0:
            return ChildFieldTypeInfo(False, None, ())

        return ChildFieldTypeInfo(
            False,
            list,
            tuple(
                chain.from_iterable(get_node_type_info(t, allow_sequence=False).types for t in args)
            ),
        )

    if is_tuple(type_) and allow_sequence:
        if len(args) == 0:
            return ChildFieldTypeInfo(False, None, ())

        if len(args) == 2 and args[1] is Ellipsis:
            return ChildFieldTypeInfo(
                False, tuple, get_node_type_info(args[0], allow_sequence=False).types
            )
        else:
            return ChildFieldTypeInfo(
                False,
                tuple,
                tuple(
                    chain.from_iterable(
                        get_node_type_info(t, allow_sequence=False).types for t in args
                    )
                ),
            )

    try:
        if issubclass(type_, ASTNode):
            return ChildFieldTypeInfo(False, None, (type_,))
        else:
            return ChildFieldTypeInfo(False, None, ())
    except TypeError:
        return ChildFieldTypeInfo(False, None, ())


def get_field_types(type_: Type[ASTNode]) -> dict[str, Any]:
    """Return the type of a dataclass field."""
    ret: dict[str, Any] = {}

    for field in fields(type_):
        f_type = field.type
        if isinstance(f_type, str):
            f_type = get_type_hints(type_).get(field.name)

        if f_type is None:
            raise RuntimeError(f"Could not determine type of field {field.name} for type {type_}")

        ret[field.name] = f_type

    return ret


def get_ast_node_child_fields(
    type_: Type[ASTNode],
) -> Mapping[Field, ChildFieldTypeInfo]:
    """Return the child fields of an AST node."""
    all_fields = {f.name: f for f in fields(type_)}
    return {
        all_fields[fname]: get_node_type_info(type_)
        for fname, type_ in get_field_types(type_).items()
        if is_child_node(type_)
    }


def get_ast_node_properties(type_: Type[ASTNode]) -> Mapping[Field, Type[Any]]:
    """Return the property fields of an AST node."""
    all_fields = {f.name: f for f in fields(type_)}
    return {
        all_fields[fname]: type_
        for fname, type_ in get_field_types(type_).items()
        if not is_child_node(type_)
    }
