from __future__ import annotations

import logging
from collections.abc import Collection, MutableMapping, MutableSequence, MutableSet
from dataclasses import KW_ONLY, InitVar, fields
from dataclasses import Field as DataClassField
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    ClassVar,
    Literal,
    Mapping,
    NamedTuple,
    NewType,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .error import InvalidFieldAnnotations

logger = logging.getLogger(__name__)

# to make mypy happy
Field = DataClassField[Any]


class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field]]


cached = lru_cache(maxsize=None)


@cached
def is_literal(type_: Any) -> bool:
    orig = get_origin(type_)
    if orig is None:
        return type_ is Literal

    return orig is Literal


@cached
def is_type_generic(type_: Any) -> bool:
    orig = get_origin(type_)
    if orig is None:
        return type_ in (type, Type)

    return orig in (type, Type)


@cached
def is_union(type_: Any) -> bool:
    orig = get_origin(type_)
    if orig is Union:
        return True
    try:
        from types import UnionType

        return isinstance(type_, UnionType)
    except TypeError:
        return False


@cached
def is_tuple(type_: Any) -> bool:
    orig = get_origin(type_)
    if orig is tuple or orig is Tuple:
        return True
    try:
        return issubclass(type_, (Tuple, tuple))  # type: ignore[arg-type]
    except TypeError:
        return False


@cached
def is_optional(type_: Any) -> bool:
    orig = get_origin(type_)
    if orig is Optional:
        return True

    if not is_union(type_):
        return False

    args = get_args(type_)
    return any(a is type(None) for a in args)


@cached
def is_collection(type_: Any) -> bool:
    """Return True if the type is a collection."""
    orig = get_origin(type_)

    try:
        if orig is not None:
            return issubclass(orig, Collection) and not issubclass(orig, str)

        return issubclass(type_, Collection) and not issubclass(type_, str)
    except TypeError:
        return False


@cached
def is_classvar(type_: Any) -> bool:
    """Return True if the type is a ClassVar."""
    return type_ is ClassVar or get_origin(type_) is ClassVar


@cached
def is_initvar(type_: Any) -> bool:
    """Return True if the type is a InitVar."""
    return type_ is InitVar or type(type_) is InitVar


@cached
def is_new_type(type_: Any) -> bool:
    """Return True if the type is a NewType."""
    try:
        return isinstance(type_, NewType)
    except TypeError:
        return False


@cached
def unwrap_newtype(type_: type) -> type:
    """Return the type of a NewType.

    For types that are derived from a NewType, this will recursively unwrap the type until the
    underlying type is found.

    """
    while isinstance(type_, NewType):
        type_ = type_.__supertype__

    return type_


@cached
def is_dataclass_kw_only(type_: Any) -> bool:
    """Return True if the type is a dataclass kw_only."""
    return type_ is KW_ONLY


@cached
def is_mutable_collection(type_: Any) -> bool:
    """Return True if the type is a mutable collection."""

    orig = get_origin(type_)
    check_type = orig or type_
    try:
        return issubclass(check_type, (MutableSequence, MutableMapping, MutableSet))
    except TypeError:
        return False


@cached
def has_check_type_in_type(type_: type, check_type: type) -> bool:
    """Return True if a given type is a subclass of check_type or a complex type that has a subclass
    of check_type among it's arguments."""

    try:
        if issubclass(type_, check_type):
            return True
    except TypeError:
        pass

    return any(has_check_type_in_type(t, check_type) for t in get_args(type_))


class InvalidTypeReason(Enum):
    OK = "Ok"
    OPT_IN_SEQ = "Optional type in sequence"
    MUT_SEQ = "Mutable sequence"
    NON_NODE_TYPE = "Non-node type or not a tuple sequence"
    EMPTY_TUPLE = "Empty tuple"
    OTHER = "Other"


@cached
def _is_valid_child_field_type(
    type_: type, node_base_type: type, allow_sequence: bool
) -> InvalidTypeReason:
    """Inner function for is_valid_child_field_type.

    It doesn't
    catch TypeErrors (which will happen on some type annotations such as
    generics). Since anything more complex than `tuple[Node | AnotherNode, ...]`
    is not a valid child field type, we can stop at the first TypeError
    caught in the outer function.

    """

    if not allow_sequence and is_optional(type_):
        # We do not allow optionals within sequences
        # So check this early
        return InvalidTypeReason.OPT_IN_SEQ

    args = get_args(type_)
    if is_optional(type_) or is_union(type_):
        # For plain unions we only allow direct subclasses of node_type
        # So easy check

        try:
            if not all(issubclass(t, node_base_type) for t in args if t is not type(None)):
                return InvalidTypeReason.NON_NODE_TYPE
        except TypeError:
            return InvalidTypeReason.NON_NODE_TYPE

        return InvalidTypeReason.OK

    if allow_sequence and is_tuple(type_):
        # For tuples we allow any combination of node_type subclasses
        if len(args) == 0:
            return InvalidTypeReason.EMPTY_TUPLE

        if len(args) == 2 and args[1] is Ellipsis:
            return _is_valid_child_field_type(args[0], node_base_type, False)
        else:
            if not all(
                _is_valid_child_field_type(t, node_base_type, False) == InvalidTypeReason.OK
                for t in args
            ):
                return InvalidTypeReason.NON_NODE_TYPE

            return InvalidTypeReason.OK

    if is_mutable_collection(type_):
        return InvalidTypeReason.MUT_SEQ

    if not issubclass(type_, node_base_type):
        return InvalidTypeReason.NON_NODE_TYPE

    return InvalidTypeReason.OK


@cached
def is_valid_child_field_type(type_: type, node_type: type) -> InvalidTypeReason:
    """Return True if the type is a calid child field type defition.

    This means either:
    - a subclass of node_type
    _ a union of subclasses of node_type and maybe None
    - a tuple of subclasses of node_type

    Args:
        type_ (type): The type to check
        node_type (type): The node type to check against

    Returns:
        InvalidTypeReason: The reason why the type is invalid or OK if it is valid

    """
    try:
        return _is_valid_child_field_type(type_, node_type, True)
    except TypeError:
        return InvalidTypeReason.OTHER


@cached
def is_valid_property_type(type_: Any) -> bool:
    """Checks if a given type is a valid property type.

    This function only disallows mutable collections at this point.

    Custom objects are not marked mutable as we leave it to the user to make sure mutable stuff
    doesn't end up in immutables nodes.

    """
    if is_collection(type_):
        if is_mutable_collection(type_):
            return False

        agrs = get_args(type_)

        return all(is_valid_property_type(t) for t in agrs)

    if is_union(type_):
        return all(is_valid_property_type(t) for t in get_args(type_))

    if is_new_type(type_):
        return is_valid_property_type(unwrap_newtype(type_))

    return True


# Do not cache, to allow checking after ForwardRefs are resolved
def check_annotations(type_: type, node_base_type: type) -> bool:
    """This function will mimic how stdlib processes type annotations in dataclasses to check for
    invalid type annotations. I.e. fields that have node_type in type, but are not valid child field
    types.

    However, since ForwardRefs may not be resolved at this point,
    it may not check some annotations that are invalid.

    Args:
        type_ (type): The class to check
        node_base_type (type): The base node type to check against

    Returns:
        bool: True if all annotations are valid, False if forward refs are not resolved

    Raises:
        InvalidFieldTypes: If any field has an invalid type annotation

    """
    try:
        cls_annotations = get_type_hints(type_)
        incorrect_fields: list[tuple[str, str, type]] = []

        for field_name, field_type in cls_annotations.items():
            if is_classvar(field_type) or is_initvar(field_type) or is_dataclass_kw_only(type_):
                continue

            if has_check_type_in_type(field_type, node_base_type):
                # Possible child field
                res = is_valid_child_field_type(field_type, node_base_type)

                if res != InvalidTypeReason.OK:
                    incorrect_fields.append((field_name, res.value, field_type))
            else:
                # Property
                if not is_valid_property_type(field_type):
                    incorrect_fields.append(
                        (field_name, "A mutable collection in type", field_type)
                    )
        if incorrect_fields:
            raise InvalidFieldAnnotations(incorrect_fields)
    except NameError:
        # Some forward refs may not be resolved at this point
        # So we can't check them
        return False
    except TypeError as err:
        if err.args[0].startswith(
            "Forward references must evaluate to types. Got <dataclasses._KW_ONLY_TYPE"
        ):
            # In py3.10 using KW_ONLY with postponed annotations will raise a TypeError
            # re-raise with a nicer message
            raise TypeError(
                "Postponed annotations in combination with KW_ONLY special type "
                f"are not supported by pyoak. Please, avoid one of them for {type_}"
                " or upgrade to Python 3.11."
            ) from err

    return True


class FieldTypeInfo(NamedTuple):
    is_collection: bool
    resolved_type: type


@cached
def get_type_info(type_: Any, allow_sequence: bool = True) -> FieldTypeInfo:
    return FieldTypeInfo(is_collection(type_), type_)


def get_field_types(type_: type[DataclassInstance]) -> dict[Field, Any]:
    """Return the type of a dataclass field.

    This recursively function unwraps NewType to the underlying type.

    """
    ret: dict[Field, Any] = {}

    for field in fields(type_):
        f_type = field.type
        if isinstance(f_type, str):
            f_type = get_type_hints(type_).get(field.name)

        if f_type is None:
            raise RuntimeError(f"Could not determine type of field {field.name} for type {type_}")

        # Unwrap newtypes to not deal with them later
        if is_new_type(f_type):  # type: ignore[arg-type]
            f_type = unwrap_newtype(f_type)  # type: ignore[arg-type]

        ret[field] = f_type

    return ret


def process_node_fields(
    type_: type[DataclassInstance], node_base_type: type
) -> tuple[Mapping[Field, FieldTypeInfo], Mapping[Field, FieldTypeInfo]]:
    """Return the child fields of an AST node."""

    incorrect_fields: list[tuple[str, str, type]] = []
    child_fields: dict[Field, FieldTypeInfo] = {}
    props: dict[Field, FieldTypeInfo] = {}

    for f, ftype in get_field_types(type_).items():
        if has_check_type_in_type(ftype, node_base_type):
            # Possible child field
            res = is_valid_child_field_type(ftype, node_base_type)
            if res == InvalidTypeReason.OK:
                child_fields[f] = get_type_info(ftype)
            else:
                incorrect_fields.append((f.name, res.value, ftype))
        else:
            # Property
            if is_valid_property_type(ftype):
                props[f] = get_type_info(ftype)
            else:
                incorrect_fields.append((f.name, "A mutable collection in type", ftype))

    if incorrect_fields:
        raise InvalidFieldAnnotations(incorrect_fields)

    return child_fields, props


def is_instance(value: Any, type_: Any) -> bool:
    """Return True if `obj` is an instance of `type_`.

    Adapted from dacite https://github.com/konradhalas/dacite

    This function assumes only valid types are passed to it,
    i.e. it is only called after check_annotations/process_node_fields
    has been called.

    """

    # We do not want Python implicit isinstance(True, int) == True
    if type_ is int and value is True or value is False:
        return False

    try:
        # As described in PEP 484 - section: "The numeric tower"
        if (type_ in [float, complex] and isinstance(value, (int, float))) or isinstance(
            value, type_
        ):
            return True
    except TypeError:
        pass
    if type_ == Any:
        return True

    if is_optional(type_) and value is None:
        return True

    if is_union(type_):
        return any(is_instance(value, t) for t in get_args(type_))

    if is_collection(type_):
        orig = get_origin(type_)

        if orig is None:
            # Non generic collection
            return isinstance(value, type_)

        # Generic collection
        if not isinstance(value, orig):
            # Not an instance of the generic collection
            return False

        args = get_args(type_)

        # Tuples are special
        if is_tuple(type_):
            if len(args) == 0:
                return len(value) == 0
            elif len(args) == 2 and args[1] is Ellipsis:
                return all(is_instance(item, args[0]) for item in value)
            else:
                if len(args) != len(value):
                    return False
                return all(is_instance(item, item_type) for item, item_type in zip(value, args))

        # Non-tuple collection with no args, assume True
        if not args:
            return True

        if len(args) > 1:
            raise RuntimeError(f"Unexpected collection type {type_}. Please, report a bug.")

        return all(is_instance(item, args[0]) for item in value)

    if is_literal(type_):
        return value in get_args(type_)

    if is_type_generic(type_):
        args = get_args(type_)
        if args:
            return issubclass(value, get_args(type_)[0])

    return False
