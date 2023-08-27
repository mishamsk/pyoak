from __future__ import annotations

import sys
from dataclasses import KW_ONLY, InitVar, dataclass
from functools import partial
from typing import (
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Literal,
    Mapping,
    NewType,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import pytest
from pyoak.error import InvalidFieldAnnotations
from pyoak.typing import (
    FieldTypeInfo,
    InvalidTypeReason,
    check_annotations,
    get_field_types,
    has_check_type_in_type,
    is_classvar,
    is_collection,
    is_dataclass_kw_only,
    is_initvar,
    is_instance,
    is_literal,
    is_mutable_collection,
    is_new_type,
    is_optional,
    is_tuple,
    is_type_generic,
    is_union,
    is_valid_child_field_type,
    is_valid_property_type,
    process_node_fields,
    unwrap_newtype,
)


@dataclass(frozen=True)
class MockASTNode:
    pass


class Custom:
    pass


MyInt = NewType("MyInt", int)


@dataclass(frozen=True)
class TypTestSubASTNode1(MockASTNode):
    prop: str
    child: MockASTNode
    child_list: tuple[MockASTNode, ...]


@dataclass(frozen=True)
class TypTestSubASTNode2(MockASTNode):
    prop: str
    child1: TypTestSubASTNode1
    child2: "TypTestSubASTNode3"


@dataclass(frozen=True)
class TypTestSubASTNode3(MockASTNode):
    prop: str
    child: TypTestSubASTNode1 | TypTestSubASTNode2


def test_is_literal() -> None:
    assert is_literal(Literal[1])
    assert is_literal(Literal["foo"])
    assert is_literal(Literal[1, "foo"])
    assert is_literal(Literal)

    assert not is_literal(int)
    assert not is_literal(list)


def test_is_type_generic() -> None:
    assert is_type_generic(type)
    assert is_type_generic(Type)
    assert is_type_generic(Type[int])
    assert is_type_generic(Type[MockASTNode])

    assert not is_type_generic(int)
    assert not is_type_generic(str)
    assert not is_type_generic(list)


def test_is_union() -> None:
    assert is_union(int | str)
    assert is_union(Union[int, str])

    assert not is_union(int)
    assert not is_union(list[int | str])


def test_is_tuple() -> None:
    assert is_tuple(tuple[int])
    assert is_tuple(tuple[int | str])
    assert is_tuple(tuple[tuple[int]])

    assert is_tuple(Tuple[int])
    assert is_tuple(Tuple[int | str])
    assert is_tuple(Tuple[tuple[int]])

    assert is_tuple(tuple)

    assert not is_tuple(Union[tuple, str])
    assert not is_tuple(int)
    assert not is_tuple(dict)
    assert not is_tuple(list)


def test_is_optional() -> None:
    assert is_optional(int | None)
    assert is_optional(Union[int, None])
    assert is_optional(Optional[int])
    assert is_optional(Optional[Union[int, None]])

    assert not is_optional(int)
    assert not is_optional(list[int | str])


def test_is_collection() -> None:
    assert is_collection(list[int])
    assert is_collection(list[int | str])
    assert is_collection(list[list[int]])

    assert is_collection(List[int])
    assert is_collection(List[int | str])
    assert is_collection(List[list[int]])

    assert is_collection(list)

    assert is_collection(tuple[int])
    assert is_collection(tuple[int | str])
    assert is_collection(tuple[tuple[int]])

    assert is_collection(Tuple[int])
    assert is_collection(Tuple[int | str])
    assert is_collection(Tuple[tuple[int]])

    assert is_collection(tuple)
    assert is_collection(dict)

    assert not is_collection(Union[tuple, str])
    assert not is_collection(int)
    assert not is_collection(str)


def test_is_classvar() -> None:
    assert is_classvar(ClassVar)
    assert is_classvar(ClassVar[int])
    assert is_classvar(ClassVar[str])
    assert is_classvar(ClassVar[Union[int, str]])

    assert not is_classvar(int)
    assert not is_classvar(list[str])


def test_is_initvar() -> None:
    assert is_initvar(InitVar)
    assert is_initvar(InitVar[int])
    assert is_initvar(InitVar[str])
    assert is_initvar(InitVar[Union[int, str]])

    assert not is_initvar(int)
    assert not is_initvar(list[str])


def test_is_new_type() -> None:
    assert is_new_type(NewType("MyInt", int))
    assert is_new_type(NewType("MyList", list))
    assert is_new_type(NewType("MyDerivative", NewType("MyList", list)))

    assert not is_new_type(int)
    assert not is_new_type(NewType)


def test_unwrap_newtype() -> None:
    MyInt = NewType("MyInt", int)
    assert unwrap_newtype(MyInt) is int

    MyList = NewType("MyList", list)
    assert unwrap_newtype(MyList) is list

    MyDerivative = NewType("MyDerivative", MyList)
    assert unwrap_newtype(MyDerivative) is list

    assert unwrap_newtype(int) is int
    assert unwrap_newtype(list) is list


def test_is_dataclass_kw_only() -> None:
    assert is_dataclass_kw_only(KW_ONLY)
    assert not is_dataclass_kw_only(int)


def test_is_mutable_collection() -> None:
    assert is_mutable_collection(list)
    assert is_mutable_collection(list[int])
    assert is_mutable_collection(list[int | str])
    assert is_mutable_collection(list[list[int]])
    assert is_mutable_collection(List[int])

    assert is_mutable_collection(dict)
    assert is_mutable_collection(dict[str, int])
    assert is_mutable_collection(Dict[str, int])

    assert is_mutable_collection(set)
    assert is_mutable_collection(set[int])
    assert is_mutable_collection(set[int | str])
    assert is_mutable_collection(Set[int])

    assert not is_mutable_collection(Union[list, str])
    assert not is_mutable_collection(int)
    assert not is_mutable_collection(tuple)
    assert not is_mutable_collection(frozenset)
    assert not is_mutable_collection(Mapping)


def test_has_check_type_in_type() -> None:
    test_func = partial(has_check_type_in_type, check_type=MockASTNode)

    assert test_func(MockASTNode)
    assert test_func(TypTestSubASTNode1)
    assert test_func(MockASTNode | TypTestSubASTNode1)
    assert test_func(Union[MockASTNode, TypTestSubASTNode1])
    assert test_func(list[MockASTNode])
    assert test_func(list[MockASTNode] | MockASTNode)
    assert test_func(tuple[MockASTNode])
    assert test_func(tuple[MockASTNode, ...])
    assert test_func(tuple[MockASTNode, ...] | MockASTNode)
    assert test_func(List[MockASTNode])
    assert test_func(List[MockASTNode] | MockASTNode)
    assert test_func(Tuple[MockASTNode])
    assert test_func(Tuple[MockASTNode, ...])
    assert test_func(Tuple[MockASTNode, ...] | MockASTNode)

    assert test_func(MockASTNode | None)
    assert test_func(tuple[MockASTNode | None, ...])

    assert test_func(MockASTNode | str)

    assert not test_func(dict)
    assert test_func(dict[str, MockASTNode])
    assert test_func(dict[str, MockASTNode] | MockASTNode)
    assert test_func(dict[MockASTNode, str])
    assert test_func(dict[MockASTNode, str] | MockASTNode)

    assert not test_func(int)

    assert not test_func(list)
    assert test_func(list[list[MockASTNode]])

    assert not test_func(tuple)

    assert test_func(Sequence[tuple[Mapping[str, MockASTNode], int, bool]])


def test_is_valid_child_field_type() -> None:
    test_func = partial(is_valid_child_field_type, node_type=MockASTNode)

    # Test all valid combinations

    # Single type
    assert test_func(MockASTNode) == InvalidTypeReason.OK
    assert test_func(TypTestSubASTNode1) == InvalidTypeReason.OK

    # Union
    assert test_func(MockASTNode | TypTestSubASTNode1) == InvalidTypeReason.OK
    assert test_func(Union[MockASTNode, TypTestSubASTNode1]) == InvalidTypeReason.OK

    # Optional
    assert test_func(MockASTNode | None) == InvalidTypeReason.OK
    assert test_func(Optional[MockASTNode]) == InvalidTypeReason.OK
    assert test_func(Optional[MockASTNode | TypTestSubASTNode1]) == InvalidTypeReason.OK
    assert test_func(Union[MockASTNode, TypTestSubASTNode1, None]) == InvalidTypeReason.OK

    # Tuple
    assert test_func(tuple[MockASTNode]) == InvalidTypeReason.OK
    assert test_func(tuple[MockASTNode, ...]) == InvalidTypeReason.OK
    assert test_func(Tuple[MockASTNode]) == InvalidTypeReason.OK
    assert test_func(tuple[MockASTNode | TypTestSubASTNode1]) == InvalidTypeReason.OK
    assert test_func(tuple[MockASTNode | TypTestSubASTNode1, ...]) == InvalidTypeReason.OK
    assert (
        test_func(tuple[MockASTNode, TypTestSubASTNode1, TypTestSubASTNode2])
        == InvalidTypeReason.OK
    )
    assert test_func(Tuple[MockASTNode]) == InvalidTypeReason.OK
    assert test_func(Tuple[MockASTNode, ...]) == InvalidTypeReason.OK

    # Now test all invalid combinations

    # No lists
    assert test_func(list[MockASTNode]) == InvalidTypeReason.MUT_SEQ
    assert test_func(list[MockASTNode] | MockASTNode) == InvalidTypeReason.NON_NODE_TYPE
    assert test_func(List[MockASTNode]) == InvalidTypeReason.MUT_SEQ
    assert test_func(List[MockASTNode] | MockASTNode) == InvalidTypeReason.NON_NODE_TYPE
    assert test_func(list[list[MockASTNode]]) == InvalidTypeReason.MUT_SEQ

    # No mixing sequence & non-sequence
    assert test_func(tuple[MockASTNode, ...] | MockASTNode) == InvalidTypeReason.NON_NODE_TYPE
    assert test_func(Tuple[MockASTNode, ...] | MockASTNode) == InvalidTypeReason.NON_NODE_TYPE

    # No optionals in sequences
    assert test_func(tuple[MockASTNode | None, ...]) == InvalidTypeReason.OPT_IN_SEQ

    # No nested sequences
    assert test_func(tuple[tuple[MockASTNode]]) == InvalidTypeReason.NON_NODE_TYPE

    # No mixing types
    assert test_func(MockASTNode | str) == InvalidTypeReason.NON_NODE_TYPE

    # No dicts
    assert test_func(dict) == InvalidTypeReason.MUT_SEQ
    assert test_func(dict[str, MockASTNode]) == InvalidTypeReason.MUT_SEQ
    assert test_func(dict[str, MockASTNode] | MockASTNode) == InvalidTypeReason.NON_NODE_TYPE
    assert test_func(dict[MockASTNode, str]) == InvalidTypeReason.MUT_SEQ
    assert test_func(dict[MockASTNode, str] | MockASTNode) == InvalidTypeReason.NON_NODE_TYPE

    # No simple types, no generics
    assert test_func(int) == InvalidTypeReason.NON_NODE_TYPE
    assert test_func(list) == InvalidTypeReason.MUT_SEQ
    assert test_func(tuple) == InvalidTypeReason.EMPTY_TUPLE


def test_is_valid_property_type() -> None:
    # Simple types & non mutable collections are ok
    assert is_valid_property_type(int)
    assert is_valid_property_type(str)
    assert is_valid_property_type(float)
    assert is_valid_property_type(bool)
    assert is_valid_property_type(tuple)
    assert is_valid_property_type(tuple[int])
    assert is_valid_property_type(tuple[int | str])
    assert is_valid_property_type(tuple[int | str, ...])
    assert is_valid_property_type(Tuple[int, str])
    assert is_valid_property_type(frozenset)
    assert is_valid_property_type(frozenset[int])
    assert is_valid_property_type(FrozenSet[int])
    assert is_valid_property_type(int | str)
    assert is_valid_property_type(Union[int, str])
    assert is_valid_property_type(NewType("MyInt", int))

    # Custom types are ok as well
    assert is_valid_property_type(Custom)

    # Mutable collections are not ok, including via NewType
    assert not is_valid_property_type(dict)
    assert not is_valid_property_type(dict[str, int])
    assert not is_valid_property_type(Dict[str, int])
    assert not is_valid_property_type(set)
    assert not is_valid_property_type(set[int])
    assert not is_valid_property_type(Set[int])
    assert not is_valid_property_type(list)
    assert not is_valid_property_type(list[int])
    assert not is_valid_property_type(List[int])
    assert not is_valid_property_type(NewType("MyList", list))


def test_check_annotations() -> None:
    @dataclass(frozen=True)
    class Good(MockASTNode):
        prop: str
        child: MockASTNode
        child_list: tuple[MockASTNode, ...]

    @dataclass(frozen=True)
    class GoodExt(MockASTNode):
        prop: str
        tuple_prop: tuple[str, ...]
        cust_prop: Custom
        clsvar: ClassVar[str]
        child: MockASTNode
        _: KW_ONLY
        child_list: tuple[MockASTNode, ...]
        initvar: InitVar[MockASTNode]

    # Good class
    assert check_annotations(Good, MockASTNode)

    if sys.version_info < (3, 11):
        with pytest.raises(TypeError):
            check_annotations(GoodExt, MockASTNode)
    else:
        assert check_annotations(GoodExt, MockASTNode)

    # Postponed annotations
    @dataclass(frozen=True)
    class Pre(MockASTNode):
        child: "Post"

    assert not check_annotations(Pre, MockASTNode)

    @dataclass(frozen=True)
    class Post(MockASTNode):
        attr: str

    @dataclass(frozen=True)
    class BadClass(MockASTNode):
        mut_prop: list[str]
        mixed: str | MockASTNode
        bseq: list[MockASTNode]
        nested: tuple[tuple[MockASTNode]]

    # Invalid child type
    with pytest.raises(InvalidFieldAnnotations) as err:
        # this will indirectly call check_annotations
        check_annotations(BadClass, MockASTNode)

    assert tuple(err.value.invalid_annotations) == (
        ("mut_prop", "A mutable collection in type", list[str]),
        ("mixed", InvalidTypeReason.NON_NODE_TYPE.value, str | MockASTNode),
        ("bseq", InvalidTypeReason.MUT_SEQ.value, list[MockASTNode]),
        ("nested", InvalidTypeReason.NON_NODE_TYPE.value, tuple[tuple[MockASTNode]]),
    )


def test_get_field_types() -> None:
    @dataclass
    class Person:
        name: str
        age: int
        height: float
        is_student: bool
        grades: list[int]
        friends: set[str]
        preferences: dict[str, str]
        favorite_color: MyInt
        children: tuple[MockASTNode, ...]

    field_types = get_field_types(Person)

    assert field_types == {
        Person.__dataclass_fields__["name"]: str,
        Person.__dataclass_fields__["age"]: int,
        Person.__dataclass_fields__["height"]: float,
        Person.__dataclass_fields__["is_student"]: bool,
        Person.__dataclass_fields__["grades"]: list[int],
        Person.__dataclass_fields__["friends"]: set[str],
        Person.__dataclass_fields__["preferences"]: dict[str, str],
        Person.__dataclass_fields__["favorite_color"]: int,
        Person.__dataclass_fields__["children"]: tuple[MockASTNode, ...],
    }


def test_process_node_fields() -> None:
    @dataclass(frozen=True)
    class GoodExt(MockASTNode):
        prop: str
        tuple_prop: tuple[str, ...]
        cust_prop: Custom
        clsvar: ClassVar[str]
        child: MockASTNode
        _: KW_ONLY
        child_list: tuple[MockASTNode, ...]
        initvar: InitVar[MockASTNode]

    # Good class
    if sys.version_info < (3, 11):
        with pytest.raises(TypeError):
            child_fields, props = process_node_fields(GoodExt, MockASTNode)
    else:
        child_fields, props = process_node_fields(GoodExt, MockASTNode)

        assert child_fields == {
            GoodExt.__dataclass_fields__["child"]: FieldTypeInfo(False, MockASTNode),
            GoodExt.__dataclass_fields__["child_list"]: FieldTypeInfo(
                True, tuple[MockASTNode, ...]
            ),
        }

        assert props == {
            GoodExt.__dataclass_fields__["prop"]: FieldTypeInfo(False, str),
            GoodExt.__dataclass_fields__["tuple_prop"]: FieldTypeInfo(True, tuple[str, ...]),
            GoodExt.__dataclass_fields__["cust_prop"]: FieldTypeInfo(False, Custom),
        }

    # Postponed annotations
    @dataclass(frozen=True)
    class Pre(MockASTNode):
        child: "Post"

    with pytest.raises(NameError) as err:
        process_node_fields(Pre, MockASTNode)

    assert str(err.value) == "name 'Post' is not defined"

    @dataclass(frozen=True)
    class Post(MockASTNode):
        attr: str

    @dataclass(frozen=True)
    class BadClass(MockASTNode):
        mut_prop: list[str]
        mixed: str | MockASTNode
        bseq: list[MockASTNode]
        nested: tuple[tuple[MockASTNode]]

    # Invalid child type
    with pytest.raises(InvalidFieldAnnotations) as err1:
        # this will indirectly call check_annotations
        process_node_fields(BadClass, MockASTNode)

    assert tuple(err1.value.invalid_annotations) == (
        ("mut_prop", "A mutable collection in type", list[str]),
        ("mixed", InvalidTypeReason.NON_NODE_TYPE.value, str | MockASTNode),
        ("bseq", InvalidTypeReason.MUT_SEQ.value, list[MockASTNode]),
        ("nested", InvalidTypeReason.NON_NODE_TYPE.value, tuple[tuple[MockASTNode]]),
    )


def test_is_instance() -> None:
    assert is_instance(42, int)
    assert is_instance(42, int | str)
    assert is_instance(3.14, float)
    assert is_instance(3 + 4j, complex)
    assert is_instance("hello", str)
    assert is_instance(True, bool)
    assert is_instance([], list)
    assert is_instance([1, 2, 3], List[int])
    assert is_instance([1, 2, 3], List[Union[int, str]])
    assert is_instance([1, 2, 3], List[Literal[1, 2, 3]])
    assert is_instance((1, 2, 3), tuple)
    assert is_instance((1, 2, 3), Tuple[int, int, int])
    assert is_instance((1, 2, 3), Tuple[int, ...])
    assert is_instance({1, 2, 3}, set)
    assert is_instance({1, 2, 3}, Set[int])
    assert is_instance({1, 2, 3}, Set[Union[int, str]])
    assert is_instance(frozenset((1, 2, 3)), FrozenSet[int])
    assert is_instance(frozenset((1, 2, 3)), FrozenSet[Union[int, str]])
    assert is_instance(None, Optional[int])
    assert is_instance(None, Optional[str])
    assert is_instance("hello", Optional[str])
    assert is_instance(1, Union[int, str])
    assert is_instance("hello", Union[int, str])
    assert is_instance(1, Literal[1])
    assert is_instance("hello", Literal[1, "hello"])
    assert is_instance(
        TypTestSubASTNode1(prop="p", child=MockASTNode(), child_list=()), MockASTNode
    )

    assert not is_instance(3.14, int)
    assert not is_instance("hello", int)
    assert not is_instance(True, int)
    assert not is_instance([1, 2, 3], tuple)
    assert not is_instance((1, 2, 3), List[int])
    assert not is_instance({1, 2, 3}, Dict[int, str])
    assert not is_instance(None, int)
    assert not is_instance(None, str)
    assert not is_instance(1.0, Union[int, str])
    assert not is_instance("hello", Union[int, float])
    assert not is_instance(2, Literal[1])
    assert not is_instance("world", Literal["hello"])
