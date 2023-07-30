from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from pyoak.helpers import (
    get_ast_node_child_fields,
    get_ast_node_properties,
    has_node_in_type,
    is_child_node,
    is_list,
    is_optional,
    is_sequence,
    is_tuple,
    is_union,
)
from pyoak.node import ASTNode


@dataclass
class HelpersTestSubASTNode1(ASTNode):
    prop: str
    child: ASTNode
    child_list: tuple[ASTNode, ...]


@dataclass
class HelpersTestSubASTNode2(ASTNode):
    prop: str
    child1: HelpersTestSubASTNode1
    child2: "HelpersTestSubASTNode3"


@dataclass
class HelpersTestSubASTNode3(ASTNode):
    prop: str
    child: HelpersTestSubASTNode1 | HelpersTestSubASTNode2


def test_is_union() -> None:
    assert is_union(int | str)
    assert is_union(Union[int, str])

    assert not is_union(int)
    assert not is_union(list[int | str])


def test_is_list() -> None:
    assert is_list(list[int])
    assert is_list(list[int | str])
    assert is_list(list[list[int]])

    assert is_list(List[int])
    assert is_list(List[int | str])
    assert is_list(List[list[int]])

    assert is_list(list)

    assert not is_list(Union[list, str])
    assert not is_list(int)
    assert not is_list(dict)
    assert not is_list(tuple)


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


def test_is_sequence() -> None:
    assert is_sequence(list[int])
    assert is_sequence(list[int | str])
    assert is_sequence(list[list[int]])

    assert is_sequence(List[int])
    assert is_sequence(List[int | str])
    assert is_sequence(List[list[int]])

    assert is_sequence(list)

    assert is_sequence(tuple[int])
    assert is_sequence(tuple[int | str])
    assert is_sequence(tuple[tuple[int]])

    assert is_sequence(Tuple[int])
    assert is_sequence(Tuple[int | str])
    assert is_sequence(Tuple[tuple[int]])

    assert is_sequence(tuple)

    assert not is_sequence(Union[tuple, str])
    assert not is_sequence(int)
    assert not is_sequence(dict)


def test_is_child_node() -> None:
    assert is_child_node(ASTNode)
    assert is_child_node(HelpersTestSubASTNode1)
    assert is_child_node(ASTNode | HelpersTestSubASTNode1)
    assert is_child_node(Union[ASTNode, HelpersTestSubASTNode1])
    assert is_child_node(list[ASTNode])
    assert is_child_node(list[ASTNode] | ASTNode)
    assert is_child_node(tuple[ASTNode])
    assert is_child_node(tuple[ASTNode, ...])
    assert is_child_node(tuple[ASTNode, ...] | ASTNode)
    assert is_child_node(List[ASTNode])
    assert is_child_node(List[ASTNode] | ASTNode)
    assert is_child_node(Tuple[ASTNode])
    assert is_child_node(Tuple[ASTNode, ...])
    assert is_child_node(Tuple[ASTNode, ...] | ASTNode)

    assert is_child_node(ASTNode | None)
    assert is_child_node(tuple[ASTNode | None, ...])

    assert is_child_node(ASTNode | str)

    assert not is_child_node(dict)
    assert not is_child_node(dict[str, ASTNode])
    assert is_child_node(dict[str, ASTNode] | ASTNode)
    assert not is_child_node(dict[ASTNode, str])
    assert is_child_node(dict[ASTNode, str] | ASTNode)

    assert not is_child_node(int)

    assert not is_child_node(list)
    assert not is_child_node(list[list[ASTNode]])

    assert not is_child_node(tuple)


def test_is_child_node_strict() -> None:
    assert is_child_node(ASTNode, strict=True)
    assert is_child_node(HelpersTestSubASTNode1, strict=True)
    assert is_child_node(ASTNode | HelpersTestSubASTNode1, strict=True)
    assert is_child_node(Union[ASTNode, HelpersTestSubASTNode1], strict=True)
    assert is_child_node(list[ASTNode], strict=True)
    assert is_child_node(list[ASTNode] | ASTNode, strict=True)
    assert is_child_node(tuple[ASTNode], strict=True)
    assert is_child_node(tuple[ASTNode, ...], strict=True)
    assert is_child_node(tuple[ASTNode, ...] | ASTNode, strict=True)
    assert is_child_node(List[ASTNode], strict=True)
    assert is_child_node(List[ASTNode] | ASTNode, strict=True)
    assert is_child_node(Tuple[ASTNode], strict=True)
    assert is_child_node(Tuple[ASTNode, ...], strict=True)
    assert is_child_node(Tuple[ASTNode, ...] | ASTNode, strict=True)

    assert is_child_node(ASTNode | None, strict=True)
    assert is_child_node(tuple[ASTNode | None, ...], strict=True)

    assert not is_child_node(ASTNode | str, strict=True)

    assert not is_child_node(dict, strict=True)
    assert not is_child_node(dict[str, ASTNode], strict=True)
    assert not is_child_node(dict[str, ASTNode] | ASTNode, strict=True)
    assert not is_child_node(dict[ASTNode, str], strict=True)
    assert not is_child_node(dict[ASTNode, str] | ASTNode, strict=True)

    assert not is_child_node(int, strict=True)

    assert not is_child_node(list, strict=True)
    assert not is_child_node(list[list[ASTNode]], strict=True)

    assert not is_child_node(tuple, strict=True)


def test_has_node_in_type() -> None:
    assert has_node_in_type(ASTNode)
    assert has_node_in_type(HelpersTestSubASTNode1)
    assert has_node_in_type(ASTNode | HelpersTestSubASTNode1)
    assert has_node_in_type(Union[ASTNode, HelpersTestSubASTNode1])
    assert has_node_in_type(list[ASTNode])
    assert has_node_in_type(list[ASTNode] | ASTNode)
    assert has_node_in_type(tuple[ASTNode])
    assert has_node_in_type(tuple[ASTNode, ...])
    assert has_node_in_type(tuple[ASTNode, ...] | ASTNode)
    assert has_node_in_type(List[ASTNode])
    assert has_node_in_type(List[ASTNode] | ASTNode)
    assert has_node_in_type(Tuple[ASTNode])
    assert has_node_in_type(Tuple[ASTNode, ...])
    assert has_node_in_type(Tuple[ASTNode, ...] | ASTNode)

    assert has_node_in_type(ASTNode | None)
    assert has_node_in_type(tuple[ASTNode | None, ...])

    assert has_node_in_type(ASTNode | str)

    assert not has_node_in_type(dict)
    assert has_node_in_type(dict[str, ASTNode])
    assert has_node_in_type(dict[str, ASTNode] | ASTNode)
    assert has_node_in_type(dict[ASTNode, str])
    assert has_node_in_type(dict[ASTNode, str] | ASTNode)

    assert not has_node_in_type(int)

    assert not has_node_in_type(list)
    assert has_node_in_type(list[list[ASTNode]])

    assert not has_node_in_type(tuple)


def test_get_ast_node_child_fields() -> None:
    assert get_ast_node_child_fields(ASTNode) == {}

    field_name_to_type_info = {
        f.name: type_info
        for f, type_info in get_ast_node_child_fields(HelpersTestSubASTNode1).items()
    }
    assert list(field_name_to_type_info.keys()) == [
        "child",
        "child_list",
    ]

    assert field_name_to_type_info["child"].sequence_type is None
    assert field_name_to_type_info["child"].types == (ASTNode,)
    assert field_name_to_type_info["child_list"].sequence_type is tuple
    assert field_name_to_type_info["child_list"].types == (ASTNode,)

    field_name_to_type_info = {
        f.name: type_info
        for f, type_info in get_ast_node_child_fields(HelpersTestSubASTNode2).items()
    }
    assert list(field_name_to_type_info.keys()) == [
        "child1",
        "child2",
    ]

    assert field_name_to_type_info["child1"].sequence_type is None
    assert field_name_to_type_info["child1"].types == (HelpersTestSubASTNode1,)
    assert field_name_to_type_info["child2"].sequence_type is None
    assert field_name_to_type_info["child2"].types == (HelpersTestSubASTNode3,)

    field_name_to_type_info = {
        f.name: type_info
        for f, type_info in get_ast_node_child_fields(HelpersTestSubASTNode3).items()
    }

    assert list(field_name_to_type_info.keys()) == [
        "child",
    ]

    assert field_name_to_type_info["child"].sequence_type is None
    assert field_name_to_type_info["child"].types == (
        HelpersTestSubASTNode1,
        HelpersTestSubASTNode2,
    )


def test_get_ast_node_properties() -> None:
    assert set([f.name for f in get_ast_node_properties(ASTNode).keys()]) == {
        "id",
        "content_id",
        "original_id",
        "id_collision_with",
        "origin",
    }
    assert set([f.name for f in get_ast_node_properties(HelpersTestSubASTNode1).keys()]) == {
        "id",
        "content_id",
        "original_id",
        "id_collision_with",
        "origin",
        "prop",
    }
