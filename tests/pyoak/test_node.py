from __future__ import annotations

import gc
import sys
from dataclasses import InitVar, dataclass, field
from typing import Any, ClassVar, Generator, Iterable, cast

import pytest
from deepdiff import DeepDiff
from mashumaro.types import SerializableType
from pyoak import config
from pyoak.error import InvalidTypes
from pyoak.match.error import ASTXpathDefinitionError
from pyoak.match.xpath import ASTXpath
from pyoak.node import (
    AST_SERIALIZE_DIALECT_KEY,
    NODE_REGISTRY,
    ASTNode,
    ASTSerializationDialects,
    NodeTraversalInfo,
)
from pyoak.origin import NO_ORIGIN, CodeOrigin, MemoryTextSource, get_code_range
from pyoak.serialize import TYPE_KEY, DataClassSerializeMixin
from pyoak.typing import Field

from tests.pyoak.conftest import ConfigFixtureProtocol

# get_child_nodes_with_field code is compiled on the fly on the first call
# since it is used in __post_init__ of ASTNode, it happens on the first
# instantiation of any ASTNode subclass
# make sure that codegen for the ASTNode class itself doesn't break the subclasses
_ = ASTNode()


@dataclass(frozen=True)
class ChildNode(ASTNode):
    attr: str


@dataclass(frozen=True)
class SubChildNode(ChildNode):
    pass


@dataclass(frozen=True)
class OtherNode(ASTNode):
    attr: str


@dataclass(frozen=True)
class NonCompareAttrNode(ASTNode):
    attr: str
    non_compare_attr: str = field(compare=False, default="test")
    non_init_attr: str = field(init=False, default="test")


@dataclass(frozen=True)
class ParentNode(ASTNode):
    attr: str
    single_child: ASTNode
    child_seq: tuple[ASTNode, ...]
    not_a_child_seq: tuple[str, ...]
    restricted_child: ChildNode | OtherNode | None = None


@dataclass(frozen=True)
class StaticFieldGettersTest(ASTNode):
    attr: str
    _hidden_attr: str
    child: ASTNode
    optional_child: ASTNode | None
    child_seq: tuple[ASTNode, ...]
    non_compare_attr: str = field(compare=False)
    non_init_attr: str = field(init=False, default="default")


@dataclass
class Custom(DataClassSerializeMixin):
    foo: int = 1


@dataclass(frozen=True)
class RuntimeTypeCheckNode(ASTNode):
    clsvar: ClassVar[str] = "test"
    str_prop: str
    tuple_prop: tuple[str, ...]
    cust_prop: Custom
    child: ChildNode
    child_seq: tuple[ASTNode, ...]
    initvar: InitVar[int]

    def __post_init__(self, initvar: int) -> None:
        return super().__post_init__()


origin = CodeOrigin(MemoryTextSource("test"), get_code_range(0, 1, 0, 3, 1, 3))


def _get_fname_set(fields: Iterable[Field]) -> set[str]:
    return {f.name for f in fields}


def test_default_origin() -> None:
    assert ChildNode("test").origin == NO_ORIGIN


@pytest.mark.skipif(
    not hasattr(SerializableType, "__slots__") or sys.version_info < (3, 11),
    reason="Mashumaro version doesn't support slots",
)
def test_slotted() -> None:
    @dataclass(frozen=True, slots=True)
    class SlottedNode(ASTNode):
        attr: str

    assert not hasattr(SlottedNode("test"), "__dict__")
    assert SlottedNode("test").origin == NO_ORIGIN


def test_id_handling(pyoak_config: ConfigFixtureProtocol) -> None:
    # Basic test
    node = ChildNode("test", origin=origin)
    assert node.id is not None
    assert len(node.id) == config.ID_DIGEST_SIZE * 2
    assert node == ChildNode.get(node.id)
    assert ASTNode.get(node.id) is None
    assert node == ASTNode.get_any(node.id)

    # Same content, different id
    new_node = ChildNode("test", origin=origin)
    assert node == new_node
    assert node.is_equal(new_node)
    assert node is not new_node
    assert node.id != new_node.id

    # Test changing digest size
    with pyoak_config(id_digest_size=config.ID_DIGEST_SIZE * 2):
        node = ChildNode("test2", origin=origin)
        assert node.id is not None
        assert len(node.id) == config.ID_DIGEST_SIZE * 2
        assert node == ChildNode.get(node.id)
        assert ASTNode.get(node.id) is None
        assert node == ASTNode.get_any(node.id)

    # Test handling of id collisions with nodes of same & different content
    # we are using a super small digest to guarantee collisions
    with pyoak_config(id_digest_size=1):
        node = ChildNode("test")
        # same content
        new_node = ChildNode("test")
        # different content that will collide
        other_node = ChildNode("L3C2089KVA")

        assert len({node.id, new_node.id, other_node.id}) == 3

        for n in (node, new_node, other_node):
            assert n is ASTNode.get_any(n.id)

        # the collision resolution is via a counter
        assert new_node.id == f"{node.id}_1"
        assert other_node.id == f"{node.id}_2"


def test_runtime_type_checks(pyoak_config: ConfigFixtureProtocol) -> None:
    # Without runtime checks
    with pyoak_config(runtime_checks=False):
        assert RuntimeTypeCheckNode(
            str_prop=1,  # type: ignore[arg-type]
            tuple_prop="s",  # type: ignore[arg-type]
            cust_prop=True,  # type: ignore[arg-type]
            child=OtherNode("1"),  # type: ignore[arg-type]
            child_seq=[ChildNode("1")],  # type: ignore[arg-type]
            initvar=1,
        )

    # With runtime checks
    with pyoak_config(runtime_checks=True):
        with pytest.raises(InvalidTypes) as excinfo:
            _ = RuntimeTypeCheckNode(
                str_prop=1,  # type: ignore[arg-type]
                tuple_prop="s",  # type: ignore[arg-type]
                cust_prop=True,  # type: ignore[arg-type]
                child=OtherNode("1"),  # type: ignore[arg-type]
                child_seq=[ChildNode("1")],  # type: ignore[arg-type]
                initvar=1,
            )

        assert _get_fname_set(excinfo.value.invalid_fields) == {
            "str_prop",
            "tuple_prop",
            "cust_prop",
            "child",
            "child_seq",
        }

        @dataclass(frozen=True)
        class SNode(ASTNode):
            ninit: str = field(init=False, default=1)  # type: ignore[assignment]

        with pytest.raises(InvalidTypes) as excinfo:
            _ = SNode()

        assert _get_fname_set(excinfo.value.invalid_fields) == {"ninit"}


def test_content_id() -> None:
    ch_nodes: list[ASTNode] = []
    ch_count = 2
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    different_origin = CodeOrigin(MemoryTextSource("different"), get_code_range(0, 1, 0, 3, 1, 3))
    # Last child is the same but with different origin
    ch_nodes.append(ChildNode("1", origin=different_origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_seq=tuple(ch_nodes[1:]),
        not_a_child_seq=tuple(str(i) for i in range(10)),
        origin=origin,
    )
    duplicate = parent.duplicate()

    assert ch_nodes[1] != ch_nodes[2]
    assert ch_nodes[1] is not ch_nodes[2]
    assert ch_nodes[1].id != ch_nodes[2].id
    assert ch_nodes[1].content_id == ch_nodes[2].content_id

    assert parent == duplicate
    assert parent is not duplicate
    assert parent.id != duplicate.id
    assert parent.content_id == duplicate.content_id

    # Test content id invariant to hidden and non-comparable fields
    one = NonCompareAttrNode(attr="1", non_compare_attr="one", origin=origin)
    one_dup = one.duplicate()
    two = NonCompareAttrNode(attr="1", non_compare_attr="two", origin=origin)
    two_dup = two.duplicate()

    assert one == two
    assert one.content_id == two.content_id == one_dup.content_id == two_dup.content_id


def test_serialization() -> None:
    child = ChildNode("1")

    mid_node = ParentNode(
        attr="Mid",
        single_child=child,
        child_seq=(),
        not_a_child_seq=(),
    )

    parent = ParentNode(
        attr="Test",
        single_child=mid_node,
        # use the same child node twice
        child_seq=(child,),
        not_a_child_seq=(),
        origin=origin,
    )

    serialized_dict = parent.as_dict()
    deserialized = ParentNode.as_obj(serialized_dict)

    # Class should be present on the root node
    assert TYPE_KEY in serialized_dict

    # Should be the same object
    assert parent is deserialized

    # Now clear the registry and try to deserialize
    NODE_REGISTRY.clear()

    deserialized = ParentNode.as_obj(serialized_dict)

    # Should be a new object
    assert parent is not deserialized
    # but equal
    assert parent == deserialized
    # Internally, the child nodes should be the same object
    assert isinstance(deserialized.single_child, ParentNode)
    assert deserialized.single_child.single_child is deserialized.child_seq[0]


def test_no_origin_serialization() -> None:
    node = ChildNode("test", origin=NO_ORIGIN)

    assert ChildNode.from_json(node.to_json()) == node


def test_ast_explorer_serialization_dialect() -> None:
    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_seq=tuple(ch_nodes[1:]),
        not_a_child_seq=tuple(str(i) for i in range(10)),
        origin=origin,
    )

    serialized_dialect_dict = parent.as_dict(
        serialization_options={
            AST_SERIALIZE_DIALECT_KEY: ASTSerializationDialects.AST_EXPLORER,
        }
    )
    serialized_regular = parent.as_dict()

    ddiff = DeepDiff(
        serialized_dialect_dict,
        serialized_regular,
        exclude_regex_paths=[r"\['_children'\]", r"root\['class'\]"],
    )
    # Check that the only difference is the _children key
    assert ddiff.to_dict() == {}

    # Check that the _children key is correct
    assert "_children" in serialized_dialect_dict
    assert serialized_dialect_dict["_children"] == [
        "single_child",
        "child_seq",
        "restricted_child",
    ]

    # Check that the _children key is not present in the regular serialization
    assert "_children" not in serialized_regular


def test_ast_test_serialization_dialect() -> None:
    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_seq=tuple(ch_nodes[1:]),
        not_a_child_seq=tuple(str(i) for i in range(10)),
        origin=origin,
    )

    serialized_dialect_dict = parent.as_dict(
        serialization_options={
            AST_SERIALIZE_DIALECT_KEY: ASTSerializationDialects.AST_TEST,
        }
    )

    assert "origin" in serialized_dialect_dict
    assert "source" in serialized_dialect_dict["origin"]
    assert serialized_dialect_dict["origin"]["source"] == {
        TYPE_KEY: "Source",
        "source_uri": "",
        "source_type": "",
    }
    assert "child_seq" in serialized_dialect_dict
    for child in serialized_dialect_dict["child_seq"]:
        assert "origin" in child
        assert "source" in child["origin"]
        assert child["origin"]["source"] == {
            TYPE_KEY: "Source",
            "source_uri": "",
            "source_type": "",
        }

    # Try to deserialize
    ParentNode.as_obj(serialized_dialect_dict)


def test_to_msgpack() -> None:
    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_seq=tuple(ch_nodes[1:]),
        not_a_child_seq=tuple(str(i) for i in range(10)),
        origin=origin,
    )

    serialized = parent.to_msgpck()
    deserialized = ParentNode.from_msgpck(serialized)

    assert parent is deserialized


def test_get() -> None:
    n = SubChildNode("test")
    o = ChildNode("test")

    assert SubChildNode.get(n.id) is n
    assert ChildNode.get(n.id) is None
    assert ChildNode.get(n.id, o) is o
    assert ChildNode.get("bogus", o) is o
    assert ChildNode.get(n.id, strict=False) is n
    assert ChildNode.get("bogus", strict=False) is None
    assert ChildNode.get(n.id, o, strict=False) is n
    assert ChildNode.get("bogus", o, strict=False) is o


def test_get_any() -> None:
    n = SubChildNode("test")

    n_id = n.id
    assert ASTNode.get_any(n_id) is n
    assert ASTNode.get_any("bogus") is None
    assert ASTNode.get_any("bogus", n) is n

    # Test that after deletion, registry is cleared
    del n
    gc.collect()
    assert ASTNode.get_any(n_id) is None


def test_detach() -> None:
    # Create a node and register it
    node = ChildNode("node1")
    assert ChildNode.get(node.id) is node

    # Test successful detach
    assert node.detach() is True
    assert ChildNode.get(node.id) is None

    # Test unsuccessful detach
    assert node.detach() is False
    assert ChildNode.get(node.id) is None


def test_replace() -> None:
    # Create a node and register it
    # Use model with non compare attr to test for id collisions
    node = NonCompareAttrNode("node1", non_compare_attr="one")
    assert NonCompareAttrNode.get(node.id) is node

    # Test successful replace
    new_node = node.replace(attr="node2")
    assert new_node is not node
    assert new_node.id != node.id
    assert new_node.attr == "node2"
    assert new_node.non_compare_attr == "one"
    assert new_node.non_init_attr == "test"
    assert NonCompareAttrNode.get(node.id) is None
    assert NonCompareAttrNode.get(new_node.id) is new_node

    # Test replace with content that produces the same id
    node = new_node
    new_node = node.replace(non_compare_attr="two")
    assert new_node is not node
    assert new_node.id == node.id
    assert new_node.is_equal(node)
    assert new_node.non_compare_attr == "two"
    assert new_node.non_init_attr == "test"
    assert NonCompareAttrNode.get(new_node.id) is new_node

    # Test unsuccessful replace with exception
    with pytest.raises(ValueError):
        new_node.replace(non_compare_attr="three", non_init_attr="three")

    # validate that the node is not replaced
    assert NonCompareAttrNode.get(new_node.id) is new_node

    # Test replacing node already not in the registry (detached)
    node = new_node
    node.detach()
    assert NonCompareAttrNode.get(node.id) is None
    new_node = node.replace(attr="node3")
    assert new_node is not node
    assert new_node.id != node.id
    assert new_node.attr == "node3"
    assert new_node.non_compare_attr == "two"
    assert new_node.non_init_attr == "test"
    assert NonCompareAttrNode.get(node.id) is None
    assert NonCompareAttrNode.get(new_node.id) is new_node


def test_duplicate() -> None:
    # Test simple replacement
    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_seq=tuple(ch_nodes[1:]),
        not_a_child_seq=tuple(str(i) for i in range(10)),
        origin=origin,
    )
    dup_parent = parent.duplicate()
    assert dup_parent is not parent
    assert dup_parent == parent


def test_children() -> None:
    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_seq=tuple(ch_nodes[1:]),
        not_a_child_seq=(),
        origin=origin,
    )

    assert list(parent.children) == ch_nodes


def test_equality() -> None:
    # Testing the full equality (==) and content-wise equality (is_equal)

    # Test euqal itself
    one = ChildNode("1", origin=origin)
    assert one.is_equal(one)
    assert one == one

    # Test different types & values
    two = ChildNode("2", origin=origin)
    other_type_ast = OtherNode("1", origin=origin)
    assert not one.is_equal("Test not ASTNode")
    assert not one.is_equal(other_type_ast)
    assert not one.is_equal(two)
    assert one != "Test not ASTNode"
    assert one != other_type_ast
    assert one != two

    # Test different origins
    other_origin = CodeOrigin(MemoryTextSource("test2"), get_code_range(0, 1, 0, 4, 1, 4))
    other_origin_one = ChildNode("1", origin=other_origin)
    assert one != other_origin_one
    assert one.is_equal(other_origin_one)

    # Same content and origin
    collided_one = ChildNode("1", origin=origin)
    assert one == collided_one
    assert one.is_equal(collided_one)

    # Test with duplicate
    duplicated_one = one.duplicate()
    assert one == duplicated_one
    assert one.is_equal(duplicated_one)

    # Test deep comparison with children

    parent = ParentNode(
        attr="Parent",
        single_child=ChildNode("single_child", origin=origin),
        child_seq=(
            ChildNode("child_list1", origin=origin),
            ChildNode("child_list2", origin=origin),
        ),
        not_a_child_seq=(),
        origin=origin,
    )

    other_parent = ParentNode(
        attr="Parent",
        single_child=ChildNode("single_child", origin=origin),
        child_seq=(
            ChildNode("child_list1", origin=origin),
            ChildNode("child_list2", origin=other_origin),  # changed origin
        ),
        not_a_child_seq=(),
        origin=origin,
    )
    assert parent != other_parent
    assert parent.is_equal(other_parent)

    other_parent = ParentNode(
        attr="Parent",
        single_child=ChildNode("single_child", origin=origin),
        child_seq=(
            ChildNode("child_list1", origin=origin),
            ChildNode("child_list2-changed", origin=origin),
        ),
        not_a_child_seq=(),
        origin=origin,
    )
    assert parent != other_parent
    assert not parent.is_equal(other_parent)

    # Check non-compare attrs (should not be compared)
    hidone = NonCompareAttrNode(attr="1", non_compare_attr="one", origin=origin)
    hidtwo = NonCompareAttrNode(attr="1", non_compare_attr="two", origin=origin)
    assert hidone.is_equal(hidtwo)
    assert hidone == hidtwo


def test_to_properties_dict() -> None:
    single_child = ChildNode("single_child", origin=origin)
    child_list = ChildNode("child_list1", origin=origin)
    parent = ParentNode(
        attr="Parent",
        single_child=single_child,
        child_seq=(child_list,),
        not_a_child_seq=("1",),
        origin=origin,
    )

    assert parent.to_properties_dict() == dict(
        attr="Parent",
        not_a_child_seq=("1",),
    )


def test_to_tree() -> None:
    n = ChildNode("test")
    assert n.to_tree().root is n


def test_iter_child_fields(clean_ser_types) -> None:
    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_seq=tuple(ch_nodes[1:]),
        not_a_child_seq=(),
        origin=origin,
    )

    assert list(parent.iter_child_fields()) == [
        (
            ch_nodes[0],
            ParentNode.__dataclass_fields__["single_child"],
        ),
        (
            tuple(ch_nodes[1:]),
            ParentNode.__dataclass_fields__["child_seq"],
        ),
        (
            None,
            ParentNode.__dataclass_fields__["restricted_child"],
        ),
    ]

    # iter_child_fields is compiled on the fly
    # Test codegen of a parent class doesn't break the subclasses
    @dataclass(frozen=True)
    class BaseNode(ASTNode):
        child: ChildNode

    @dataclass(frozen=True)
    class SubNode(BaseNode):
        child_seq: tuple[ChildNode, ...]

    # trigger codegen for get_properties
    _ = list(BaseNode(ChildNode("1")).iter_child_fields())

    assert list(SubNode(ChildNode("1"), (ChildNode("2"),)).iter_child_fields()) == [
        (ChildNode("1"), SubNode.__dataclass_fields__["child"]),
        ((ChildNode("2"),), SubNode.__dataclass_fields__["child_seq"]),
    ]


@dataclass(frozen=True)
class BTreeNode(ASTNode):
    v: int
    left: BTreeNode | None
    right: BTreeNode | None


def _btree_from_value_dict(d: dict[int, Any]) -> BTreeNode | None:
    v, children = next(iter(d.items()))
    left_d: dict[int, Any] | None = None
    right_d: dict[int, Any] | None = None

    if children:
        lr = list(children.items())
        if len(lr) > 2:
            raise ValueError("Too many children")

        k1, v1 = lr[0]
        if k1 > 0:
            left_d = {k1: v1}

        if len(lr) == 2:
            k2, v2 = lr[1]
            right_d = {k2: v2}

    left = _btree_from_value_dict(left_d) if left_d is not None else None
    right = _btree_from_value_dict(right_d) if right_d is not None else None
    return BTreeNode(v, left, right)


def _node_traversal_info_to_values(res: Generator[NodeTraversalInfo, None, None]) -> list[int]:
    return [cast(BTreeNode, n.node).v for n in res]


def test_walkers() -> None:
    # DFS, top-down tree
    dfs_td = cast(
        BTreeNode,
        _btree_from_value_dict(
            {1: {2: {3: {4: {}, 5: {6: {}}}, 7: {8: {}}}, 9: {10: {11: {-1: {}, 12: {}}}}}}
        ),
    )
    # no filter, no prune
    assert _node_traversal_info_to_values(dfs_td.dfs()) == list(range(2, 13))

    # pruned
    def _prune(node: NodeTraversalInfo) -> bool:
        return cast(BTreeNode, node.node).v == 2

    assert _node_traversal_info_to_values(dfs_td.dfs(prune=_prune)) == [2, *list(range(9, 13))]

    # filtered
    def _filter(node: NodeTraversalInfo) -> bool:
        return cast(BTreeNode, node.node).v % 2 == 0

    assert _node_traversal_info_to_values(dfs_td.dfs(filter=_filter)) == [
        v for v in range(2, 13) if v % 2 == 0
    ]

    # filtered & pruned
    assert _node_traversal_info_to_values(dfs_td.dfs(filter=_filter, prune=_prune)) == [2, 10, 12]

    # DFS, bottom-up tree
    dfs_bu = cast(
        BTreeNode,
        _btree_from_value_dict(
            {12: {7: {4: {1: {}, 3: {2: {}}}, 6: {5: {}}}, 11: {10: {9: {-1: {}, 8: {}}}}}}
        ),
    )

    assert _node_traversal_info_to_values(dfs_bu.dfs(bottom_up=True)) == list(range(1, 12))

    # pruned
    def _prune(node: NodeTraversalInfo) -> bool:  # type: ignore[no-redef]
        return cast(BTreeNode, node.node).v == 7

    assert _node_traversal_info_to_values(dfs_bu.dfs(prune=_prune, bottom_up=True)) == list(
        range(7, 12)
    )

    # filtered
    assert _node_traversal_info_to_values(dfs_bu.dfs(filter=_filter, bottom_up=True)) == [
        v for v in range(1, 12) if v % 2 == 0
    ]

    # filtered & pruned
    assert _node_traversal_info_to_values(
        dfs_bu.dfs(filter=_filter, prune=_prune, bottom_up=True)
    ) == [8, 10]

    # BFS tree (always top-down)
    bfs_r = cast(
        BTreeNode,
        _btree_from_value_dict(
            {1: {2: {4: {7: {}, 8: {11: {}}}, 5: {9: {}}}, 3: {6: {10: {12: {}}}}}}
        ),
    )

    assert _node_traversal_info_to_values(bfs_r.bfs()) == list(range(2, 13))

    # pruned
    def _prune(node: NodeTraversalInfo) -> bool:  # type: ignore[no-redef]
        return cast(BTreeNode, node.node).v == 2

    assert _node_traversal_info_to_values(bfs_r.bfs(prune=_prune)) == [2, 3, 6, 10, 12]

    # filtered
    assert _node_traversal_info_to_values(bfs_r.bfs(filter=_filter)) == [
        v for v in range(2, 13) if v % 2 == 0
    ]

    # filtered & pruned
    assert _node_traversal_info_to_values(bfs_r.bfs(filter=_filter, prune=_prune)) == [2, 6, 10, 12]


def test_gather() -> None:
    ch = ChildNode("1")
    ch1 = ChildNode("1")
    sub_ch = SubChildNode("1")
    other = OtherNode("1")
    other1 = OtherNode("1")

    mid = ParentNode(
        attr="Mid",
        single_child=ch,
        child_seq=(sub_ch, other),
        not_a_child_seq=(),
        origin=origin,
    )

    root = ParentNode(
        attr="Root",
        single_child=mid,
        child_seq=(ch1, other1),
        not_a_child_seq=(),
    )

    assert set(root.gather(ChildNode)) == {ch, ch1, sub_ch}
    assert set(root.gather((ChildNode, OtherNode))) == {ch, ch1, sub_ch, other, other1}


def test_find() -> None:
    ch = ChildNode("1")
    ch1 = ChildNode("1")
    sub_ch = SubChildNode("1")
    other = OtherNode("1")
    other1 = OtherNode("1")

    mid = ParentNode(
        attr="Mid",
        single_child=ch,
        child_seq=(sub_ch, other),
        not_a_child_seq=(),
        origin=origin,
    )

    root = ParentNode(
        attr="Root",
        single_child=mid,
        child_seq=(ch1, other1),
        not_a_child_seq=(),
    )

    # Only minimal test. Full in test_xpath.py
    assert root.find("//ChildNode") is ch
    assert root.find(ASTXpath("//ChildNode")) is ch
    assert root.find("/OtherNode") is None

    with pytest.raises(ASTXpathDefinitionError):
        root.find("NonExistentClass")


def test_findall() -> None:
    ch = ChildNode("1")
    ch1 = ChildNode("1")
    sub_ch = SubChildNode("1")
    other = OtherNode("1")
    other1 = OtherNode("1")

    mid = ParentNode(
        attr="Mid",
        single_child=ch,
        child_seq=(sub_ch, other),
        not_a_child_seq=(),
        origin=origin,
    )

    root = ParentNode(
        attr="Root",
        single_child=mid,
        child_seq=(ch1, other1),
        not_a_child_seq=(),
    )

    # Only minimal test. Full in test_xpath.py
    assert set(root.findall("//ChildNode")) == {ch, ch1, sub_ch}
    assert set(root.findall(ASTXpath("//ChildNode"))) == {ch, ch1, sub_ch}
    assert set(root.findall("/OtherNode")) == set()

    with pytest.raises(ASTXpathDefinitionError):
        next(root.findall("NonExistentClass"))


def test_get_property_fields() -> None:
    def_props = {"attr", "_hidden_attr", "non_compare_attr", "non_init_attr"}
    assert _get_fname_set(StaticFieldGettersTest.get_property_fields()) == def_props

    assert _get_fname_set(
        StaticFieldGettersTest.get_property_fields(skip_non_compare=True)
    ) == def_props - {"non_compare_attr"}

    assert _get_fname_set(
        StaticFieldGettersTest.get_property_fields(skip_non_init=True)
    ) == def_props - {"non_init_attr"}

    assert _get_fname_set(
        StaticFieldGettersTest.get_property_fields(skip_id=False)
    ) == def_props | {"id"}

    assert _get_fname_set(
        StaticFieldGettersTest.get_property_fields(skip_content_id=False)
    ) == def_props | {"content_id"}

    assert _get_fname_set(
        StaticFieldGettersTest.get_property_fields(skip_origin=False)
    ) == def_props | {"origin"}


def test_get_child_fields() -> None:
    assert _get_fname_set(StaticFieldGettersTest.get_child_fields().keys()) == {
        "child",
        "optional_child",
        "child_seq",
    }


def test_get_properties() -> None:
    all_props = {
        "attr_v": StaticFieldGettersTest.__dataclass_fields__["attr"],
        "hidden_v": StaticFieldGettersTest.__dataclass_fields__["_hidden_attr"],
        "non_compare_v": StaticFieldGettersTest.__dataclass_fields__["non_compare_attr"],
        "default": StaticFieldGettersTest.__dataclass_fields__["non_init_attr"],
    }

    node = StaticFieldGettersTest(
        attr="attr_v",
        _hidden_attr="hidden_v",
        child=ChildNode("1"),
        optional_child=None,
        child_seq=(),
        non_compare_attr="non_compare_v",
        origin=origin,
    )

    assert list(node.get_properties()) == list(all_props.items())

    assert list(node.get_properties(skip_id=False)) == [
        (node.id, StaticFieldGettersTest.__dataclass_fields__["id"]),
        *list(all_props.items()),
    ]

    assert list(node.get_properties(skip_content_id=False)) == [
        (node.content_id, StaticFieldGettersTest.__dataclass_fields__["content_id"]),
        *list(all_props.items()),
    ]

    assert list(node.get_properties(skip_origin=False)) == [
        (origin, StaticFieldGettersTest.__dataclass_fields__["origin"]),
        *list(all_props.items()),
    ]

    no_non_compare = all_props.copy()
    no_non_compare.pop("non_compare_v")

    assert list(node.get_properties(skip_non_compare=True)) == list(no_non_compare.items())

    no_non_init = all_props.copy()
    no_non_init.pop("default")

    assert list(node.get_properties(skip_non_init=True)) == list(no_non_init.items())

    # get_properties code is compiled on the fly on the first call
    # Test codegen of a parent class doesn't break the subclasses
    @dataclass(frozen=True)
    class BaseNode(ASTNode):
        attr: str

    @dataclass(frozen=True)
    class SubNode(BaseNode):
        sattr: str

    # trigger codegen for get_properties
    _ = list(BaseNode("1").get_properties())

    assert list(SubNode("1", "2").get_properties()) == [
        ("1", SubNode.__dataclass_fields__["attr"]),
        ("2", SubNode.__dataclass_fields__["sattr"]),
    ]


def test_get_child_nodes(clean_ser_types) -> None:
    child = ChildNode("1")

    mid_node = ParentNode(
        attr="Mid",
        single_child=child,
        child_seq=(),
        not_a_child_seq=(),
    )

    parent = ParentNode(
        attr="Test",
        single_child=mid_node,
        # use the same child node twice
        child_seq=(child,),
        not_a_child_seq=(),
        origin=origin,
    )

    assert list(parent.get_child_nodes()) == [mid_node, child]

    # get_child_nodes code is compiled on the fly on the first call
    # Test codegen of a parent class doesn't break the subclasses
    @dataclass(frozen=True)
    class BaseNode(ASTNode):
        child: ChildNode

    @dataclass(frozen=True)
    class SubNode(BaseNode):
        schild: ChildNode

    # trigger codegen for get_child_nodes
    _ = list(BaseNode(ChildNode("1")).get_child_nodes())

    assert list(SubNode(ChildNode("1"), ChildNode("2")).get_child_nodes()) == [
        ChildNode("1"),
        ChildNode("2"),
    ]


def test_get_child_nodes_with_field(clean_ser_types) -> None:
    child = ChildNode("1")

    mid_node = ParentNode(
        attr="Mid",
        single_child=child,
        child_seq=(),
        not_a_child_seq=(),
    )

    parent = ParentNode(
        attr="Test",
        single_child=mid_node,
        # use the same child node twice
        child_seq=(child,),
        not_a_child_seq=(),
        origin=origin,
    )

    assert list(parent.get_child_nodes_with_field()) == [
        (mid_node, ParentNode.__dataclass_fields__["single_child"], None),
        (child, ParentNode.__dataclass_fields__["child_seq"], 0),
    ]

    # get_child_nodes_with_field code is compiled on the fly on the first call
    # Test codegen of a parent class doesn't break the subclasses
    @dataclass(frozen=True)
    class BaseNode(ASTNode):
        child: ChildNode

    @dataclass(frozen=True)
    class SubNode(BaseNode):
        schild: ChildNode

    # trigger codegen for get_child_nodes_with_field
    _ = list(BaseNode(ChildNode("1")).get_child_nodes_with_field())

    assert list(SubNode(ChildNode("1"), ChildNode("2")).get_child_nodes_with_field()) == [
        (ChildNode("1"), SubNode.__dataclass_fields__["child"], None),
        (ChildNode("2"), SubNode.__dataclass_fields__["schild"], None),
    ]
