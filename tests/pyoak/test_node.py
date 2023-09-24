from __future__ import annotations

import gc
from dataclasses import InitVar, dataclass, field, replace
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
    ASTNode,
    ASTSerializationDialects,
    NodeTraversalInfo,
)
from pyoak.origin import NO_ORIGIN, CodeOrigin, MemoryTextSource, get_code_range
from pyoak.registry import _REF_TO_NODE
from pyoak.serialize import TYPE_KEY, DataClassSerializeMixin
from pyoak.typing import Field

from tests.pyoak.conftest import ConfigFixtureProtocol

# get_child_nodes_with_field code is compiled on the fly on the first call
# since it is used in __post_init__ of ASTNode, it happens on the first
# instantiation of any ASTNode subclass
# this indirectly tests that codegen for the ASTNode class itself doesn't break the subclasses
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

    def __post_init__(self, initvar: int, attached: bool) -> None:
        return super().__post_init__(attached)


origin = CodeOrigin(MemoryTextSource("test"), get_code_range(0, 1, 0, 3, 1, 3))


def _get_fname_set(fields: Iterable[Field]) -> set[str]:
    return {f.name for f in fields}


def test_default_origin() -> None:
    assert ChildNode("test").origin == NO_ORIGIN


@pytest.mark.skipif(
    not hasattr(SerializableType, "__slots__"),
    reason="Mashumaro version doesn't support slots",
)
def test_slotted() -> None:
    @dataclass(frozen=True, slots=True)
    class SlottedNode(ASTNode):
        attr: str

    assert not hasattr(SlottedNode("test"), "__dict__")
    assert SlottedNode("test").origin == NO_ORIGIN


def test_repr() -> None:
    # Makre sure custom repr is set on subclasses
    # and compare formatting to expected.
    # Test on nested tree to make sure it works (repr goal is to avoid printing the whole tree)
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
        attached=True,
    )

    assert repr(child) == (
        f"Leaf<ChildNode>(Props:id={child.id}, "
        f"content_id={child.content_id}, attr='1';origin=NoSource@NoOrigin)"
    )
    assert repr(parent) == (
        f"Sub-tree<ParentNode>(Props:id={parent.id}, "
        f"content_id={parent.content_id}, ref={parent.ref}"
        ", attr='Test', not_a_child_seq=();"
        "Children:single_child=<ParentNode>, child_seq=<ChildNode>...(1 total),"
        f" restricted_child=None;origin=<memory>@{origin.fqn})"
    )


def test_id_handling(pyoak_config: ConfigFixtureProtocol) -> None:
    # Basic test
    node = ChildNode("test", origin=origin)
    assert node.id is not None
    assert len(node.id) == config.ID_DIGEST_SIZE * 2

    # Same content, same id
    new_node = ChildNode("test", origin=origin)
    assert node == new_node
    assert node.is_equal(new_node)
    assert node is not new_node
    assert node.id == new_node.id
    assert hash(node) == hash(new_node)

    # Test changing digest size
    with pyoak_config(id_digest_size=config.ID_DIGEST_SIZE * 2):
        node = ChildNode("test2", origin=origin)
        assert node.id is not None
        assert len(node.id) == config.ID_DIGEST_SIZE * 2

    # Test handling of id collisions with different content
    # we are using a super small digest to guarantee collisions
    with pyoak_config(id_digest_size=1):
        node = ChildNode("test")
        # different content that will collide
        collided_node = ChildNode("L3C2089KVA")

        assert node.id == collided_node.id
        assert node is not collided_node
        # But since we are using both id & content_id for equality
        # the latter must be not equal
        assert node.content_id != collided_node.content_id
        assert node != collided_node
        assert not node.is_equal(collided_node)

    # Test that changing the field order in class definition doesn't change the id
    @dataclass(frozen=True)
    class IdOrderTest(ASTNode):
        attr1: str
        attr2: str
        ch1: ChildNode
        ch2: ChildNode

    onode = IdOrderTest(attr1="1", attr2="2", ch1=ChildNode("3"), ch2=ChildNode("4"))

    # Redefine the class with different field order
    @dataclass(frozen=True)
    class IdOrderTest(ASTNode):  # type: ignore[no-redef]
        ch2: ChildNode
        attr2: str
        ch1: ChildNode
        attr1: str

    sonode = IdOrderTest(attr1="1", attr2="2", ch1=ChildNode("3"), ch2=ChildNode("4"))
    assert onode.id == sonode.id
    assert onode.content_id == sonode.content_id


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
    duplicate = replace(parent)

    assert ch_nodes[1] != ch_nodes[2]
    assert ch_nodes[1] is not ch_nodes[2]
    assert ch_nodes[1].id != ch_nodes[2].id
    assert ch_nodes[1].content_id == ch_nodes[2].content_id

    assert parent == duplicate
    assert parent is not duplicate
    assert parent.content_id == duplicate.content_id

    # Test content id invariant to non-comparable fields
    one = NonCompareAttrNode(attr="1", non_compare_attr="one", origin=origin)
    two = NonCompareAttrNode(attr="1", non_compare_attr="two", origin=origin)

    assert one.content_id == two.content_id


def test_ref_handling(pyoak_config: ConfigFixtureProtocol) -> None:
    # By default ref is not craeted
    node = ChildNode("test", origin=origin)
    assert node.ref is None
    assert node not in _REF_TO_NODE.values()

    # Create via init var
    rnode = ChildNode("test", origin=origin, attached=True)

    # content, id, hash must match and should be equal
    # ref doesn't influence equality
    assert rnode.ref is not None
    assert rnode is ChildNode.get(rnode.ref)
    assert rnode.ref in _REF_TO_NODE
    assert _REF_TO_NODE[rnode.ref] is rnode
    assert rnode == node
    assert hash(rnode) == hash(node)
    assert rnode.id == node.id
    assert rnode.content_id == node.content_id

    # Attach via attach method
    nref = node.attach()

    # Nodes should still be equal, but refs should be different
    assert node.ref is not None
    assert node.ref == nref
    assert node is ChildNode.get(node.ref)
    assert node.ref in _REF_TO_NODE
    assert _REF_TO_NODE[node.ref] is node
    assert rnode == node
    assert hash(rnode) == hash(node)
    assert rnode.id == node.id
    assert rnode.content_id == node.content_id
    assert rnode.ref != node.ref


def test_ref_or_raise() -> None:
    node = ChildNode("test")

    with pytest.raises(AttributeError):
        node.ref_or_raise

    nref = node.attach()

    assert node.ref_or_raise == nref


def test_is_attached() -> None:
    node = ChildNode("test")
    assert not node.is_attached

    node.attach()
    assert node.is_attached

    # duplication should drop the ref
    assert not replace(node).is_attached


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


def test_serialization() -> None:
    child = ChildNode("1")

    # Basic test
    assert (
        child.to_json()
        == f'{{"{TYPE_KEY}":"ChildNode","id":"{child.id}","content_id":"{child.content_id}","origin":{{}},"attr":"1"}}'
    )

    rchild = ChildNode("1", attached=True)
    assert (
        rchild.to_json()
        == f'{{"{TYPE_KEY}":"ChildNode","id":"{rchild.id}","content_id":"{rchild.content_id}","origin":{{}},"attr":"1","__node_ref__":"{rchild.ref}"}}'
    )

    # Test roundtrip for a tree. Without attached nodes
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

    # Should be equal but not the same object
    assert parent is not deserialized
    assert parent == deserialized

    # Internally, the child nodes should not be the same object (we loose the fact they were the same)
    assert isinstance(deserialized.single_child, ParentNode)
    assert deserialized.single_child.single_child is not deserialized.child_seq[0]
    assert deserialized.single_child.single_child == deserialized.child_seq[0]

    # Now the same, but with some nodes attached
    parent = ParentNode(
        attr="Test",
        single_child=mid_node,
        # use the same child node twice
        child_seq=(rchild,),
        not_a_child_seq=(),
        origin=origin,
    )

    serialized_dict = parent.as_dict()

    # The dict should contain the ref for parent and the rchild
    assert "__node_ref__" not in serialized_dict
    assert serialized_dict["child_seq"][0]["__node_ref__"] == rchild.ref

    # Deserializing should return the exact same object for all nodes
    # that were attached
    parent_deserialized = ParentNode.as_obj(serialized_dict)

    assert parent_deserialized is not parent
    assert parent_deserialized == parent
    assert hash(parent_deserialized) == hash(parent)
    assert parent_deserialized.single_child is not parent.single_child
    # Same object, because it was attached
    assert parent_deserialized.child_seq[0] is parent.child_seq[0]


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

    assert parent is not deserialized
    assert parent == deserialized


def test_get() -> None:
    # Test properly attached
    n = SubChildNode("test", attached=True)
    o = ChildNode("test", attached=True)

    assert n.ref is not None
    assert SubChildNode.get(n.ref) is n
    assert ChildNode.get(n.ref) is None
    assert ChildNode.get(n.ref, o) is o
    assert ChildNode.get("bogus", o) is o
    assert ChildNode.get(n.ref, strict=False) is n
    assert ChildNode.get("bogus", strict=False) is None
    assert ChildNode.get(n.ref, o, strict=False) is n
    assert ChildNode.get("bogus", o, strict=False) is o

    # Test unattached never available via get
    n = SubChildNode("test")
    assert n.ref is None
    assert SubChildNode.get(n.id) is None
    assert SubChildNode.get(n.id, strict=False) is None


def test_get_any() -> None:
    # Test properly attached
    n = SubChildNode("test", attached=True)

    assert n.ref is not None
    n_ref = n.ref
    assert ASTNode.get_any(n_ref) is n
    assert ASTNode.get_any("bogus") is None
    assert ASTNode.get_any("bogus", n) is n

    # Test that after deletion, registry is cleared
    del n
    gc.collect()
    assert ASTNode.get_any(n_ref) is None


def test_replace() -> None:
    # Create a a non-attached node
    node = NonCompareAttrNode("node1", non_compare_attr="one")

    # Test successful replace
    new_node = node.replace(attr="node2")
    assert new_node is not node
    assert new_node.id != node.id
    assert new_node.attr == "node2"
    assert new_node.non_compare_attr == "one"
    assert new_node.non_init_attr == "test"
    assert new_node.ref is None

    # Test the same with dataclass replace
    new_node = replace(node, attr="node2")
    assert new_node is not node
    assert new_node.id != node.id
    assert new_node.attr == "node2"
    assert new_node.non_compare_attr == "one"
    assert new_node.non_init_attr == "test"
    assert new_node.ref is None

    # Now test with attached node
    anode = NonCompareAttrNode("node1", non_compare_attr="one", attached=True)

    # First dataclass replace. This should not preserve the ref
    new_anode = replace(anode, attr="node2")
    assert new_anode is not anode
    assert new_anode.id != anode.id
    assert new_anode.attr == "node2"
    assert new_anode.non_compare_attr == "one"
    assert new_anode.non_init_attr == "test"
    assert new_anode.ref is None

    # Now test with replace method. This should preserve the ref
    # And should remove the ref from the old node
    o_ref = anode.ref
    new_anode = anode.replace(attr="node2")
    assert new_anode is not anode
    assert new_anode.id != anode.id
    assert new_anode.attr == "node2"
    assert new_anode.non_compare_attr == "one"
    assert new_anode.non_init_attr == "test"
    assert new_anode.ref == o_ref
    assert NonCompareAttrNode.get(o_ref) is new_anode
    assert anode.ref is None

    # Test replace with content that produces the same id
    node = new_node
    new_node = node.replace(non_compare_attr="two")
    assert new_node is not node
    assert new_node == node
    assert hash(new_node) == hash(node)
    assert new_node.id == node.id
    assert new_node.is_equal(node)
    assert new_node.non_compare_attr == "two"
    assert new_node.non_init_attr == "test"

    # Same with dataclass replace
    node = new_node
    new_node = replace(node, non_compare_attr="two")
    assert new_node is not node
    assert new_node == node
    assert hash(new_node) == hash(node)
    assert new_node.id == node.id
    assert new_node.is_equal(node)
    assert new_node.non_compare_attr == "two"
    assert new_node.non_init_attr == "test"

    # Test unsuccessful replace with exception for attached node
    # Make sure it doesn't detach the original
    with pytest.raises(ValueError):
        new_anode.replace(non_compare_attr="three", non_init_attr="three")

    # validate that the node is not replaced
    assert NonCompareAttrNode.get(new_anode.ref) is new_anode


def test_replace_with() -> None:
    # Create two non-attached nodes
    node1 = ChildNode("node1")
    node2 = OtherNode("node2")

    # Replacing non-attached node yields the original
    assert node1.replace_with(node2) is node2

    # Now create attached versions
    anode1 = replace(node1, attached=True)
    anode2 = replace(node2, attached=True)

    anode1_ref = anode1.ref
    anode2_ref = anode2.ref

    # First test replacing attached with non-attached
    rnode1 = anode1.replace_with(node2)
    assert anode1.ref is None
    assert node2.ref is None
    assert rnode1.ref == anode1_ref
    assert ASTNode.get_any(anode1_ref) is rnode1
    assert rnode1 == node2
    assert rnode1.is_equal(node2)
    assert rnode1.id == node2.id
    assert rnode1.origin == node2.origin

    # Now test replacing attached with attached
    # This should preserve the ref on the original replaced node
    rnode2 = rnode1.replace_with(anode2)
    assert rnode1.ref is None
    assert anode2.ref == anode2_ref
    assert rnode2.ref == anode1_ref
    assert ASTNode.get_any(anode1_ref) is rnode2
    assert ASTNode.get_any(anode2_ref) is anode2
    assert rnode2 == anode2
    assert rnode2.is_equal(anode2)
    assert rnode2.id == anode2.id
    assert rnode2.origin == anode2.origin


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

    # Test equal itself
    one = ChildNode("1", origin=origin)
    assert one.is_equal(one)
    assert one == one

    # Test equal to an attached version of itself
    aone = replace(one, attached=True)
    assert aone is not one
    assert one == aone
    assert one.is_equal(aone)

    # Test multiple attached versions are all equal
    aone2 = replace(one, attached=True)
    assert aone is not aone2
    assert aone.ref != aone2.ref
    assert aone == aone2
    assert aone.is_equal(aone2)
    assert one == aone2
    assert one.is_equal(aone2)

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
    duplicated_one = replace(one)
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

    assert list(parent.iter_child_fields(sort_keys=True)) == [
        (
            tuple(ch_nodes[1:]),
            ParentNode.__dataclass_fields__["child_seq"],
        ),
        (
            None,
            ParentNode.__dataclass_fields__["restricted_child"],
        ),
        (
            ch_nodes[0],
            ParentNode.__dataclass_fields__["single_child"],
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
    assert list(node.get_properties(sort_keys=True)) != list(all_props.items())
    assert dict(node.get_properties(sort_keys=True)) == all_props

    assert list(node.get_properties(skip_id=False)) == [
        (node.id, StaticFieldGettersTest.__dataclass_fields__["id"]),
        *list(all_props.items()),
    ]
    assert list(node.get_properties(skip_id=False, sort_keys=True)) != [
        (node.id, StaticFieldGettersTest.__dataclass_fields__["id"]),
        *list(all_props.items()),
    ]
    assert dict(node.get_properties(skip_id=False, sort_keys=True)) == {
        node.id: StaticFieldGettersTest.__dataclass_fields__["id"],
        **all_props,
    }

    assert list(node.get_properties(skip_content_id=False)) == [
        (node.content_id, StaticFieldGettersTest.__dataclass_fields__["content_id"]),
        *list(all_props.items()),
    ]
    assert list(node.get_properties(skip_content_id=False, sort_keys=True)) != [
        (node.content_id, StaticFieldGettersTest.__dataclass_fields__["content_id"]),
        *list(all_props.items()),
    ]
    assert dict(node.get_properties(skip_content_id=False, sort_keys=True)) == {
        node.content_id: StaticFieldGettersTest.__dataclass_fields__["content_id"],
        **all_props,
    }

    assert list(node.get_properties(skip_origin=False)) == [
        (origin, StaticFieldGettersTest.__dataclass_fields__["origin"]),
        *list(all_props.items()),
    ]
    assert list(node.get_properties(skip_origin=False, sort_keys=True)) != [
        (origin, StaticFieldGettersTest.__dataclass_fields__["origin"]),
        *list(all_props.items()),
    ]
    assert dict(node.get_properties(skip_origin=False, sort_keys=True)) == {  # type: ignore[misc]
        origin: StaticFieldGettersTest.__dataclass_fields__["origin"],
        **all_props,
    }

    no_non_compare = all_props.copy()
    no_non_compare.pop("non_compare_v")

    assert list(node.get_properties(skip_non_compare=True)) == list(no_non_compare.items())
    assert list(node.get_properties(skip_non_compare=True, sort_keys=True)) != list(
        no_non_compare.items()
    )
    assert dict(node.get_properties(skip_non_compare=True, sort_keys=True)) == no_non_compare

    no_non_init = all_props.copy()
    no_non_init.pop("default")

    assert list(node.get_properties(skip_non_init=True)) == list(no_non_init.items())
    assert list(node.get_properties(skip_non_init=True, sort_keys=True)) != list(
        no_non_init.items()
    )
    assert dict(node.get_properties(skip_non_init=True, sort_keys=True)) == no_non_init

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
    assert list(parent.get_child_nodes(sort_keys=True)) == [child, mid_node]

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
    assert list(parent.get_child_nodes_with_field(sort_keys=True)) == [
        (child, ParentNode.__dataclass_fields__["child_seq"], 0),
        (mid_node, ParentNode.__dataclass_fields__["single_child"], None),
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
