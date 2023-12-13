from __future__ import annotations

from dataclasses import dataclass, field
from itertools import repeat
from typing import Any, Generator, Iterable, cast

import pytest
from deepdiff import DeepDiff
from pyoak.error import (
    ASTNodeDuplicateChildrenError,
    ASTNodeParentCollisionError,
    ASTNodeRegistryCollisionError,
    ASTNodeReplaceError,
    ASTNodeReplaceWithError,
)
from pyoak.match.error import ASTXpathDefinitionError
from pyoak.match.xpath import ASTXpath
from pyoak.node import (
    AST_SERIALIZE_DIALECT_KEY,
    ASTNode,
    ASTSerializationDialects,
    Field,
    NodeTraversalInfo,
)
from pyoak.origin import NO_ORIGIN, CodeOrigin, MemoryTextSource, get_code_range
from pyoak.serialize import TYPE_KEY


@dataclass
class ChildNode(ASTNode):
    attr: str


@dataclass
class SubChildNode(ChildNode):
    pass


@dataclass
class OtherNode(ASTNode):
    attr: str


@dataclass
class NonCompareAttrNode(ASTNode):
    attr: str
    non_compare_attr: str = field(compare=False, default="test")
    non_init_attr: str = field(init=False, default="test")


@dataclass
class ParentNode(ASTNode):
    attr: str
    single_child: ASTNode
    child_list: tuple[ASTNode, ...]
    not_a_child_list: list[str]
    restricted_child: ChildNode | OtherNode | None = None


@dataclass
class SubParentNode(ParentNode):
    extra_attr: str | None = None


@dataclass
class StaticFieldGettersTest(ASTNode):
    attr: str
    _hidden_attr: str
    child: ASTNode
    optional_child: ASTNode | None
    child_list: tuple[ASTNode, ...]
    optional_child_list: tuple[ASTNode, ...] | None
    _not_hidden_child: ASTNode
    not_a_pure_child: str | ASTNode
    non_compare_attr: str = field(compare=False)
    non_init_attr: str = field(init=False)


origin = CodeOrigin(MemoryTextSource("test"), get_code_range(0, 1, 0, 3, 1, 3))


def _get_fname_set(fields: Iterable[tuple[str, Field]]) -> set[str]:
    return {f[0] for f in fields}


def test_get_property_fields() -> None:
    def_props = {"attr", "non_compare_attr", "non_init_attr"}
    assert _get_fname_set(StaticFieldGettersTest.get_property_fields()) == def_props
    assert _get_fname_set(
        StaticFieldGettersTest.get_property_fields(skip_hidden=False)
    ) == def_props | {
        "_hidden_attr",
    }
    assert _get_fname_set(
        StaticFieldGettersTest.get_property_fields(skip_non_compare=True)
    ) == def_props - {"non_compare_attr"}
    assert _get_fname_set(
        StaticFieldGettersTest.get_property_fields(skip_id=False)
    ) == def_props | {
        "id",
        "content_id",
    }
    assert _get_fname_set(
        StaticFieldGettersTest.get_property_fields(skip_origin=False)
    ) == def_props | {"origin"}
    assert _get_fname_set(
        StaticFieldGettersTest.get_property_fields(skip_original_id=False)
    ) == def_props | {
        "original_id",
    }
    assert _get_fname_set(
        StaticFieldGettersTest.get_property_fields(skip_id_collision_with=False)
    ) == def_props | {
        "id_collision_with",
    }


def test_get_child_fields() -> None:
    assert set(StaticFieldGettersTest.get_child_fields().keys()) == {
        "child",
        "optional_child",
        "child_list",
        "optional_child_list",
        "_not_hidden_child",
        "not_a_pure_child",
    }


def test_subclass_error() -> None:
    with pytest.raises(ValueError):

        @dataclass
        class ChildNode(ASTNode):
            other_attr: str


def test_id_handling() -> None:
    # Explicitly set id
    node = ChildNode("test", origin=origin, id="test_id")
    assert node.id == "test_id"
    assert node == ChildNode.get(node.id)
    assert ASTNode.get(node.id) is None
    assert node == ASTNode.get_any(node.id)

    # Implicitly set id
    node = ChildNode("test", origin=origin)
    assert node.id is not None
    assert node == ChildNode.get(node.id)
    assert ASTNode.get(node.id) is None
    assert node == ASTNode.get_any(node.id)

    # Allow duplicate Id, as long as it is not in registry
    assert node.detach()
    new_node = ChildNode("test", origin=origin)
    assert node == new_node
    assert node is not new_node
    assert node.id == new_node.id

    # Test esnure unique id
    duplicate_node = ChildNode("test", origin=origin)
    assert duplicate_node.id != new_node.id
    assert duplicate_node == new_node
    assert ASTNode.get_any(duplicate_node.id_collision_with or "") is new_node
    assert duplicate_node.original_id is None

    # Test create as duplicate
    created_as_duplicate_node = ChildNode("test", origin=origin, create_as_duplicate=True)
    assert created_as_duplicate_node.id != new_node.id
    assert created_as_duplicate_node.original_id == new_node.id
    assert created_as_duplicate_node == new_node
    assert ASTNode.get_any(created_as_duplicate_node.original_id) is new_node
    assert created_as_duplicate_node.id_collision_with is None

    # Cleanup in case GC won't collect them before the next test
    new_node.detach()
    duplicate_node.detach()
    created_as_duplicate_node.detach()


def test_detached_clones() -> None:
    # First create attached base nodes
    ch_nodes: list[ASTNode] = []
    ch_count = 2
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    # now test creating a new node that would normally collide with the parent
    # as detached + would normally fail due to child nodes being attached
    # to another parent
    det_parent = ParentNode(
        id=parent.id,
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
        create_detached=True,
    )

    assert det_parent.id == parent.id
    assert det_parent == parent
    assert det_parent is not parent
    assert det_parent.detached
    assert det_parent.id_collision_with is None
    assert det_parent.original_id is None

    for new_child, old_child in zip([det_parent.single_child, *det_parent.child_list], ch_nodes):
        assert new_child is old_child
        assert new_child.is_attached_subtree
        assert new_child.parent is parent

    # Now the same test but with detached duplicate
    det_parent = parent.duplicate(as_detached_clone=True)

    assert det_parent.id == parent.id
    assert det_parent == parent
    assert det_parent is not parent
    assert det_parent.detached
    assert det_parent.id_collision_with is None
    assert det_parent.original_id is None

    for new_child, old_child in zip([det_parent.single_child, *det_parent.child_list], ch_nodes):
        assert new_child.id == old_child.id
        assert new_child == old_child
        assert new_child is not old_child
        assert new_child.detached
        assert new_child.id_collision_with is None
        assert new_child.original_id is None


def test_parent_handling() -> None:
    # Test automatic parent assignment
    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    assert ch_nodes == list(parent.get_child_nodes())
    assert all([ASTNode.get_any(ch_nodes[i].id) is ch_nodes[i] for i in range(ch_count)])
    assert all(ch_nodes.parent is parent for ch_nodes in ch_nodes)

    # Test disallowing of assigning children to multiple parents
    with pytest.raises(ASTNodeParentCollisionError) as excinfo:
        _ = ParentNode(
            attr="Test Second Parent",
            single_child=ch_nodes[1],
            child_list=tuple(ch_nodes[3:5]),
            not_a_child_list=[],
            origin=origin,
        )

    assert cast(ParentNode, excinfo.value.new_node).attr == "Test Second Parent"
    assert excinfo.value.collided_child is ch_nodes[1]
    assert excinfo.value.collided_parent is parent

    # Test allowing to re-assign children to a new parent, after dropping the old one
    assert parent.detach_self()

    new_parent = ParentNode(
        attr="Test New Parent",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    assert ch_nodes == list(new_parent.get_child_nodes())
    assert all([ASTNode.get_any(ch_nodes[i].id) is ch_nodes[i] for i in range(ch_count)])
    assert all(ch_nodes.parent is new_parent for ch_nodes in ch_nodes)

    # Test node re-attachment after dropping
    assert new_parent.detach_self()
    for i in range(ch_count - 2):
        assert ch_nodes[i].detach()

    new_parent = ParentNode(
        attr="Test New Parent",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    assert ch_nodes == list(new_parent.get_child_nodes())
    assert all([ASTNode.get_any(ch_nodes[i].id) is ch_nodes[i] for i in range(ch_count)])
    assert all(ch_nodes.parent is new_parent for ch_nodes in ch_nodes)

    # Cleanup in case GC won't collect them before the next test
    for ch_node in ch_nodes:
        ch_node.detach()
    new_parent.detach()


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
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
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

    assert one == two == one_dup == two_dup
    assert one.content_id == two.content_id == one_dup.content_id == two_dup.content_id


def test_unique_children_check() -> None:
    ch_nodes: list[ASTNode] = []
    ch_count = 2
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    # Test disallowing of assigning children to multiple parents
    with pytest.raises(ASTNodeDuplicateChildrenError) as excinfo:
        _ = ParentNode(
            attr="Test",
            single_child=ch_nodes[0],
            child_list=tuple(repeat(ch_nodes[1], 2)),
            not_a_child_list=[str(i) for i in range(10)],
            origin=origin,
        )

    assert excinfo.value.child is ch_nodes[1]
    assert excinfo.value.last_field_name == "child_list"
    assert excinfo.value.last_index == 0
    assert excinfo.value.new_field_name == "child_list"
    assert excinfo.value.new_index == 1

    # Cleanup in case GC won't collect them before the next test
    for ch_node in ch_nodes:
        ch_node.detach()


def test_detach() -> None:
    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    # Test full subtree drop
    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    assert parent.detach()
    assert parent.detached

    assert all([ASTNode.get_any(ch_nodes[i].id) is None for i in range(ch_count)])
    assert all([ch.detached for ch in ch_nodes])

    # Test self drop
    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    assert parent.detach(only_self=True)
    assert parent.detached

    assert all([ChildNode.get(ch_nodes[i].id) is ch_nodes[i] for i in range(ch_count)])
    assert all([not ch.detached for ch in ch_nodes])

    # Test self drop alias function
    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    assert parent.detach_self()
    assert parent.detached

    assert all([ChildNode.get(ch_nodes[i].id) is ch_nodes[i] for i in range(ch_count)])
    assert all([not ch.detached for ch in ch_nodes])

    # Test disallowing dropping of attached subtrees
    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    assert not ch_nodes[0].detach()

    # Cleanup in case GC won't collect them before the next test
    for ch_node in ch_nodes:
        ch_node.detach()
    parent.detach()


def test_attach() -> None:
    # First create an attached tree
    ch_nodes: list[ASTNode] = []
    ch_count = 3
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    # Now detach and reattach it
    parent.detach()
    assert parent.detached
    parent.attach()
    assert all([not n.node.detached for n in parent.dfs()])

    # Another test - try attaching an old tree with some children
    # already attached to a new parent
    parent.detach()

    new_parent = ParentNode(
        attr="New Test",
        single_child=parent.single_child,
        child_list=tuple(),
        not_a_child_list=[],
        origin=origin,
    )

    with pytest.raises(ASTNodeParentCollisionError) as excinfo:
        parent.attach()

    assert excinfo.value.collided_child is parent.single_child
    assert excinfo.value.collided_parent is new_parent

    # Also test that attaching a node with id already present in the tree
    # will raise an error as well
    existing_node = ChildNode("Colliding Node", id=parent.id, origin=origin)

    with pytest.raises(ASTNodeRegistryCollisionError) as excinfo1:
        parent.attach()

    assert excinfo1.value.existing_node is existing_node
    assert excinfo1.value.new_node is parent
    assert excinfo1.value.operation == "attach"


def test_replace() -> None:
    # Test simple replacement
    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    another_parent = ParentNode(
        attr="Another Test",
        single_child=ChildNode("Another Child", origin=origin),
        child_list=tuple(),
        not_a_child_list=[],
        origin=origin,
    )

    # Test checks for forbidden replacement
    with pytest.raises(ASTNodeDuplicateChildrenError) as excinfo:
        _ = parent.replace(single_child=ch_nodes[1])

    assert excinfo.value.child is ch_nodes[1]

    with pytest.raises(ASTNodeParentCollisionError) as excinfo1:
        _ = parent.replace(single_child=another_parent.single_child)

    assert excinfo1.value.collided_child is another_parent.single_child
    assert excinfo1.value.collided_parent is another_parent

    to_replace = NonCompareAttrNode("test", origin=origin)
    changes = dict(
        attr="good replacement",
        non_compare_attr="good replacement",
        id="forbidden",
        id_collision_with="forbidden",
        original_id="forbidden",
        content_id="forbidden",
        extra="forbidden",
        non_init_attr="forbidden",
    )
    with pytest.raises(ASTNodeReplaceError) as excinfo2:
        _ = to_replace.replace(**changes)

    assert excinfo2.value.node is to_replace
    assert excinfo2.value.changes == changes
    assert set(excinfo2.value.error_keys) == {
        "id",
        "id_collision_with",
        "original_id",
        "content_id",
        "extra",
        "non_init_attr",
    }

    new_child = ChildNode("New Child", origin=origin)
    new_parent = parent.replace(single_child=new_child)

    assert new_parent is not parent
    assert new_parent.is_attached_root
    assert parent.detached
    assert new_parent.single_child is new_child
    assert new_child.parent is new_parent
    assert parent.single_child is not new_child
    assert parent.attr == new_parent.attr
    assert parent.child_list == new_parent.child_list
    assert parent.not_a_child_list == new_parent.not_a_child_list
    assert parent.origin == new_parent.origin
    assert ParentNode.get(parent.id) is new_parent
    assert not new_child.detached
    assert ch_nodes[0].is_attached_root  # because it's parent was detached, but it wasn't

    assert [new_child] + ch_nodes[1:] == list(new_parent.get_child_nodes())
    assert all(ch_nodes.parent is new_parent for ch_nodes in ch_nodes[1:])

    # Test replacing with a detached node
    # Parent has been detached in the previous test
    # Now we should be able to things not allowed normally
    # including replacing with a node that is attached to another parent
    new_detached_parent = parent.replace(single_child=new_child)

    assert new_detached_parent is not parent
    assert new_detached_parent.detached
    assert new_detached_parent.single_child is new_child

    # Test replacing sub-tree
    new_parent.detach_self()  # detach all children

    sub = ParentNode(
        attr="Sub",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:2]),
        not_a_child_list=[],
        origin=origin,
    )

    sub1 = ParentNode(
        attr="Sub1",
        single_child=ch_nodes[2],
        child_list=tuple(ch_nodes[3:5]),
        not_a_child_list=[],
        origin=origin,
    )

    sub2 = ParentNode(
        attr="Sub2",
        single_child=ch_nodes[5],
        child_list=tuple(ch_nodes[6:8]),
        not_a_child_list=[],
        origin=origin,
    )

    root = ParentNode(
        attr="Sub",
        single_child=sub,
        child_list=(sub1, sub2),
        not_a_child_list=[],
        origin=origin,
    )

    assert ch_nodes[8].is_attached_root
    assert sub2.single_child is ch_nodes[5]
    assert sub2.child_list == (ch_nodes[6], ch_nodes[7])
    new_sub2 = sub2.replace(single_child=ch_nodes[8], child_list=(ch_nodes[6], ch_nodes[9]))

    assert new_sub2 is not sub2
    assert new_sub2.single_child is ch_nodes[8]
    assert new_sub2.child_list == (ch_nodes[6], ch_nodes[9])
    assert new_sub2.parent is root
    assert new_sub2 in root.get_child_nodes()
    assert sub2 not in root.get_child_nodes()

    # Now check that content id is updated when child node value has changed
    orig_root_content_id = root.content_id
    orig_sub_content_id = sub.content_id
    orig_ch_0_content_id = ch_nodes[0].content_id

    replaced_ch_0 = ch_nodes[0].replace(attr="New Value")

    assert root.content_id != orig_root_content_id
    assert sub.content_id != orig_sub_content_id
    assert replaced_ch_0.content_id != orig_ch_0_content_id

    # Cleanup in case GC won't collect them before the next test
    for ch_node in ch_nodes:
        ch_node.detach()
    root.detach()


def test_replace_with(caplog: pytest.LogCaptureFixture) -> None:
    # Test replace with another node, including type checking

    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = SubParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:-2]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    another = ParentNode(
        attr="Test2",
        single_child=ch_nodes[-2],
        child_list=(),
        not_a_child_list=[],
        origin=origin,
    )

    # Test replacing within a tuple, same type
    old = ch_nodes[1]
    new_child = ch_nodes[-1]
    orig_id = new_child.id
    old.replace_with(new_child)
    assert new_child.parent is parent
    assert old.detached
    assert old not in parent.get_child_nodes()
    assert new_child in parent.get_child_nodes()
    assert new_child is parent.child_list[0]
    assert new_child.original_id == orig_id
    assert new_child.id == old.id

    # Test replacing a regular field with different type
    old = ch_nodes[0]
    new_child = another
    orig_id = new_child.id
    old.replace_with(new_child)
    assert new_child.parent is parent
    assert old.detached
    assert old not in parent.get_child_nodes()
    assert new_child in parent.get_child_nodes()
    assert new_child is parent.single_child
    assert new_child.original_id == orig_id
    assert new_child.id == old.id

    # Test failure on type mismatch
    old = ChildNode("old", origin=origin)
    parent = parent.replace(restricted_child=old)
    another = ParentNode(
        attr="Test2",
        single_child=ChildNode("single", origin=origin),
        child_list=(),
        not_a_child_list=[],
        origin=origin,
    )

    new_child = another
    with pytest.raises(ASTNodeReplaceWithError) as exc_info:
        old.replace_with(new_child)

    assert exc_info.value.node is old
    assert exc_info.value.new_node is new_child
    assert exc_info.value.message.endswith(
        "because parent expects nodes of type: ChildNode, OtherNode"
    )
    assert new_child.is_attached_root
    assert old.is_attached_subtree
    assert old in parent.get_child_nodes()
    assert new_child not in parent.get_child_nodes()
    assert new_child.original_id is None
    assert new_child.id != old.id

    # Test failure on trying to replace with a subtree
    old = ch_nodes[2]
    new_child = ch_nodes[3]
    assert new_child.is_attached_subtree

    with pytest.raises(ASTNodeReplaceWithError) as exc_info:
        old.replace_with(new_child)

    assert exc_info.value.node is old
    assert exc_info.value.new_node is new_child
    assert exc_info.value.message.endswith("because the new node has a parent already")
    assert old.is_attached_subtree
    assert old in parent.get_child_nodes()
    assert new_child in parent.get_child_nodes()  # still in the tree
    assert new_child.original_id is None
    assert new_child.id != old.id

    # Test replace attached root with detached node
    parent.detach()

    old = ChildNode("attached old", origin=origin)
    new_child = ch_nodes[3]
    orig_id = new_child.id
    assert new_child.detached
    assert old.is_attached_root

    old.replace_with(new_child)
    assert old.detached
    assert new_child.is_attached_root
    assert new_child.original_id == orig_id

    # Test replace with detached node that can't be attached
    # validate that the original node is not detached
    conflicting_child = ChildNode("conflict", origin=origin)
    old = ChildNode("old", origin=origin)
    attached_parent = ParentNode(
        attr="Test2",
        single_child=old,
        child_list=(),
        not_a_child_list=[],
        origin=origin,
    )

    new_child = ChildNode(
        "new with conflict",
        id=conflicting_child.id,
        origin=origin,
        create_detached=True,
    )
    new = ParentNode(
        attr="Test2",
        single_child=new_child,
        child_list=(),
        not_a_child_list=[],
        origin=origin,
        create_detached=True,
    )

    with pytest.raises(ASTNodeReplaceWithError) as exc_info:
        old.replace_with(new)

    assert exc_info.value.node is old
    assert exc_info.value.new_node is new
    ctx = exc_info.value.__context__
    assert isinstance(ctx, ASTNodeRegistryCollisionError)
    assert ctx.existing_node is conflicting_child
    assert ctx.new_node is new_child
    assert not old.detached
    assert new.detached
    assert old.parent is attached_parent

    # Test replace attached root with None (effective detach)
    new_child.replace_with(None)
    assert new_child.detached

    # Test replace optional field with None
    to_repace = ChildNode("other", origin=origin)
    with_restricted = ParentNode(
        attr="Test3",
        single_child=ChildNode("some", origin=origin),
        child_list=(),
        not_a_child_list=[],
        restricted_child=to_repace,
        origin=origin,
    )

    assert with_restricted.restricted_child is to_repace
    assert to_repace.is_attached_subtree
    assert with_restricted.is_attached_root
    to_repace.replace_with(None)

    assert with_restricted.restricted_child is None
    assert to_repace.detached
    assert with_restricted.is_attached_root

    # Test replace with None in a sequence
    parent.detach()
    another.detach()

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:-2]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    ch_nodes[2].replace_with(None)
    assert ch_nodes[2].detached
    assert ch_nodes[2] not in parent.get_child_nodes()
    assert parent.child_list == (ch_nodes[1], *ch_nodes[3:-2])
    for ch_node in ch_nodes[3:-2]:
        # check the parent index has been updated to reflect the new order
        assert ch_node.parent is parent
        assert parent.child_list.index(ch_node) == ch_node.parent_index

    # Test replace non optional field with None
    with pytest.raises(ASTNodeReplaceWithError) as exc_info:
        # currently in parent `single_child`, which is not optional
        ch_nodes[0].replace_with(None)

    assert exc_info.value.node is ch_nodes[0]
    assert exc_info.value.new_node is None
    assert exc_info.value.message.endswith("because parent expects a non-optional node")

    # Cleanup in case GC won't collect them before the next test
    for ch_node in ch_nodes:
        ch_node.detach()
    another.detach()


def test_duplicate() -> None:
    # Test simple replacement
    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )
    dup_parent = parent.duplicate()
    assert dup_parent is not parent
    assert dup_parent == parent
    assert dup_parent.original_id
    assert ParentNode.get(dup_parent.original_id) is parent

    parent.detach()

    # Test duplicate with a detached node, must create
    # an attached node that is not marked as a duplicate
    dup_detached = parent.duplicate()
    assert parent.detached
    assert dup_detached is not parent
    assert dup_detached == parent
    assert dup_detached.original_id is None
    assert dup_detached.id == parent.id

    # Cleanup in case GC won't collect them before the next test
    dup_detached.detach()
    dup_parent.detach()


def test_serialization_and_equality() -> None:
    ch_nodes: list[ChildNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    # Make sure we test with duplicate ids
    ch_nodes.append(ChildNode(ch_nodes[-1].attr, origin=origin))
    ch_nodes.append(ch_nodes[-1].duplicate())

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    parent.detach()  # we must drop the node before we can deserialized it (duplicate ids)
    serialized_dict = parent.to_dict()
    deserialized = ParentNode.from_dict(serialized_dict)

    # Class should be present on the root node
    assert TYPE_KEY in serialized_dict

    # Should equal even without parents assigned after deserialization
    assert parent == deserialized

    # id_collision_with & original_id are not used in comparison, but should be set correctly
    assert deserialized.child_list[-1].id_collision_with == ch_nodes[-1].id_collision_with
    assert deserialized.child_list[-1].original_id == ch_nodes[-1].original_id

    # Should still be equal even if id is changed
    deserialized.detach()
    serialized_dict = parent.to_dict()
    serialized_dict["id"] = "other_id"
    assert parent == ParentNode.from_dict(serialized_dict)


def test_collision_handling_at_deserialization() -> None:
    orig_node = ChildNode("Test", origin=origin)
    dup_node = orig_node.duplicate()
    collided_node = ChildNode("Test", origin=origin)

    assert dup_node.id_collision_with is None
    assert dup_node.original_id == orig_node.id
    assert collided_node.id_collision_with == orig_node.id
    assert collided_node.original_id is None

    # Serialize & deseriale everything
    orig_deserialized = ChildNode.as_obj(orig_node.as_dict())
    dup_deserialized = ChildNode.as_obj(dup_node.as_dict())
    collided_deserialized = ChildNode.as_obj(collided_node.as_dict())

    # Original didn't have a collision but deserialized should have
    assert orig_deserialized.id_collision_with == orig_node.id
    assert orig_deserialized.original_id is None

    # Duplicate was created as duplicate, so deserialize should have the same
    assert dup_deserialized.id_collision_with is None
    assert dup_deserialized.original_id == orig_node.id

    # Collided node should have the same collision as the original
    # Desptie the fact that during deserialization it's id will collide
    # with a different node than original (because it will have _2 prefix
    # which will collide with the collided node itself, not the original)
    assert collided_deserialized.id_collision_with == orig_node.id
    assert collided_deserialized.original_id is None


def test_ast_explorer_serialization_dialect() -> None:
    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
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
        "child_list",
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
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
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
    assert "child_list" in serialized_dialect_dict
    for child in serialized_dialect_dict["child_list"]:
        assert "origin" in child
        assert "source" in child["origin"]
        assert child["origin"]["source"] == {
            TYPE_KEY: "Source",
            "source_uri": "",
            "source_type": "",
        }

    parent.detach()
    # Try to deserialize
    ParentNode.from_dict(serialized_dialect_dict)


@pytest.mark.dep_msgpack
def test_to_msgpack() -> None:
    pytest.importorskip("msgpack")

    ch_nodes: list[ASTNode] = []
    ch_count = 10
    for i in range(ch_count):
        ch_nodes.append(ChildNode(str(i), origin=origin))

    parent = ParentNode(
        attr="Test",
        single_child=ch_nodes[0],
        child_list=tuple(ch_nodes[1:]),
        not_a_child_list=[str(i) for i in range(10)],
        origin=origin,
    )

    serialized = parent.to_msgpck()
    parent.detach()
    deserialized = ParentNode.from_msgpck(serialized)

    assert parent == deserialized


def test_is_equal() -> None:
    one = ChildNode("1", origin=origin)
    assert one.is_equal(one)

    two = ChildNode("2", origin=origin)
    other_type_ast = OtherNode("1", origin=origin)
    assert not one.is_equal("Test not ASTNode")
    assert not one.is_equal(other_type_ast)
    assert not one.is_equal(two)

    other_origin = CodeOrigin(MemoryTextSource("test2"), get_code_range(0, 1, 0, 4, 1, 4))
    other_origin_one = ChildNode("1", origin=other_origin)
    assert one != other_origin_one
    assert one.is_equal(other_origin_one)

    collided_one = ChildNode("1", origin=origin)
    assert one == collided_one
    assert one.is_equal(collided_one)

    duplicated_one = one.duplicate()
    assert one == duplicated_one
    assert one.is_equal(duplicated_one)

    created_as_duplicate_one = ChildNode("1", origin=origin, create_as_duplicate=True)
    assert one == created_as_duplicate_one
    assert one.is_equal(created_as_duplicate_one)

    # Test deep comparison with children

    parent = ParentNode(
        attr="Parent",
        single_child=ChildNode("single_child", origin=origin),
        child_list=(
            ChildNode("child_list1", origin=origin),
            ChildNode("child_list2", origin=origin),
        ),
        not_a_child_list=[],
        origin=origin,
    )

    other_parent = ParentNode(
        attr="Parent",
        single_child=ChildNode("single_child", origin=other_origin),
        child_list=(
            ChildNode("child_list1", origin=other_origin),
            ChildNode("child_list2", origin=other_origin),
        ),
        not_a_child_list=[],
        origin=other_origin,
    )
    assert parent != other_parent
    assert parent.is_equal(other_parent)

    other_parent.detach()

    other_parent = ParentNode(
        attr="Parent",
        single_child=ChildNode("single_child", origin=other_origin),
        child_list=(
            ChildNode("child_list1", origin=other_origin),
            ChildNode("child_list2-changed", origin=other_origin),
        ),
        not_a_child_list=[],
        origin=other_origin,
    )
    assert parent != other_parent
    assert not parent.is_equal(other_parent)

    # Check hidden attrs (should not be compared)
    hidone = NonCompareAttrNode(attr="1", non_compare_attr="one", origin=origin)
    hidtwo = NonCompareAttrNode(attr="1", non_compare_attr="two", origin=origin)
    assert hidone.is_equal(hidtwo)


def test_to_content_dict() -> None:
    single_child = ChildNode("single_child", origin=origin)
    child_list = ChildNode("child_list1", origin=origin)
    parent = ParentNode(
        attr="Parent",
        single_child=single_child,
        child_list=(child_list,),
        not_a_child_list=["1"],
        origin=origin,
    )

    assert parent.to_properties_dict() == dict(
        attr="Parent",
        not_a_child_list=["1"],
    )


@dataclass
class Nested(ASTNode):
    attr: str


@dataclass
class Middle(ASTNode):
    nested: Nested | Middle


@dataclass
class Root(ASTNode):
    middle_tuple: tuple[Middle, ...]


def test_xpath() -> None:
    n = Nested("test", origin=origin)
    assert n.xpath is None

    assert n.calculate_xpath()
    assert n.xpath == "/@root[0]Nested"

    m1 = Middle(n, origin=origin)

    assert not n.calculate_xpath()
    assert m1.calculate_xpath()
    assert m1.xpath == "/@root[0]Middle"
    assert n.xpath == "/@root[0]Middle/@nested[0]Nested"

    n2 = Nested("test2", origin=origin)
    m2 = Middle(n2, origin=origin)
    r = Root((m1, m2), origin=origin)

    assert r.calculate_xpath()
    assert r.xpath == "/@root[0]Root"
    assert m1.xpath == "/@root[0]Root/@middle_tuple[0]Middle"
    assert m2.xpath == "/@root[0]Root/@middle_tuple[1]Middle"
    assert n2.xpath == "/@root[0]Root/@middle_tuple[1]Middle/@nested[0]Nested"

    r.detach()


@dataclass
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
    return BTreeNode(v, left, right, origin=origin)


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
    ch = ChildNode("1", origin=origin)
    ch1 = ChildNode("1", origin=origin)
    sub_ch = SubChildNode("1", origin=origin)
    other = OtherNode("1", origin=origin)
    other1 = OtherNode("1", origin=origin)

    mid = ParentNode(
        attr="Mid",
        single_child=ch,
        child_list=(sub_ch, other),
        not_a_child_list=[],
        origin=origin,
    )

    root = ParentNode(
        attr="Root",
        single_child=mid,
        child_list=(ch1, other1),
        not_a_child_list=[],
        origin=origin,
    )

    assert set(root.gather(ChildNode)) == {ch, ch1, sub_ch}
    assert set(root.gather((ChildNode, OtherNode))) == {ch, ch1, sub_ch, other, other1}


def test_find() -> None:
    ch = ChildNode("1", origin=origin)
    ch1 = ChildNode("1", origin=origin)
    sub_ch = SubChildNode("1", origin=origin)
    other = OtherNode("1", origin=origin)
    other1 = OtherNode("1", origin=origin)

    mid = ParentNode(
        attr="Mid",
        single_child=ch,
        child_list=(sub_ch, other),
        not_a_child_list=[],
        origin=origin,
    )

    root = ParentNode(
        attr="Root",
        single_child=mid,
        child_list=(ch1, other1),
        not_a_child_list=[],
        origin=origin,
    )

    # Only minimal test. Full in test_xpath.py
    assert root.find("//ChildNode") is ch
    assert root.find(ASTXpath("//ChildNode")) is ch
    assert root.find("/OtherNode") is None

    with pytest.raises(ASTXpathDefinitionError):
        root.find("NonExistentClass")


def test_findall() -> None:
    ch = ChildNode("1", origin=origin)
    ch1 = ChildNode("1", origin=origin)
    sub_ch = SubChildNode("1", origin=origin)
    other = OtherNode("1", origin=origin)
    other1 = OtherNode("1", origin=origin)

    mid = ParentNode(
        attr="Mid",
        single_child=ch,
        child_list=(sub_ch, other),
        not_a_child_list=[],
        origin=origin,
    )

    root = ParentNode(
        attr="Root",
        single_child=mid,
        child_list=(ch1, other1),
        not_a_child_list=[],
        origin=origin,
    )

    # Only minimal test. Full in test_xpath.py
    assert set(root.findall("//ChildNode")) == {ch, ch1, sub_ch}
    assert set(root.findall(ASTXpath("//ChildNode"))) == {ch, ch1, sub_ch}
    assert set(root.findall("/OtherNode")) == set()

    with pytest.raises(ASTXpathDefinitionError):
        next(root.findall("NonExistentClass"))


class MiddleSubclass(Middle):
    pass


def test_get_first_ancestor_of_type() -> None:
    n1 = Nested("test1", origin=origin)
    m_m = MiddleSubclass(n1, origin=origin)
    m_r = Middle(m_m, origin=origin)
    r = Root((m_r,), origin=origin)

    assert n1.get_first_ancestor_of_type(Middle) is m_m
    assert n1.get_first_ancestor_of_type(Middle, exact_type=True) is m_r
    assert n1.get_first_ancestor_of_type(Root) is r
    assert m_m.get_first_ancestor_of_type(Middle) is m_r
    assert m_m.get_first_ancestor_of_type(Middle, exact_type=True) is m_r
    assert m_m.get_first_ancestor_of_type(ParentNode) is None
    assert m_m.get_first_ancestor_of_type(ParentNode, exact_type=True) is None


def test_no_origin_serialization() -> None:
    node = ChildNode("test", origin=NO_ORIGIN, create_detached=True)

    assert ChildNode.from_json(node.to_json()) == node
