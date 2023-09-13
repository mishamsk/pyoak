from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import pytest
from pyoak.node import NODE_REGISTRY, ASTNode
from pyoak.visitor import ASTTransformVisitor


@dataclass(frozen=True)
class TRChildNode(ASTNode):
    what: Literal["keep", "remove", "transform"]
    attr: str


@dataclass(frozen=True)
class TRParentNode(ASTNode):
    attr: str
    child: TRChildNode | None
    child_nodes: tuple[TRChildNode | TRParentNode, ...]


@dataclass(frozen=True)
class TRRootNode(ASTNode):
    child: TRParentNode
    child_nodes: tuple[TRParentNode, ...]


def test_no_op_transform_visitor() -> None:
    class TRVisitor(ASTTransformVisitor):
        ...

    tree = TRParentNode("parent", TRChildNode("keep", "1"), (TRChildNode("keep", "2"),))

    assert TRVisitor().transform(tree) is tree


def test_partial_change_transform_visitor() -> None:
    class TRVisitor(ASTTransformVisitor):
        def visit_TRChildNode(self, node: TRChildNode) -> TRChildNode:
            if node.attr == "1":
                return replace(node, attr="changed " + node.attr)

            return node

    tree = TRParentNode("parent", TRChildNode("keep", "1"), (TRChildNode("keep", "2"),))

    new_tree = TRVisitor().transform(tree)

    assert isinstance(new_tree, TRParentNode)
    assert new_tree is not tree
    assert new_tree.child is not tree.child
    assert new_tree.child is not None and new_tree.child.attr == "changed 1"
    assert new_tree.child_nodes[0] is tree.child_nodes[0]


def test_transform_visitor() -> None:
    class TRVisitor(ASTTransformVisitor):
        def visit_TRChildNode(self, node: TRChildNode) -> TRChildNode:
            return replace(node, attr="changed " + node.attr)

        def visit_TRParentNode(self, node: TRParentNode) -> TRParentNode:
            changes: dict[str, Any] = {"attr": "changed " + node.attr}
            changes.update(self._transform_children(node))
            kept = 0

            if node.child is not None:
                if node.child.what == "keep":
                    changes.pop("child", None)
                    kept += 1

            new_child_nodes = []
            for tr_child, orig_child in zip(changes["child_nodes"], node.child_nodes):
                if not isinstance(orig_child, TRChildNode):
                    continue

                if orig_child.what == "keep":
                    new_child_nodes.append(orig_child)
                    kept += 1
                elif orig_child.what == "transform":
                    new_child_nodes.append(tr_child)

            changes["child_nodes"] = tuple(new_child_nodes)

            if len(node.child_nodes) + int(node.child is not None) == kept:
                return node

            return replace(node, **changes)

    child_keep1 = TRChildNode("keep", "child_keep1")
    child_keep2 = TRChildNode("keep", "child_keep2")
    child_keep3 = TRChildNode("keep", "child_keep3")
    child_keep4 = TRChildNode("keep", "child_keep4")
    child_remove1 = TRChildNode("remove", "child_remove1")
    child_remove2 = TRChildNode("remove", "child_remove2")
    child_transform1 = TRChildNode("transform", "child_transform1")
    child_transform2 = TRChildNode("transform", "child_transform2")

    parent1 = TRParentNode(
        attr="parent1",
        child=None,
        child_nodes=(child_remove1, child_transform1, child_remove2),
    )
    parent2 = TRParentNode(
        attr="parent2",
        child=None,
        child_nodes=(child_keep1, child_keep2, child_transform2),
    )
    parent3 = TRParentNode(attr="parent3", child=child_keep4, child_nodes=(child_keep3,))

    orig_root = TRRootNode(child=parent1, child_nodes=(parent2, parent3))

    new_root = TRVisitor().transform(orig_root)

    assert new_root is not orig_root
    assert isinstance(new_root, TRRootNode)

    # Check first child
    assert new_root.child is not parent1
    assert new_root.child != parent1
    assert new_root.child.attr == "changed parent1"
    assert len(new_root.child.child_nodes) == 1
    assert new_root.child.child_nodes[0] is not child_transform1
    assert new_root.child.child_nodes[0] != child_transform1
    assert new_root.child.child_nodes[0].attr == "changed child_transform1"

    # Check last child, must be the same and no children
    # transformed. This checks that even if visitor visits
    # children, if they are discarded, then neither parents
    # nor the original nodes are affected.
    assert new_root.child_nodes[-1] is parent3
    assert parent3.child is child_keep4
    assert child_keep4.attr == "child_keep4"

    # Check middle child
    assert new_root.child_nodes[0] is not parent2
    assert new_root.child_nodes[0] != parent2
    assert new_root.child_nodes[0].attr == "changed parent2"
    assert len(new_root.child_nodes[0].child_nodes) == 3
    assert new_root.child_nodes[0].child_nodes[0] is child_keep1
    assert new_root.child_nodes[0].child_nodes[1] is child_keep2
    assert new_root.child_nodes[0].child_nodes[2] is not child_transform2
    assert new_root.child_nodes[0].child_nodes[2] != child_transform2
    assert new_root.child_nodes[0].child_nodes[2].attr == "changed child_transform2"


def test_transform_visitor_error_recovery() -> None:
    class TRVisitor(ASTTransformVisitor):
        def visit_TRChildNode(self, node: TRChildNode) -> TRChildNode:
            # Fail on the very last child
            if node.attr == "child_transform2":
                raise ValueError("child_transform2")

            return replace(node, attr="changed " + node.attr)

        def visit_TRParentNode(self, node: TRParentNode) -> TRParentNode:
            changes: dict[str, Any] = {"attr": "changed " + node.attr}
            changes.update(self._transform_children(node))
            kept = 0

            if node.child is not None:
                if node.child.what == "keep":
                    changes.pop("child", None)
                    kept += 1

            new_child_nodes = []
            for tr_child, orig_child in zip(changes["child_nodes"], node.child_nodes):
                if not isinstance(orig_child, TRChildNode):
                    continue

                if orig_child.what == "keep":
                    new_child_nodes.append(orig_child)
                    kept += 1
                elif orig_child.what == "transform":
                    new_child_nodes.append(tr_child)

            changes["child_nodes"] = tuple(new_child_nodes)

            if len(node.child_nodes) + int(node.child is not None) == kept:
                return node

            return replace(node, **changes)

    child_keep1 = TRChildNode("keep", "child_keep1")
    child_keep2 = TRChildNode("keep", "child_keep2")
    child_remove1 = TRChildNode("remove", "child_remove1")
    child_remove2 = TRChildNode("remove", "child_remove2")
    child_transform1 = TRChildNode("transform", "child_transform1")
    child_transform2 = TRChildNode("transform", "child_transform2")

    parent1 = TRParentNode(
        attr="parent1",
        child=None,
        child_nodes=(child_remove1, child_transform1, child_remove2),
    )
    parent2 = TRParentNode(
        attr="parent2",
        child=None,
        child_nodes=(child_keep1, child_keep2, child_transform2),
    )

    orig_root = TRRootNode(child=parent1, child_nodes=(parent2,))
    serialized = orig_root.as_dict()

    # Run a failing visitor
    with pytest.raises(ValueError):
        _ = TRVisitor().transform(orig_root)

    # Make sure the original tree is not modified
    # Clear registry to ensure the same id's are used
    NODE_REGISTRY.clear()
    assert orig_root == TRRootNode.as_obj(serialized)
