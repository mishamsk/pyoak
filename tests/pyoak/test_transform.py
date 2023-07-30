from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
from pyoak.error import ASTTransformError
from pyoak.node import ASTNode, ASTTransformer, ASTTransformVisitor
from pyoak.origin import NoOrigin


@dataclass
class TRChildNode(ASTNode):
    what: Literal["keep", "remove", "transform"]
    attr: str
    origin: NoOrigin = field(default_factory=NoOrigin, init=False)


@dataclass
class TRParentNode(ASTNode):
    attr: str
    child: TRChildNode | None
    child_nodes: tuple[TRChildNode | TRParentNode, ...]
    origin: NoOrigin = field(default_factory=NoOrigin, init=False)


@dataclass
class TRRootNode(ASTNode):
    child: TRParentNode
    child_nodes: tuple[TRParentNode, ...]
    origin: NoOrigin = field(default_factory=NoOrigin, init=False)


def test_transform_visitor() -> None:
    class TRVisitor(ASTTransformVisitor):
        def visit_TRChildNode(self, node: TRChildNode) -> TRChildNode:
            return node.replace(attr="changed " + node.attr)

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

            return node.replace(**changes)

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

    # Check that the original tree is fully detached
    assert all(node.detached for node in orig_root.dfs())

    assert new_root is not orig_root
    assert isinstance(new_root, TRRootNode)
    assert new_root.is_attached_root
    assert new_root.id == orig_root.id

    # Check first child
    assert new_root.child is not parent1
    assert new_root.child.id == parent1.id
    assert new_root.child.attr == "changed parent1"
    assert len(new_root.child.child_nodes) == 1
    assert new_root.child.child_nodes[0] is not child_transform1
    assert new_root.child.child_nodes[0].id == child_transform1.id
    assert new_root.child.child_nodes[0].attr == "changed child_transform1"

    # Check last child, must be the same and no children
    # transformed. This checks that even if visitor visits
    # children, if they are discarded, then neither parents
    # nor the original nodes are affected.
    assert new_root.child_nodes[-1] is not parent3
    assert new_root.child_nodes[-1] == parent3
    assert parent3.child is child_keep4
    assert child_keep4.attr == "child_keep4"

    # Check middle child
    assert new_root.child_nodes[0] is not parent2
    assert new_root.child_nodes[0].id == parent2.id
    assert new_root.child_nodes[0].attr == "changed parent2"
    assert len(new_root.child_nodes[0].child_nodes) == 3
    assert new_root.child_nodes[0].child_nodes[0] is not child_keep1
    assert new_root.child_nodes[0].child_nodes[0] == child_keep1
    assert new_root.child_nodes[0].child_nodes[1] is not child_keep2
    assert new_root.child_nodes[0].child_nodes[1] == child_keep2
    assert new_root.child_nodes[0].child_nodes[2] is not child_transform2
    assert new_root.child_nodes[0].child_nodes[2].id == child_transform2.id
    assert new_root.child_nodes[0].child_nodes[2].attr == "changed child_transform2"

    new_root.detach()


def test_transform_visitor_error_recovery() -> None:
    class TRVisitor(ASTTransformVisitor):
        def visit_TRChildNode(self, node: TRChildNode) -> TRChildNode:
            # Fail on the very last child
            if node.attr == "child_transform2":
                raise ValueError("child_transform2")

            return node.replace(attr="changed " + node.attr)

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

            return node.replace(**changes)

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

    # Run a failing visitor
    with pytest.raises(ASTTransformError):
        _ = TRVisitor().transform(orig_root)

    # Make sure the original tree is not modified
    assert orig_root.is_attached_root
    assert all(not n.detached for n in orig_root.dfs(skip_self=True))
    assert orig_root.child is parent1
    assert parent1.attr == "parent1"
    assert orig_root.child.child_nodes[0] is child_remove1
    assert child_remove1.attr == "child_remove1"
    assert orig_root.child.child_nodes[1] is child_transform1
    assert child_transform1.attr == "child_transform1"
    assert orig_root.child.child_nodes[2] is child_remove2
    assert child_remove2.attr == "child_remove2"
    assert orig_root.child_nodes[0] is parent2
    assert parent2.attr == "parent2"
    assert orig_root.child_nodes[0].child_nodes[0] is child_keep1
    assert child_keep1.attr == "child_keep1"
    assert orig_root.child_nodes[0].child_nodes[1] is child_keep2
    assert child_keep2.attr == "child_keep2"
    assert orig_root.child_nodes[0].child_nodes[2] is child_transform2
    assert child_transform2.attr == "child_transform2"

    # Cleanup
    orig_root.detach()


def test_transformer() -> None:
    class TRTransformer(ASTTransformer):
        def filter(self, node: ASTNode) -> bool:
            return isinstance(node, TRChildNode) and node.what in (
                "transform",
                "remove",
            )

        def transform(self, node: ASTNode) -> ASTNode | None:
            assert isinstance(node, TRChildNode)

            if node.what == "transform":
                return node.replace(attr="changed " + node.attr)
            elif node.what == "remove":
                return None
            else:
                raise AssertionError("Invalid node.what")

    child_keep1 = TRChildNode("keep", "child_keep1")
    child_keep2 = TRChildNode("keep", "child_keep2")
    child_keep3 = TRChildNode("keep", "child_keep3")
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
    parent3 = TRParentNode(attr="parent3", child=None, child_nodes=(child_keep3,))

    orig_root = TRRootNode(child=parent1, child_nodes=(parent2, parent3))

    new_root = TRTransformer().execute(orig_root)

    assert new_root is orig_root

    # Check first child
    assert orig_root.child is parent1
    assert len(orig_root.child.child_nodes) == 1
    assert child_remove1.detached
    assert child_remove2.detached
    assert orig_root.child.child_nodes[0] is not child_transform1
    assert orig_root.child.child_nodes[0].id == child_transform1.id
    assert orig_root.child.child_nodes[0].attr == "changed child_transform1"

    # Check last child, must be the same
    assert orig_root.child_nodes[-1] is parent3

    # Check middle child
    assert orig_root.child_nodes[0] is parent2
    assert len(orig_root.child_nodes[0].child_nodes) == 3
    assert orig_root.child_nodes[0].child_nodes[0] is child_keep1
    assert orig_root.child_nodes[0].child_nodes[1] is child_keep2
    assert orig_root.child_nodes[0].child_nodes[2] is not child_transform2
    assert orig_root.child_nodes[0].child_nodes[2].id == child_transform2.id
    assert orig_root.child_nodes[0].child_nodes[2].attr == "changed child_transform2"

    orig_root.detach()
