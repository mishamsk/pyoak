from __future__ import annotations

from dataclasses import dataclass

import pytest
from pyoak.node import ASTNode
from pyoak.origin import NO_ORIGIN
from pyoak.visitor import ASTVisitor


@dataclass
class VisitorTestChildNode(ASTNode):
    attr: str


@dataclass
class VisitorTestSubclassChildNode(VisitorTestChildNode):
    attr2: str


@dataclass
class VisitorTestParentNode(ASTNode):
    child: VisitorTestChildNode


def test_visitor() -> None:
    test_child = VisitorTestSubclassChildNode("test", "test2", origin=NO_ORIGIN)
    test_parent = VisitorTestParentNode(test_child, origin=NO_ORIGIN)

    class NonStrictVisitor(ASTVisitor[list[str]]):
        def generic_visit(self, node: ASTNode) -> list[str]:
            ret = [node.__class__.__name__]
            for child in node.get_child_nodes():
                ret.extend(self.visit(child))
            return ret

        def visit_VisitorTestChildNode(self, node: VisitorTestChildNode) -> list[str]:
            return [node.attr]

    assert NonStrictVisitor().visit(test_parent) == [
        "VisitorTestParentNode",
        "test",
    ]

    class StrictVisitor2(ASTVisitor[list[str]]):
        strict = True

        def generic_visit(self, node: ASTNode) -> list[str]:
            ret = [node.__class__.__name__]
            for child in node.get_child_nodes():
                ret.extend(self.visit(child))
            return ret

        def visit_VisitorTestChildNode(self, node: VisitorTestChildNode) -> list[str]:
            return [node.attr]

        def visit_VisitorTestSubclassChildNode(
            self, node: VisitorTestSubclassChildNode
        ) -> list[str]:
            return [node.attr2]

    assert StrictVisitor2().visit(test_parent) == [
        "VisitorTestParentNode",
        "test2",
    ]


def test_visitor_validation() -> None:
    # Test validation does not kick in without the validate flag
    class FirstDummyVisitor(ASTVisitor[None]):
        def visit_SomeNode(self, node: VisitorTestChildNode) -> None:
            pass

    # Test validation passes with no type annotations
    class SecondDummyVisitor(ASTVisitor[None], validate=True):
        def visit_SomeNode(self, node) -> None:
            pass

    with pytest.raises(TypeError) as err:

        class FailingDummyVisitor(ASTVisitor[None], validate=True):
            def visit_SomeNode(self, node: VisitorTestChildNode) -> None:
                pass

            def visit_OtherNode(self, node: VisitorTestChildNode) -> None:
                pass

    assert (
        err.value.args[0]
        == "Visitor class 'FailingDummyVisitor' method(s) 'visit_OtherNode, visit_SomeNode' do not match node type annotation(s) 'VisitorTestChildNode, VisitorTestChildNode'"
    )
