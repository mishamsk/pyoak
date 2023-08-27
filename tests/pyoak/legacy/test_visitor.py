from __future__ import annotations

from dataclasses import dataclass

import pytest
from pyoak.legacy.node import ASTVisitor
from pyoak.legacy.node import AwareASTNode as ASTNode
from pyoak.origin import NoOrigin


@dataclass
class LegacyVisitorTestChildNode(ASTNode):
    attr: str


@dataclass
class LegacyVisitorTestSubclassChildNode(LegacyVisitorTestChildNode):
    attr2: str


@dataclass
class LegacyVisitorTestParentNode(ASTNode):
    child: LegacyVisitorTestChildNode


def test_visitor() -> None:
    test_child = LegacyVisitorTestSubclassChildNode("test", "test2", origin=NoOrigin())
    test_parent = LegacyVisitorTestParentNode(test_child, origin=NoOrigin())

    class NonStrictVisitor(ASTVisitor[list[str]]):
        def generic_visit(self, node: ASTNode) -> list[str]:
            ret = [node.__class__.__name__]
            for child in node.get_child_nodes():
                ret.extend(self.visit(child))
            return ret

        def visit_LegacyVisitorTestChildNode(self, node: LegacyVisitorTestChildNode) -> list[str]:
            return [node.attr]

    assert NonStrictVisitor().visit(test_parent) == [
        "LegacyVisitorTestParentNode",
        "test",
    ]

    class StrictVisitor2(ASTVisitor[list[str]]):
        strict = True

        def generic_visit(self, node: ASTNode) -> list[str]:
            ret = [node.__class__.__name__]
            for child in node.get_child_nodes():
                ret.extend(self.visit(child))
            return ret

        def visit_LegacyVisitorTestChildNode(self, node: LegacyVisitorTestChildNode) -> list[str]:
            return [node.attr]

        def visit_LegacyVisitorTestSubclassChildNode(
            self, node: LegacyVisitorTestSubclassChildNode
        ) -> list[str]:
            return [node.attr2]

    assert StrictVisitor2().visit(test_parent) == [
        "LegacyVisitorTestParentNode",
        "test2",
    ]


def test_visitor_validation() -> None:
    # Test validation does not kick in without the validate flag
    class FirstDummyVisitor(ASTVisitor[None]):
        def visit_SomeNode(self, node: LegacyVisitorTestChildNode) -> None:
            pass

    # Test validation passes with no type annotations
    class SecondDummyVisitor(ASTVisitor[None], validate=True):
        def visit_SomeNode(self, node) -> None:
            pass

    with pytest.raises(TypeError) as err:

        class FailingDummyVisitor(ASTVisitor[None], validate=True):
            def visit_SomeNode(self, node: LegacyVisitorTestChildNode) -> None:
                pass

            def visit_OtherNode(self, node: LegacyVisitorTestChildNode) -> None:
                pass

    assert (
        err.value.args[0]
        == "Visitor class 'FailingDummyVisitor' method(s) 'visit_OtherNode, visit_SomeNode' do not match node type annotation(s) 'LegacyVisitorTestChildNode, LegacyVisitorTestChildNode'"
    )
