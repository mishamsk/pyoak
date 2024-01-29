from dataclasses import dataclass, field

import pytest
from pyoak.node import ASTNode
from pyoak.origin import NO_ORIGIN, Origin
from pyoak.visitor import ASTVisitor


def test_non_strict_visitor(clean_ser_types) -> None:
    @dataclass
    class Base(ASTNode):
        origin: Origin = field(init=False, default=NO_ORIGIN)

    @dataclass
    class Sub(Base):
        pass

    @dataclass
    class SubSub(Sub):
        pass

    @dataclass
    class Sibling(Base):
        pass

    class FooVisitor(ASTVisitor[str]):
        def generic_visit(self, node: ASTNode) -> str:
            return "foo generic visit"

        def visit(self, node: ASTNode) -> str:
            return "foo visit: " + super().visit(node)

        def visit_Sub(self, node: Sub) -> str:
            return "foo visit sub"

    class BarVisitor(FooVisitor):
        def generic_visit(self, node: ASTNode) -> str:
            return "bar generic visit"

        def visit_SubSub(self, node: SubSub) -> str:
            return "bar visit subsub"

        def visit_Sibling(self, node: Sibling) -> str:
            return "bar visit sibling"

    # First test foo visitor
    foo_visitor = FooVisitor()
    assert foo_visitor.visit(Base()) == "foo visit: foo generic visit"
    assert foo_visitor.visit(Sub()) == "foo visit: foo visit sub"
    assert foo_visitor.visit(SubSub()) == "foo visit: foo visit sub"
    assert foo_visitor.visit(Sibling()) == "foo visit: foo generic visit"

    # Then test bar visitor
    bar_visitor = BarVisitor()
    assert bar_visitor.visit(Base()) == "foo visit: bar generic visit"
    assert bar_visitor.visit(Sub()) == "foo visit: foo visit sub"
    assert bar_visitor.visit(SubSub()) == "foo visit: bar visit subsub"
    assert bar_visitor.visit(Sibling()) == "foo visit: bar visit sibling"


def test_strict_visitor(clean_ser_types) -> None:
    @dataclass
    class Base(ASTNode):
        origin: Origin = field(init=False, default=NO_ORIGIN)

    @dataclass
    class Sub(Base):
        pass

    @dataclass
    class SubSub(Sub):
        pass

    @dataclass
    class Sibling(Base):
        pass

    class FooVisitor(ASTVisitor[str]):
        strict = True

        def generic_visit(self, node: ASTNode) -> str:
            return "foo generic visit"

        def visit(self, node: ASTNode) -> str:
            return "foo visit: " + super().visit(node)

        def visit_Sub(self, node: Sub) -> str:
            return "foo visit sub"

    class BarVisitor(FooVisitor):
        def generic_visit(self, node: ASTNode) -> str:
            return "bar generic visit"

        def visit_SubSub(self, node: SubSub) -> str:
            return "bar visit subsub"

        def visit_Sibling(self, node: Sibling) -> str:
            return "bar visit sibling"

    # First test foo visitor
    foo_visitor = FooVisitor()
    assert foo_visitor.visit(Base()) == "foo visit: foo generic visit"
    assert foo_visitor.visit(Sub()) == "foo visit: foo visit sub"
    assert foo_visitor.visit(SubSub()) == "foo visit: foo generic visit"
    assert foo_visitor.visit(Sibling()) == "foo visit: foo generic visit"

    # Then test bar visitor
    bar_visitor = BarVisitor()
    assert bar_visitor.visit(Base()) == "foo visit: bar generic visit"
    assert bar_visitor.visit(Sub()) == "foo visit: foo visit sub"
    assert bar_visitor.visit(SubSub()) == "foo visit: bar visit subsub"
    assert bar_visitor.visit(Sibling()) == "foo visit: bar visit sibling"


def test_visitor_with_extra_args(clean_ser_types) -> None:
    @dataclass
    class Base(ASTNode):
        origin: Origin = field(init=False, default=NO_ORIGIN)

    class FooVisitor(ASTVisitor[str]):
        def visit(self, node: ASTNode, extra_arg: int = 0) -> str:
            return self._dispatch_visit_method(node)(node, extra_arg)

        def generic_visit(self, node: ASTNode, extra_arg: int = 0) -> str:
            return f"foo generic visit: {extra_arg}"

        def visit_Base(self, node: Base, extra_arg: int) -> str:
            return f"foo visit base: {extra_arg}"

    assert FooVisitor().visit(Base(), 42) == "foo visit base: 42"


def test_visitor_validation(clean_ser_types) -> None:
    @dataclass
    class Base(ASTNode):
        origin: Origin = field(init=False, default=NO_ORIGIN)

    @dataclass
    class Sub(Base):
        pass

    # Method name vs Node type test validation does not kick in without the validate flag
    class FirstDummyVisitor(ASTVisitor[None]):
        def visit_SomeNode(self, node: Base) -> None:
            pass

    with pytest.raises(TypeError) as err:

        class FailingDummyVisitor(ASTVisitor[None], validate=True):
            def visit_NotEnoughArgs(self) -> None:
                pass

            def visit_NoAnnotation(self, node) -> None:
                pass

            def visit_NotASubclass(self, node: str) -> None:
                pass

            def visit_ComplexType(self, node: Base | Sub) -> None:
                pass

            def visit_StringAnnotation(self, node: "Base") -> None:
                # local types can't be used in string annotations
                # this also tests use of __future__.annotations with local types
                pass

            def visit_SomeNode(self, node: Base) -> None:
                pass

            # This should be fine
            def visit_Base(self, node: Base) -> None:
                pass

            def visitNotAVisitorMethod(self) -> None:
                pass

    assert (
        err.value.args[0]
        == "Visitor class 'FailingDummyVisitor' method(s) have invalid signature(s):\n"
        "  - 'visit_ComplexType': Node type annotation must be a single ASTNode subclass\n"
        "  - 'visit_NoAnnotation': Node type annotation is missing\n"
        "  - 'visit_NotASubclass': Node type annotation must be a subclass of ASTNode\n"
        "  - 'visit_NotEnoughArgs': Method must have at least two parameters: self and node\n"
        "  - 'visit_SomeNode': Method name doesn't match the second argument type annotation\n"
        "  - 'visit_StringAnnotation': Invalid signature or a string annotation that can't be resolved: name 'Base' is not defined"
    )
