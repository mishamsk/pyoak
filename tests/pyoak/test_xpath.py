from __future__ import annotations

from dataclasses import dataclass

from pyoak.match.xpath import ASTXpath
from pyoak.node import ASTNode
from pyoak.origin import NoOrigin

origin = NoOrigin()


@dataclass
class XpathNested(ASTNode):
    attr: str


@dataclass
class XpathNestedSub(XpathNested):
    attr1: str = "test"


@dataclass
class XpathMiddle(ASTNode):
    nested: XpathNested | XpathMiddle


@dataclass
class XpathRoot(ASTNode):
    middle_tuple: tuple[XpathMiddle, ...]


def test_xpath_match() -> None:
    n = XpathNested("test", origin=origin)
    m1 = XpathMiddle(n, origin=origin)
    n2 = XpathNestedSub("test2", origin=origin)
    m2 = XpathMiddle(n2, origin=origin)
    mm = XpathMiddle(m1, origin=origin)
    _ = XpathRoot((mm, m2), origin=origin)

    # Shape of the tree:
    # XpathRoot
    #   XpathMiddle
    #     XpathMiddle
    #       XpathNestedSub
    #   XpathMiddle
    #     XpathNestedSub

    xpath = ASTXpath("//XpathNested")
    assert xpath.match(n2)
    assert xpath.match(n)

    xpath = ASTXpath("/XpathRoot/XpathMiddle/XpathNested")
    assert xpath.match(n2)
    assert not xpath.match(n)

    xpath = ASTXpath("/XpathRoot//XpathNested")
    assert xpath.match(n2)
    assert xpath.match(n)

    xpath = ASTXpath("/XpathRoot/[0]XpathMiddle//XpathNested")
    assert not xpath.match(n2)
    assert xpath.match(n)

    xpath = ASTXpath("/XpathRoot/[]XpathMiddle//XpathNested")
    assert xpath.match(n2)
    assert xpath.match(n)

    xpath = ASTXpath("//@middle_tuple/@nested[]XpathNested")
    assert xpath.match(n2)
    assert not xpath.match(n)

    xpath = ASTXpath("@middle_tuple/@nested[]XpathNested")
    assert xpath.match(n2)
    assert not xpath.match(n)

    xpath = ASTXpath("//@middle_tuple/XpathMiddle/@nested[]XpathNested")
    assert not xpath.match(n2)
    assert xpath.match(n)

    xpath = ASTXpath("@middle_tuple/XpathMiddle/@nested[]XpathNested")
    assert not xpath.match(n2)
    assert xpath.match(n)
