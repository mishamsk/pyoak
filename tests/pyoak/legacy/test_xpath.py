from __future__ import annotations

from dataclasses import dataclass

from pyoak.legacy.match.xpath import ASTXpath
from pyoak.legacy.node import AwareASTNode as ASTNode
from pyoak.origin import NoOrigin

origin = NoOrigin()


@dataclass
class LegacyXpathNested(ASTNode):
    attr: str


@dataclass
class LegacyXpathNestedSub(LegacyXpathNested):
    attr1: str = "test"


@dataclass
class LegacyXpathMiddle(ASTNode):
    nested: LegacyXpathNested | LegacyXpathMiddle


@dataclass
class LegacyXpathRoot(ASTNode):
    middle_tuple: tuple[LegacyXpathMiddle, ...]


def test_xpath_match() -> None:
    n = LegacyXpathNested("test", origin=origin)
    m1 = LegacyXpathMiddle(n, origin=origin)
    n2 = LegacyXpathNestedSub("test2", origin=origin)
    m2 = LegacyXpathMiddle(n2, origin=origin)
    mm = LegacyXpathMiddle(m1, origin=origin)
    _ = LegacyXpathRoot((mm, m2), origin=origin)

    # Shape of the tree:
    # LegacyXpathRoot
    #   LegacyXpathMiddle
    #     LegacyXpathMiddle
    #       LegacyXpathNestedSub
    #   LegacyXpathMiddle
    #     LegacyXpathNestedSub

    xpath = ASTXpath("//LegacyXpathNested")
    assert xpath.match(n2)
    assert xpath.match(n)

    xpath = ASTXpath("/LegacyXpathRoot/LegacyXpathMiddle/LegacyXpathNested")
    assert xpath.match(n2)
    assert not xpath.match(n)

    xpath = ASTXpath("/LegacyXpathRoot//LegacyXpathNested")
    assert xpath.match(n2)
    assert xpath.match(n)

    xpath = ASTXpath("/LegacyXpathRoot/[0]LegacyXpathMiddle//LegacyXpathNested")
    assert not xpath.match(n2)
    assert xpath.match(n)

    xpath = ASTXpath("/LegacyXpathRoot/[]LegacyXpathMiddle//LegacyXpathNested")
    assert xpath.match(n2)
    assert xpath.match(n)

    xpath = ASTXpath("//@middle_tuple/@nested[]LegacyXpathNested")
    assert xpath.match(n2)
    assert not xpath.match(n)

    xpath = ASTXpath("@middle_tuple/@nested[]LegacyXpathNested")
    assert xpath.match(n2)
    assert not xpath.match(n)

    xpath = ASTXpath("//@middle_tuple/LegacyXpathMiddle/@nested[]LegacyXpathNested")
    assert not xpath.match(n2)
    assert xpath.match(n)

    xpath = ASTXpath("@middle_tuple/LegacyXpathMiddle/@nested[]LegacyXpathNested")
    assert not xpath.match(n2)
    assert xpath.match(n)
