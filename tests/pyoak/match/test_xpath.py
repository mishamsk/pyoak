from __future__ import annotations

from dataclasses import dataclass

import pytest
from pyoak.match.error import ASTXpathDefinitionError
from pyoak.match.xpath import ASTXpath
from pyoak.node import ASTNode
from pyoak.origin import NO_ORIGIN

origin = NO_ORIGIN


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


def test_init() -> None:
    xpath = ASTXpath("//XpathNested")

    # Test caching
    assert xpath is ASTXpath("//XpathNested")
    assert xpath is not ASTXpath("//XpathNestedSub")

    # Test non-existent class
    with pytest.raises(ASTXpathDefinitionError) as excinfo:
        ASTXpath("NonExistentClass")

    assert "NonExistentClass" in str(excinfo.value)

    # Test invalid XPath
    with pytest.raises(ASTXpathDefinitionError) as excinfo:
        ASTXpath("//")

    assert "Incorrect xpath definition" in str(excinfo.value)


def test_xpath_match() -> None:
    n = XpathNested("test", origin=origin)
    m1 = XpathMiddle(n, origin=origin)
    n2 = XpathNestedSub("test2", origin=origin)
    m2 = XpathMiddle(n2, origin=origin)
    mm = XpathMiddle(m1, origin=origin)
    _ = XpathRoot((mm, m2), origin=origin)

    # Shape of the tree:
    # XpathRoot
    #   XpathMiddle (mm)
    #     XpathMiddle (m1)
    #       XpathNested (n)
    #   XpathMiddle (m2)
    #     XpathNestedSub (n2)

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


def test_xpath_find() -> None:
    n = XpathNested("test", origin=origin)
    m1 = XpathMiddle(n, origin=origin)
    n2 = XpathNestedSub("test2", origin=origin)
    m2 = XpathMiddle(n2, origin=origin)
    mm = XpathMiddle(m1, origin=origin)
    r = XpathRoot((mm, m2), origin=origin)

    # Shape of the tree:
    # XpathRoot (r)
    #   XpathMiddle (mm)
    #     XpathMiddle (m1)
    #       XpathNested (n)
    #   XpathMiddle (m2)
    #     XpathNestedSub (n2)

    xpath = ASTXpath("//XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n.id, n2.id}

    xpath = ASTXpath("/XpathRoot/XpathMiddle/XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n2.id}

    xpath = ASTXpath("/XpathRoot//XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n.id, n2.id}

    xpath = ASTXpath("/XpathRoot/[0]XpathMiddle//XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n.id}

    xpath = ASTXpath("/XpathRoot/[]XpathMiddle//XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n.id, n2.id}

    xpath = ASTXpath("//@middle_tuple/@nested[]XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n2.id}

    xpath = ASTXpath("@middle_tuple/@nested[]XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n2.id}

    xpath = ASTXpath("//@middle_tuple/XpathMiddle/@nested[]XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n.id}

    xpath = ASTXpath("@middle_tuple/XpathMiddle/@nested[]XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n.id}
