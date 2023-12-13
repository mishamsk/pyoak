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


def test_match_checks() -> None:
    n = XpathNested("test", origin=origin)
    other_n = XpathNested("test", origin=origin)

    # Test node matching not belonging to the tree
    with pytest.raises(ValueError) as excinfo:
        ASTXpath("//XpathNested").match(n, other_n)

    assert "The node is not in the tree" in str(excinfo.value)


def test_xpath_match() -> None:
    n = XpathNested("test", origin=origin)
    m1 = XpathMiddle(n, origin=origin)
    n2 = XpathNestedSub("test2", origin=origin)
    m2 = XpathMiddle(n2, origin=origin)
    mm = XpathMiddle(m1, origin=origin)
    r = XpathRoot((mm, m2), origin=origin)
    r_tree = r.to_tree()
    # Shape of the tree:
    # XpathRoot (r)
    #   XpathMiddle (mm)
    #     XpathMiddle (m1)
    #       XpathNested (n)
    #   XpathMiddle (m2)
    #     XpathNestedSub (n2)

    xpath = ASTXpath("//XpathNested")
    assert xpath.match(r_tree, n2)
    assert xpath.match(r_tree, n)

    # Also one test from root node instead of tree
    assert xpath.match(r, n2)
    assert xpath.match(r, n)

    xpath = ASTXpath("/XpathRoot/XpathMiddle/XpathNested")
    assert xpath.match(r_tree, n2)
    assert not xpath.match(r_tree, n)

    xpath = ASTXpath("/XpathRoot//XpathNested")
    assert xpath.match(r_tree, n2)
    assert xpath.match(r_tree, n)

    xpath = ASTXpath("/XpathRoot/[0]XpathMiddle//XpathNested")
    assert not xpath.match(r_tree, n2)
    assert xpath.match(r_tree, n)

    xpath = ASTXpath("/XpathRoot/[]XpathMiddle//XpathNested")
    assert xpath.match(r_tree, n2)
    assert xpath.match(r_tree, n)

    xpath = ASTXpath("//@middle_tuple/@nested[]XpathNested")
    assert xpath.match(r_tree, n2)
    assert not xpath.match(r_tree, n)

    xpath = ASTXpath("@middle_tuple/@nested[]XpathNested")
    assert xpath.match(r_tree, n2)
    assert not xpath.match(r_tree, n)

    xpath = ASTXpath("//@middle_tuple/XpathMiddle/@nested[]XpathNested")
    assert not xpath.match(r_tree, n2)
    assert xpath.match(r_tree, n)

    xpath = ASTXpath("@middle_tuple/XpathMiddle/@nested[]XpathNested")
    assert not xpath.match(r_tree, n2)
    assert xpath.match(r_tree, n)


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
    assert set(xpath.findall(r)) == {n2, n}

    xpath = ASTXpath("/XpathRoot/XpathMiddle/XpathNested")
    assert set(xpath.findall(r)) == {n2}

    xpath = ASTXpath("/XpathRoot//XpathNested")
    assert set(xpath.findall(r)) == {n2, n}

    xpath = ASTXpath("/XpathRoot/[0]XpathMiddle//XpathNested")
    assert set(xpath.findall(r)) == {n}

    xpath = ASTXpath("/XpathRoot/[]XpathMiddle//XpathNested")
    assert set(xpath.findall(r)) == {n2, n}

    xpath = ASTXpath("//@middle_tuple/@nested[]XpathNested")
    assert set(xpath.findall(r)) == {n2}

    xpath = ASTXpath("@middle_tuple/@nested[]XpathNested")
    assert set(xpath.findall(r)) == {n2}

    xpath = ASTXpath("//@middle_tuple/XpathMiddle/@nested[]XpathNested")
    assert set(xpath.findall(r)) == {n}

    xpath = ASTXpath("@middle_tuple/XpathMiddle/@nested[]XpathNested")
    assert set(xpath.findall(r)) == {n}
