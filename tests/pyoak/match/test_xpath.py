from __future__ import annotations

from dataclasses import dataclass

import pytest
from pyoak.match.error import ASTXpathOrPatternDefinitionError
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
    left: XpathNested | XpathMiddle | None
    right: XpathNested | XpathMiddle | None
    value: str


@dataclass
class XpathRoot(ASTNode):
    middle_tuple: tuple[XpathMiddle, ...]


def test_init() -> None:
    xpath = ASTXpath("//XpathNested")

    # Test caching
    assert xpath is ASTXpath("//XpathNested")
    assert xpath is not ASTXpath("//XpathNestedSub")

    # Test non-existent class
    with pytest.raises(ASTXpathOrPatternDefinitionError) as excinfo:
        ASTXpath("NonExistentClass")

    assert "NonExistentClass" in str(excinfo.value)

    # Test invalid XPath
    with pytest.raises(ASTXpathOrPatternDefinitionError) as excinfo:
        ASTXpath("//")

    assert "Incorrect xpath definition" in str(excinfo.value)


def test_xpath_match() -> None:
    n = XpathNested("test", origin=origin)
    n1 = XpathNested("test", origin=origin)
    m1_left = XpathMiddle(n, None, "mid m1 left", origin=origin)
    m1_right = XpathMiddle(n1, None, "mid m1 right", origin=origin)
    n2 = XpathNestedSub("test2", origin=origin)
    m2 = XpathMiddle(n2, None, "mid m2", origin=origin)
    mm = XpathMiddle(m1_left, m1_right, "mid mm", origin=origin)
    _ = XpathRoot((mm, m2), origin=origin)

    # Shape of the tree:
    # XpathRoot
    #   @middle_tuple[0]XpathMiddle (mm)
    #     @left[]XpathMiddle (m1_left)
    #       @left[]XpathNested (n)
    #     @right[]XpathMiddle (m1_right)
    #       @left[]XpathNested (n1)
    #   @middle_tuple[1]XpathMiddle (m2)
    #     @left[]XpathNestedSub (n2)

    xpath = ASTXpath("//XpathNested")
    assert xpath.match(n2)
    assert xpath.match(n1)
    assert xpath.match(n)

    xpath = ASTXpath("/XpathRoot/XpathMiddle/XpathNested")
    assert xpath.match(n2)
    assert not xpath.match(n1)
    assert not xpath.match(n)

    xpath = ASTXpath("/XpathRoot//XpathNested")
    assert xpath.match(n2)
    assert xpath.match(n1)
    assert xpath.match(n)

    xpath = ASTXpath("/XpathRoot/[0]XpathMiddle//XpathNested")
    assert not xpath.match(n2)
    assert xpath.match(n1)
    assert xpath.match(n)

    xpath = ASTXpath("/XpathRoot/[]XpathMiddle//XpathNested")
    assert xpath.match(n2)
    assert xpath.match(n1)
    assert xpath.match(n)

    xpath = ASTXpath("//@middle_tuple/@left[]XpathNested")
    assert xpath.match(n2)
    assert not xpath.match(n)

    xpath = ASTXpath("@middle_tuple/@left[]XpathNested")
    assert xpath.match(n2)
    assert not xpath.match(n1)
    assert not xpath.match(n)

    xpath = ASTXpath("@middle_tuple//@left[]XpathNested")
    assert xpath.match(n2)
    assert xpath.match(n1)
    assert xpath.match(n)

    xpath = ASTXpath("//@middle_tuple/XpathMiddle/@left[]XpathNested")
    assert not xpath.match(n2)
    assert xpath.match(n)

    xpath = ASTXpath("@middle_tuple/XpathMiddle/@left[]XpathNested")
    assert not xpath.match(n2)
    assert xpath.match(n)

    # With subpatterns
    xpath = ASTXpath('//(XpathMiddle @value=".*left")//XpathNested')
    assert xpath.match(n)
    assert not xpath.match(n2)


def test_xpath_find() -> None:
    n = XpathNested("test", origin=origin)
    n1 = XpathNested("test", origin=origin)
    m1_left = XpathMiddle(n, None, "mid m1 left", origin=origin)
    m1_right = XpathMiddle(n1, None, "mid m1 right", origin=origin)
    n2 = XpathNestedSub("test2", origin=origin)
    m2 = XpathMiddle(n2, None, "mid m2", origin=origin)
    mm = XpathMiddle(m1_left, m1_right, "mid mm", origin=origin)
    r = XpathRoot((mm, m2), origin=origin)

    # Shape of the tree:
    # XpathRoot
    #   @middle_tuple[0]XpathMiddle (mm)
    #     @left[]XpathMiddle (m1_left)
    #       @left[]XpathNested (n)
    #     @right[]XpathMiddle (m1_right)
    #       @left[]XpathNested (n1)
    #   @middle_tuple[1]XpathMiddle (m2)
    #     @left[]XpathNestedSub (n2)

    xpath = ASTXpath("//XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n.id, n1.id, n2.id}

    xpath = ASTXpath("/XpathRoot/XpathMiddle/XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n2.id}

    xpath = ASTXpath("/XpathRoot//XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n.id, n1.id, n2.id}

    xpath = ASTXpath("/XpathRoot/[0]XpathMiddle//XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n.id, n1.id}

    xpath = ASTXpath("/XpathRoot/[]XpathMiddle//XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n.id, n1.id, n2.id}

    xpath = ASTXpath("//@middle_tuple/@left[]XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n2.id}

    xpath = ASTXpath("@middle_tuple/@left[]XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n2.id}

    xpath = ASTXpath("//@middle_tuple/@right[]XpathMiddle/@left[]XpathNested")
    assert set(node.id for node in xpath.findall(r)) == {n1.id}

    xpath = ASTXpath("@middle_tuple/@left[]ASTNode")
    assert set(node.id for node in xpath.findall(r)) == {m1_left.id, n2.id}

    # With subpatterns
    xpath = ASTXpath('//(XpathMiddle @value=".*left")//XpathNested')
    assert set(node.id for node in xpath.findall(r)) == {n.id}
