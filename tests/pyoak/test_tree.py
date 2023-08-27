from __future__ import annotations

from dataclasses import dataclass

import pytest
from pyoak.node import ASTNode


@dataclass(frozen=True)
class TreeOtherNode(ASTNode):
    attr: str


@dataclass(frozen=True)
class TreeNested(ASTNode):
    attr: str


@dataclass(frozen=True)
class TreeMiddle(ASTNode):
    nested: TreeNested | TreeMiddle


@dataclass(frozen=True)
class TreeRoot(ASTNode):
    middle_tuple: tuple[TreeMiddle, ...]


@dataclass(frozen=True)
class TreeMiddleSubclass(TreeMiddle):
    pass


def test_get_xpath() -> None:
    n = TreeNested("test")
    n_tree = n.to_tree()

    assert n_tree.get_xpath(n) == "/@root[0]TreeNested"

    m1 = TreeMiddle(n)
    m1_tree = m1.to_tree()

    assert m1_tree.get_xpath(m1) == "/@root[0]TreeMiddle"
    assert m1_tree.get_xpath(n) == "/@root[0]TreeMiddle/@nested[0]TreeNested"
    # Test that the original tree is not affected
    assert n_tree.get_xpath(n) == "/@root[0]TreeNested"

    n2 = TreeNested("test2")
    m2 = TreeMiddle(n2)
    r = TreeRoot((m1, m2))
    r_tree = r.to_tree()

    assert r_tree.get_xpath(r) == "/@root[0]TreeRoot"
    assert r_tree.get_xpath(m1) == "/@root[0]TreeRoot/@middle_tuple[0]TreeMiddle"
    assert r_tree.get_xpath(m2) == "/@root[0]TreeRoot/@middle_tuple[1]TreeMiddle"
    assert (
        r_tree.get_xpath(n2) == "/@root[0]TreeRoot/@middle_tuple[1]TreeMiddle/@nested[0]TreeNested"
    )


def test_parent() -> None:
    n1 = TreeNested("test1")
    m_m = TreeMiddleSubclass(n1)
    m_r = TreeMiddle(m_m)
    r = TreeRoot((m_r,))
    r_tree = r.to_tree()

    assert r_tree.parent(n1) is m_m
    assert r_tree.parent(m_m) is m_r
    assert r_tree.parent(m_r) is r
    assert r_tree.parent(r) is None

    with pytest.raises(KeyError):
        r_tree.parent(TreeOtherNode("test"))


def test_parent_info() -> None:
    n1 = TreeNested("test1")
    m_m = TreeMiddleSubclass(n1)
    m_r = TreeMiddle(m_m)
    r = TreeRoot((m_r,))
    r_tree = r.to_tree()

    assert r_tree.parent_info(n1) == (m_m, TreeMiddleSubclass.__dataclass_fields__["nested"], None)
    assert r_tree.parent_info(m_m) == (m_r, TreeMiddle.__dataclass_fields__["nested"], None)
    assert r_tree.parent_info(m_r) == (r, TreeRoot.__dataclass_fields__["middle_tuple"], 0)
    assert r_tree.parent_info(r) == (None, None, None)

    with pytest.raises(KeyError):
        r_tree.parent_info(TreeOtherNode("test"))


def test_is_root() -> None:
    n = TreeNested("test1")
    assert n.to_tree().is_root(n)


def test_is_in_tree() -> None:
    n1 = TreeNested("test1")
    m_m = TreeMiddleSubclass(n1)
    m_r = TreeMiddle(m_m)
    n2 = TreeNested("test2")
    m2 = TreeMiddle(n2)
    r = TreeRoot((m_r, m2))
    r_tree = r.to_tree()
    assert all(r_tree.is_in_tree(n) for n in (n1, m_m, m_r, n2, m2, r))
    assert not r_tree.is_in_tree(TreeOtherNode("test"))


def test_get_depth() -> None:
    n1 = TreeNested("test1")
    m_m = TreeMiddleSubclass(n1)
    m_r = TreeMiddle(m_m)
    n2 = TreeNested("test2")
    m2 = TreeMiddle(n2)
    r = TreeRoot((m_r, m2))
    r_tree = r.to_tree()

    assert r_tree.get_depth(n1) == 3
    assert r_tree.get_depth(n1, m_m) == 1

    with pytest.raises(ValueError):
        # not an ancestor
        assert r_tree.get_depth(n1, m2) == 3

    assert r_tree.get_depth(n1, m2, check_ancestor=False) == 3

    assert r_tree.get_depth(m_m) == 2
    assert r_tree.get_depth(m_r) == 1
    assert r_tree.get_depth(r) == 0

    with pytest.raises(KeyError):
        r_tree.get_depth(TreeOtherNode("test"))


def test_ancestors() -> None:
    n1 = TreeNested("test1")
    m_m = TreeMiddleSubclass(n1)
    m_r = TreeMiddle(m_m)
    r = TreeRoot((m_r,))
    r_tree = r.to_tree()

    assert tuple(r_tree.ancestors(n1)) == (m_m, m_r, r)
    assert tuple(r_tree.ancestors(m_m)) == (m_r, r)
    assert tuple(r_tree.ancestors(m_r)) == (r,)
    assert tuple(r_tree.ancestors(r)) == ()

    with pytest.raises(KeyError):
        next(r_tree.ancestors(TreeOtherNode("test")))


def test_get_first_ancestor_of_type() -> None:
    n1 = TreeNested("test1")
    m_m = TreeMiddleSubclass(n1)
    m_r = TreeMiddle(m_m)
    r = TreeRoot((m_r,))
    r_tree = r.to_tree()

    assert r_tree.get_first_ancestor_of_type(n1, TreeMiddle) is m_m
    assert r_tree.get_first_ancestor_of_type(n1, TreeMiddle, exact_type=True) is m_r
    assert r_tree.get_first_ancestor_of_type(n1, TreeRoot) is r
    assert r_tree.get_first_ancestor_of_type(m_m, TreeMiddle) is m_r
    assert r_tree.get_first_ancestor_of_type(m_m, TreeMiddle, exact_type=True) is m_r
    assert r_tree.get_first_ancestor_of_type(m_m, TreeOtherNode) is None
    assert r_tree.get_first_ancestor_of_type(m_m, TreeOtherNode, exact_type=True) is None


def test_is_ancestor() -> None:
    n1 = TreeNested("test1")
    m_m = TreeMiddleSubclass(n1)
    m_r = TreeMiddle(m_m)
    n2 = TreeNested("test2")
    m2 = TreeMiddle(n2)
    r = TreeRoot((m_r, m2))
    r_tree = r.to_tree()

    assert r_tree.is_ancestor(n1, m_m)
    assert r_tree.is_ancestor(n1, m_r)
    assert r_tree.is_ancestor(n1, r)
    assert not r_tree.is_ancestor(n1, n2)
    assert not r_tree.is_ancestor(n1, m2)
