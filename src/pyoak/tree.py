from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, NamedTuple, TypeVar, cast

if TYPE_CHECKING:
    from .node import ASTNode
    from .typing import Field

_AT = TypeVar("_AT", bound="ASTNode")


class ParentInfo(NamedTuple):
    parent: ASTNode
    field: Field
    findex: int | None


class Tree:
    def __init__(self, root: ASTNode) -> None:
        self._root = root
        self._node_to_parent_info: dict[ASTNode, ParentInfo] = {}
        self._node_to_xpath: dict[ASTNode, str] = {root: f"/@root[0]{root.__class__.__name__}"}

        for n in root.dfs():
            self._node_to_parent_info[n.node] = ParentInfo(n.parent, n.field, n.findex)
            self._node_to_xpath[
                n.node
            ] = f"{self._node_to_xpath[n.parent]}/@{n.field.name}[{n.findex or '0'}]{n.node.__class__.__name__}"

    @property
    def root(self) -> ASTNode:
        return self._root

    def get_xpath(self, node: ASTNode) -> str:
        """Get the XPath of a node as a str.

        Args:
            node (ASTNode): The node to get the XPath of.

        Raises:
            KeyError: If the node is not in the tree.

        Returns:
            str: The XPath of the node.
        """
        return self._node_to_xpath[node]

    def get_parent(self, node: ASTNode) -> ASTNode | None:
        """Get the parent of the `node`.

        Args:
            node (ASTNode): The node to get the parent of.

        Raises:
            KeyError: If the node is not in the tree.

        Returns:
            ASTNode | None: The parent node or None if the node is the root.
        """
        if node is self._root:
            return None

        return self._node_to_parent_info[node].parent

    def get_parent_info(self, node: ASTNode) -> tuple[ASTNode | None, Field | None, int | None]:
        """Get a tuple if parent, parent field & index in the parent field.

        Args:
            node (ASTNode): The node to get the parent info of.

        Raises:
            KeyError: If the node is not in the tree.

        Returns:
            tuple[ASTNode, Field, int | None]: The parent info.
        """
        if node is self._root:
            return None, None, None

        return self._node_to_parent_info[node]

    def is_root(self, node: ASTNode) -> bool:
        """Return True if the node is the root of the tree."""
        return self._root is node

    def is_in_tree(self, node: ASTNode) -> bool:
        """Return True if the node is in the tree."""
        return node in self._node_to_xpath

    def get_depth(
        self, node: ASTNode, relative_to: ASTNode | None = None, check_ancestor: bool = True
    ) -> int:
        """Returns the depth of `node` in the tree either up to root or up to
        `relative_to` node (if it is the ancestor at all).

        Args:
            relative_to (ASTNode | None): The node to stop at. If None, the depth is calculated
                up to the root node.
            check_ancestor (bool, optional): Whether to check if `relative_to` is an ancestor of
                the node. Defaults to True.

        Returns:
            int: The depth of this node in the tree.

        Raises:
            ValueError: If `relative_to` is not an ancestor of this node.
            KeyError: If the node is not in the tree.
        """
        if relative_to is not None and check_ancestor and not self.is_ancestor(node, relative_to):
            raise ValueError("relative_to must be an ancestor of the node")

        parent = self.get_parent(node)
        if parent is None:
            return 0

        if relative_to is not None and parent is relative_to:
            return 1

        return self.get_depth(parent, relative_to, False) + 1

    def get_ancestors(self, node: ASTNode) -> Iterator[ASTNode]:
        """Iterates over all ancestors of the `node`.

        Args:
            node (ASTNode): The node to get ancestors of.

        Yields:
            Iterator[ASTNode]: The ancestors of the node.

        Raises:
            KeyError: If the node is not in the tree.
        """
        parent = self.get_parent(node)
        while parent is not None:
            yield parent
            parent = self.get_parent(parent)

    def get_first_ancestor_of_type(
        self,
        node: ASTNode,
        ancestor_class: type[_AT] | tuple[type[_AT], ...],
        *,
        exact_type: bool = False,
    ) -> _AT | None:
        """Returns the first ancestor of the `node` that is an instance of
        `ancestor_class`. Or None if no such ancestor exists.

        Args:
            node (ASTNode): The node to get ancestors of.
            ancestor_class (type[ASTNode] | tuple[type[ASTNode], ...]): The
                ancestor class or tuple of classes to search for.
            exact_type (bool, optional): Whether to search for exact type match,
                or match any subclasses (isintance check). Defaults to False.

        Returns:
            _AT | None: The ancestor node or None if no such ancestor exists.

        Raises:
            KeyError: If the node is not in the tree.
        """
        if not isinstance(ancestor_class, tuple):
            ancestor_classes = cast(tuple[type[_AT], ...], (ancestor_class,))
        else:
            ancestor_classes = ancestor_class

        for ancestor in self.get_ancestors(node):
            if exact_type and type(ancestor) in ancestor_classes:
                return cast(_AT, ancestor)

            if not exact_type and isinstance(ancestor, ancestor_classes):
                return cast(_AT, ancestor)

        return None

    def is_ancestor(self, node: ASTNode, ancestor: ASTNode) -> bool:
        """Returns True if the `ancestor` is an ancestor of the `node` in the
        tree.

        Raises:
            KeyError: If the node is not in the tree.
        """
        for a in self.get_ancestors(node):
            if a is ancestor:
                return True

        return False
