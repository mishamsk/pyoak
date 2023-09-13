from abc import ABC, abstractmethod
from dataclasses import replace
from inspect import getmembers, isfunction
from operator import itemgetter
from typing import Any, Generic, Mapping, TypeVar

from .node import ASTNode

_VRT = TypeVar("_VRT")


class ASTVisitor(Generic[_VRT], ABC):
    """A visitor generic base class for an AST visitor.

    Args:
        t (_type_): Vistior return type
    """

    strict: bool = False
    """Strict visitors match visit methods to nodes by exact type.

    Non-strict visitors will match by isinstance check in MRO order.
    """

    @abstractmethod
    def generic_visit(self, node: ASTNode) -> _VRT:
        raise NotImplementedError

    def visit(self, node: ASTNode) -> _VRT:
        """Visits the given node and returns the result."""
        return node.accept(self)

    def __init_subclass__(cls, *, validate: bool = False) -> None:
        """Iterate over new visitor methods and check that names match the node
        type annotation."""
        if validate:
            mismatched_pairs: list[tuple[str, str]] = []
            for method_name, method in getmembers(cls, isfunction):
                if method_name.startswith("visit_"):
                    expected_node_type = method_name[6:]
                    node_type_annotation = method.__annotations__.get("node", None)
                    if node_type_annotation is not None:
                        node_type_str = (
                            node_type_annotation
                            if isinstance(node_type_annotation, str)
                            else node_type_annotation.__name__
                        )

                        if expected_node_type != node_type_str:
                            mismatched_pairs.append((method_name, node_type_str))

            if mismatched_pairs:
                raise TypeError(
                    f"Visitor class '{cls.__name__}' method(s) '{', '.join(itemgetter(0)(pair) for pair in mismatched_pairs)}' do not match node type annotation(s) '{', '.join(itemgetter(1)(pair) for pair in mismatched_pairs)}'"
                )

        return super().__init_subclass__()


class ASTTransformVisitor(ASTVisitor[ASTNode | None]):
    """A visitor that transforms an AST by applying changes to its nodes.

    Methods:
        _transform_children: Transforms the children of a given node and returns a
            dictionary with the changes suitable to be passed to ASTNode.replace
            method.
        generic_visit: Transforms the children of the given node and returns a new
            node with the changes.
        visit: alias of transform. Prefer `transform`.
        transform: Transforms a given node and returns the transformed node or None
            if the node was removed.

    Raises:
        ASTTransformError: If the transformation fails with the original exception
            as context.
    """

    def _transform_children(self, node: ASTNode) -> Mapping[str, Any]:
        """Transforms the children of a given node and returns a mapping of
        field names to changes.

        This mapping can be passed to ASTNode.replace method.
        """
        changes: dict[str, ASTNode | None | list[ASTNode] | tuple[ASTNode, ...]] = {}
        field_names_with_changes = set()

        # Iterate over all child nodes and collect changes
        for child, f, index in node.get_child_nodes_with_field():
            fname = f.name
            if index is not None:
                # child field with a sequence
                # we need to store both changes and unchanged nodes to create a new sequence
                if fname not in changes:
                    changes[fname] = []

                new_child = self.visit(child)

                if new_child is not None:
                    changes[fname].append(new_child)  # type: ignore[union-attr]

                    if new_child is not child:
                        # New child, mark as changed field
                        field_names_with_changes.add(fname)
                else:
                    # Removed child, mark as changed field
                    field_names_with_changes.add(fname)
            else:
                new_child = self.transform(child)

                changes[fname] = new_child

                if new_child is not child:
                    # New child, mark as changed field
                    field_names_with_changes.add(fname)

        if not field_names_with_changes:
            # No changes, return empty dict
            return {}

        # Remove unchanged fields
        changes = {fname: changes[fname] for fname in field_names_with_changes}

        for fname in changes:
            val = changes[fname]
            if isinstance(val, list):
                # For sequences enforce tuple
                changes[fname] = tuple(val)

        # Return the changes
        return changes

    def generic_visit(self, node: ASTNode) -> ASTNode | None:
        """Transforms children of the given node and returns a new node with
        the changes."""

        changes = self._transform_children(node)

        # No changes, return the original node
        if not changes:
            return node

        # Return a new node with the changes
        return replace(node, **changes)

    def transform(self, node: ASTNode) -> ASTNode | None:
        """Am alias for visit method."""
        return self.visit(node)
