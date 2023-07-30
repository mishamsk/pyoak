from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Type

if TYPE_CHECKING:
    from pyoak.node import ASTNode


class ASTNodeError(Exception):
    """Base class for all ASTNode errors."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message})"


class ASTNodeRegistryCollisionError(ASTNodeError):
    """Raised when trying to create a new or attach a detached node to a registry, but there is
    already an existing ASTNode with the same ID."""

    def __init__(
        self,
        existing_node: ASTNode,
        new_node: ASTNode,
        operation: Literal["create", "attach", "replace"],
    ) -> None:
        msg = (
            f"You've tried to {operation} an AST node of type <{new_node.__class__.__name__}>"
            f" with id <{new_node.id}>, but there is already an existing attached node of type"
            f" <{existing_node.__class__.__name__}> with the same id in the registry."
            f" Make sure the existing node is detached before trying to {operation} a "
            "new one with the same id"
        )
        super().__init__(
            msg,
            existing_node,
            new_node,
            operation,
        )
        self.existing_node = existing_node
        self.new_node = new_node
        self.operation = operation


class ASTNodeIDCollisionError(ASTNodeError):
    """Raised when an ASTNode ID collision occurs."""

    def __init__(self, existing_node: ASTNode, new_node_class: Type[ASTNode]) -> None:
        super().__init__(
            f"ID Collision with an existing node type <{existing_node.__class__.__name__}> from <{existing_node.origin.fqn}>. Please use {new_node_class.__name__}.replace() method",
            existing_node,
            new_node_class,
        )
        self.existing_node = existing_node
        self.new_node_class = new_node_class


class ASTNodeParentCollisionError(ASTNodeError):
    """Raised when an ASTNode parent collision occurs."""

    def __init__(
        self,
        new_node: ASTNode,
        collided_child: ASTNode,
        collided_parent: ASTNode,
    ) -> None:
        super().__init__(
            f"Tried to create an AST Node of type <{new_node.__class__.__name__}> with id <{new_node.id}>, but some of it's children id's collided with existing ones in the registry. At least child node of type <{collided_child.__class__.__name__}> with id <{collided_child.id}> already has a different parent of type <{collided_parent.__class__.__name__}> with id <{collided_parent.id}>. Consider using {new_node.__class__.__name__}.detach() method",
            new_node,
            collided_child,
            collided_parent,
        )
        self.new_node = new_node
        self.collided_child = collided_child
        self.collided_parent = collided_parent


class ASTNodeDuplicateChildrenError(ASTNodeError):
    """Raised when an ASTNode has the same child more than once among children."""

    def __init__(
        self,
        child: ASTNode,
        last_field_name: str,
        last_index: int | None,
        new_field_name: str,
        new_index: int | None,
    ) -> None:
        super().__init__(
            f"You've tried to create a new AST node with a child node of type <{child.__class__.__name__}> with id <{child.id}> appearing multiple times among children. First time in field <{last_field_name}> {'at index <' + str(last_index) + '>' if last_index else ''}. The new time it appears is in field <{new_field_name}> {'at index <' + str(new_index) + '>' if new_index else ''}. This is not allowed. Consider using {child.__class__.__name__}.detach() method",
            child,
            last_field_name,
            last_index,
            new_field_name,
            new_index,
        )
        self.child = child
        self.last_field_name = last_field_name
        self.last_index = last_index
        self.new_field_name = new_field_name
        self.new_index = new_index


class ASTNodeReplaceError(ASTNodeError):
    """Raised when an ASTNode replace method fail."""

    def __init__(
        self,
        node: ASTNode,
        changes: dict[str, Any],
        error_keys: list[str],
    ) -> None:
        super().__init__(
            f"The following fields can't be replaced using this function: {', '.join(error_keys)}",
            node,
            changes,
            error_keys,
        )
        self.node = node
        self.changes = changes
        self.error_keys = error_keys


class ASTNodeReplaceWithError(ASTNodeError):
    """Raised when an ASTNode replace_with method fail."""

    def __init__(
        self,
        message: str,
        node: ASTNode,
        new_node: ASTNode | None,
    ) -> None:
        super().__init__(
            message,
            node,
            new_node,
        )
        self.node = node
        self.new_node = new_node


class ASTTransformError(Exception):
    """Error indicating incorrect transformation applied."""

    def __init__(self, orig_node: ASTNode, transformed_node: ASTNode | None) -> None:
        super().__init__(
            f"Transform for node of type <{orig_node.__class__.__name__}> "
            f"with id <{orig_node.id}> failed due to an upstream exception.",
            orig_node,
            transformed_node,
        )
        self.orig_node = orig_node
        self.transformed_node = transformed_node
