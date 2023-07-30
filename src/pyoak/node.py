from __future__ import annotations

import enum
import hashlib
import logging
import typing as t
import weakref
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Generator, Iterable
from dataclasses import Field as DataClassField
from dataclasses import InitVar, dataclass, field, fields, replace
from inspect import getmembers, getmro, isfunction
from operator import itemgetter

from rich.markup import escape
from rich.tree import Tree

from .error import (
    ASTNodeDuplicateChildrenError,
    ASTNodeIDCollisionError,
    ASTNodeParentCollisionError,
    ASTNodeRegistryCollisionError,
    ASTNodeReplaceError,
    ASTNodeReplaceWithError,
    ASTTransformError,
)
from .helpers import ChildFieldTypeInfo, get_ast_node_child_fields, get_ast_node_properties
from .origin import Origin
from .serialize import TYPE_KEY, DataClassSerializeMixin

logger = logging.getLogger(__name__)

TRACE_LOGGING = False

# to make mypy happy
Field = DataClassField[t.Any]

VisitorReturnType = t.TypeVar("VisitorReturnType")

ChildIndex = int

UNSET_ID = "~~~UNSET~~~"  # Hopefully no one will ever use this as an ID
CONTENT_ID_FIELD = "_content_id"

# Hack to make dataclasses InitVar work with future annotations
# See https://stackoverflow.com/questions/70400639/how-do-i-get-python-dataclass-initvar-fields-to-work-with-typing-get-type-hints
InitVar.__call__ = lambda *args: None  # type: ignore


def _get_next_unique_id(id_: str) -> str:
    """Return a unique ID."""
    i = 1
    original_id = id_
    while ASTNode._nodes.get(id_) is not None:
        id_ = f"{original_id}_{i}"
        i += 1

    return id_


def _is_skip_field(
    field: Field,
    skip_id: bool,
    skip_origin: bool,
    skip_original_id: bool,
    skip_id_collision_with: bool,
    skip_hidden: bool,
    skip_non_compare: bool,
    skip_non_init: bool,
) -> bool:
    """Check if a field should be skipped."""
    # Skip id
    if (field.name == "id" or field.name == "content_id") and skip_id:
        return True

    # Skip origin
    if field.name == "origin" and skip_origin:
        return True

    # Skip original id
    if field.name == "original_id" and skip_original_id:
        return True

    # Skip id collision with
    if field.name == "id_collision_with" and skip_id_collision_with:
        return True

    # Skip hidden fields
    if field.name.startswith("_") and skip_hidden:
        return True

    # Skip non-comparable fields
    if not field.compare and skip_non_compare:
        return True

    # Skip non-init fields
    if not field.init and skip_non_init:
        return True

    return False


AST_SERIALIZE_DIALECT_KEY = "ast_serialize_dialect"


def _set_xpath(node: ASTNode, parent_xpath: str) -> None:
    """Set the xpath for the subtree."""
    if node.parent_field is None:
        raise RuntimeError("Parent field is not set in AST Node")

    xpath = f"{parent_xpath}/@{node.parent_field.name}[{node.parent_index or '0'}]{node.__class__.__name__}"

    object.__setattr__(node, "_xpath", xpath)

    for child in node.get_child_nodes():
        _set_xpath(child, xpath)


class ASTSerializationDialects(enum.Enum):
    AST_EXPLORER = enum.auto()
    AST_TEST = enum.auto()


@dataclass
class ASTNode(DataClassSerializeMixin):
    """A base class for all AST Node classes.

    Provides the following functions:
        - Maintains a weakref dictionary of all nodes, accessible through the class methods `get` and `get_any`
        - Maintains a parent-child relationship between nodes
        - Methods for replacing attributes or whole nodes
        - Various tree walking methods
        - Visitor API (visit method)
        - Serialization & deserialization
        - rich console API support (for printing)

    Notes:
        - Subclasses must be dataclasses
        - Only fields with subclasses of ASTNode or tuples/lists of ASTNode subclasses are considered children
        - fields starting with "_" are not serialized

    Raises:
        ASTNodeError: If there are children with a different parent alredy assigned (every AST node can only have one parent)

    """

    _nodes: t.ClassVar[weakref.WeakValueDictionary[str, ASTNode]] = weakref.WeakValueDictionary()
    """Registry of all node objects that are considered "attached"."""

    _props: t.ClassVar[t.Mapping[str, tuple[Field, t.Type[t.Any]]] | None] = None
    """Mapping of field names to field instances & resolved types that are considered properties
    (i.e. not children). This mapping is determined statically at based only on field type
    annotations.

    Most methods check whether a field is a property or not by checking instances at runtime

    """

    _child_fields: t.ClassVar[t.Mapping[str, tuple[Field, ChildFieldTypeInfo]] | None] = None
    """"Mapping of field names to field instances & resolved types that are considered children.
    This mapping is determined statically at based only on field type annotations.

    Most methods check whether a field is a child or not by checking instances at runtime

    """

    original_id: str | None = field(
        default=None,
        compare=False,
        kw_only=True,
    )
    """The "original" ID of this node. Set automatically, DO NOT SET MANUALLY.

    - If the node was duplicated for another node, it will be the original ID.
    - If the node is a replacement (replace_with), it will be the original ID of the node
    before it was used for replacement.

    None otherwise.

    """

    id_collision_with: str | None = field(
        default=None,
        compare=False,
        kw_only=True,
    )
    """An ID of attached node whose Id was the same when this node was created (with
    ensure_unique_id option) or None if there was no collision.

    Set automatically, DO NOT SET MANUALLY.

    """

    origin: Origin = field(kw_only=True)
    """The origin of this node.

    Provides information of the source (such as text file) and it's path, as well as position within
    the source (e.g. char index)

    """

    id: str = field(default=UNSET_ID, kw_only=True, compare=False)
    """The unique ID of this node. If not set in the constructor, it will be auto-generated based on
    the node's properties and origin.

    This means that two nodes with the same "content" (i.e. properties and childre)
    but generated from different sources will have different IDs.

    For a stable ID based purely on the node's content, use the `content_id` property.

    """

    content_id: str = field(
        default=UNSET_ID,
        init=False,
        compare=False,
    )
    """The ID of this node based on it's content (i.e. properties and children).

    Unlike the `id` property, this ID is stable and will be the same for two nodes
    with the same content, regardless of their origin.

    Naturally, this id can't be used to identify a node in the registry, since
    it is by definition non-unique.

    """

    ensure_unique_id: InitVar[bool] = field(default=False, kw_only=True)
    """When set to True, if a new node being created has non-unique ID an ASTNodeIDCollisionError
    exception is raised."""

    create_as_duplicate: InitVar[bool] = field(default=False, kw_only=True)
    """When set to True, if node collision registered, it is recorded as node duplication, not id
    collision. This is similar to getting the node from the registry and calling `duplicate` on it.

    This overrides `ensure_unique_id`.

    """

    create_detached: InitVar[bool] = field(default=False, kw_only=True)
    """When set to True, a node is not added to the registry upon creation. Nodes passed as children
    are not attached to the newly created node (will not have their parent set to the new node id)
    and id is not checked for collisions (id_collision_with, original_id are never set / ignored).

    This has the same effect as creating a normal node and then calling
    `detach_self` on it. But unlike the `detach_self` method, this option
    allows creating a node with an existing ID from the registry, and then
    using it as a replacement for attached node with the same ID.

    This is used as part of the transform visitor machinery when an
    attached subtrees is passed.

    """

    def __post_init__(
        self,
        ensure_unique_id: bool,
        create_as_duplicate: bool,
        create_detached: bool,
    ) -> None:
        # First ensure all children are unique. This will raise an error if not
        self._check_unique_children()

        # Calculate ID
        new_id: str
        id_collision_with: str | None = None
        original_id: str | None = None
        if self.id == UNSET_ID:
            hasher = hashlib.sha256()
            hasher.update(self.__class__.__name__.encode("utf-8"))
            hasher.update(f":{self.origin.fqn}".encode("utf-8"))
            for val, f in sorted(
                self.get_properties(skip_non_compare=False), key=lambda x: x[1].name
            ):
                hasher.update(f":{f.name}=".encode("utf-8"))
                hasher.update(str(val).encode("utf-8"))

            for c, f, i in sorted(
                self.get_child_nodes_with_field(), key=lambda x: (x[1].name, x[2] or -1)
            ):
                i = i or -1
                hasher.update(f":{f.name}[{i}]=".encode("utf-8"))
                hasher.update(f"{c.id}".encode("utf-8"))

            new_id = hasher.hexdigest()
        else:
            new_id = self.id

        if not create_detached:
            if (existing_node := ASTNode._nodes.get(new_id)) is not None:
                # Node with the same ID already exists
                if not ensure_unique_id or create_as_duplicate:
                    # We are ok if this is a duplicate node (contains the same data) but want to make sure that the ID is unique

                    if not create_as_duplicate:
                        id_collision_with = new_id
                    else:
                        original_id = new_id

                    new_id = _get_next_unique_id(new_id)
                else:
                    # Check and register in global registry
                    raise ASTNodeIDCollisionError(existing_node, self.__class__)

            # self.original_id & self.id_collision_with are supposed to be coming
            # from deserialization (user should not set them manually).
            # We need to handle various cases:
            if self.original_id is not None:
                # If node was a duplicate before serialization, we should
                # force treat it as a duplicate after deserialization
                original_id = self.original_id

                if id_collision_with is not None:
                    # If deserializaed node collided with some node
                    # it may mean two things:
                    # 1. It collided with another node that was a duplicate from the same original node
                    # 2. It collided with another node that was not a duplicate which unfortunately
                    #    used the ID matching the serialized node (should be extremely rare)
                    if id_collision_with == self.original_id or (
                        (collided := ASTNode._nodes.get(id_collision_with)) is not None
                        and collided.original_id == self.original_id
                    ):
                        # Case 1 - we should treat it as a duplicate
                        # Recalculate ID from the original base and
                        # reset collision information
                        new_id = _get_next_unique_id(self.original_id)
                        id_collision_with = None

            if self.id_collision_with is not None:
                # If a node had a collision with another node before serialization
                # we should use the same collision ID after deserialization

                if id_collision_with is not None:
                    # This means that the node collided after deserialization,
                    # it can't be the original id since this node was serialized with
                    # a unique ID... so again, same 2 cases as above
                    if (
                        collided := ASTNode._nodes.get(id_collision_with)
                    ) is not None and collided.original_id == self.id_collision_with:
                        # Case 1 Recalculate ID from the original base
                        new_id = _get_next_unique_id(self.id_collision_with)

                id_collision_with = self.id_collision_with

        object.__setattr__(self, "id", new_id)
        object.__setattr__(self, "id_collision_with", id_collision_with)
        object.__setattr__(self, "original_id", original_id)

        # Set parent related field & xpath to None
        self._clear_parent()

        if not create_detached:
            # Besides attaching oneself to the registry and assigning parent to children
            # It is possible that some children has been previously dropped (detached)
            # from the registry. We need to re-attach them which will also check for collisions
            self._attach(operation="create")

        self._set_content_id()

    def _clear_parent(self) -> None:
        object.__setattr__(self, "_parent_id", None)
        object.__setattr__(self, "_parent_field", None)
        object.__setattr__(self, "_parent_index", None)
        object.__setattr__(self, "_xpath", None)

    def _set_parent(self, parent: ASTNode, field: Field, index: int | None) -> None:
        object.__setattr__(self, "_parent_id", parent.id)
        object.__setattr__(self, "_parent_field", field)
        object.__setattr__(self, "_parent_index", index)

    def __post_serialize__(self, d: dict[str, t.Any]) -> dict[str, t.Any]:
        # Run first, otherwise _children will be dropped from the output
        out = super().__post_serialize__(d)

        if (
            self._get_serialization_options().get(AST_SERIALIZE_DIALECT_KEY)
            == ASTSerializationDialects.AST_EXPLORER
        ):
            out["_children"] = []
            out["_children"].extend([f.name for f in fields(self) if self._is_field_child(f)])

        if (
            self._get_serialization_options().get(AST_SERIALIZE_DIALECT_KEY)
            == ASTSerializationDialects.AST_TEST
        ):
            out.get("origin", {})["source"] = {
                TYPE_KEY: "Source",
                "source_uri": "",
                "source_type": "",
            }

        return out

    def _set_content_id(self) -> None:
        """Set the ID of this node based on its content.

        Unlike automatic node ids, these do not include origin information, hidden & non-compare
        fields and hence by definition represent the content of the node.

        """
        hasher = hashlib.sha256()
        hasher.update(self.__class__.__name__.encode("utf-8"))
        for val, f in sorted(
            self.get_properties(
                skip_id=True,
                skip_origin=True,
                skip_original_id=True,
                skip_id_collision_with=True,
                skip_hidden=True,
                skip_non_compare=True,
            ),
            key=lambda x: x[1].name,
        ):
            hasher.update(f":{f.name}=".encode("utf-8"))
            hasher.update(str(val).encode("utf-8"))

        for c, f, i in sorted(
            self.get_child_nodes_with_field(), key=lambda x: (x[1].name, x[2] or -1)
        ):
            resolved_index = i or -1
            hasher.update(f":{f.name}[{resolved_index}]=".encode("utf-8"))
            hasher.update(f"{getattr(c, 'content_id', '')}".encode("utf-8"))

        object.__setattr__(self, "content_id", hasher.hexdigest())

    def _reset_content_id(self) -> None:
        """Recursively reset the content_id of this node and all its parents."""
        node: ASTNode | None = self
        while node is not None:
            node._set_content_id()
            node = node.parent

    def _check_unique_children(self) -> None:
        seen = set()
        last_field_name = ""
        last_index = None
        for c, f, i in self.get_child_nodes_with_field():
            if c.id in seen:
                raise ASTNodeDuplicateChildrenError(
                    c,
                    last_field_name=last_field_name,
                    last_index=last_index,
                    new_field_name=f.name,
                    new_index=i,
                )
            last_field_name = f.name
            last_index = i
            seen.add(c.id)

    def _attach_inner(
        self, operation: t.Literal["create", "attach", "replace"]
    ) -> tuple[ASTNode, ASTNode] | None:
        """Assigns parent to all child nodes, checking that they don't have a different parent that
        is also in the registry (aka attached node).

        Then attaches this node and all of it's detached children to the AST registry.

        Returns:
            tuple[ASTNode, ASTNode] | None: If there is a collision, returns the first
            child node that collided and it's parent. None if the node was successfully attached

        Raises:
            ASTNodeRegistryCollisionError: If this node's id is already in the registry

        """

        # Check this id is not already in the registry
        if self.id in ASTNode._nodes:
            raise ASTNodeRegistryCollisionError(
                new_node=self,
                existing_node=ASTNode._nodes[self.id],
                operation=operation,
            )

        # Walk through children, who may be either:
        # * detached from registry entirely (result of a previsous `detach()` call)
        # * or attached roots, meaning they were just created and have no parent
        # If a child is attached to a different parent, it means an error has occured.
        for c, f, i in self.get_child_nodes_with_field():
            if c.detached:
                # Means an already existing but previously detached node was re-added as a child to this new node
                # Thus we need to recursively attach it
                if (ret := c._attach_inner(operation=operation)) is not None:
                    return ret
            elif not c.is_attached_root:
                # This means that a child node is already attached to a different parent
                assert c.parent is not None
                return (c, c.parent)

            c._set_parent(self, f, i)

        # Now we can safely attach this node to the registry
        ASTNode._nodes[self.id] = self

        return None

    def _attach(self, operation: t.Literal["create", "attach", "replace"]) -> None:
        if (ret := self._attach_inner(operation=operation)) is not None:
            c, p = ret
            raise ASTNodeParentCollisionError(self, c, p)

    def _replace_child(
        self, old: ASTNode, field: Field, index: int | None, new: ASTNode | None
    ) -> None:
        """Replaces an `old` child node in `field` at index `index` with a `new` one.

        Assumes that the old node has already been detached and
        the new one is not attached.

        Args:
            old (ASTNode): The old child node to replace
            field (Field): The field in which to replace the child
            index (int | None): The index of the child to replace if field is a list/tuple.
            new (ASTNode): The new child node

        """

        if index is not None:
            # This means the old node was a child in a list/tuple
            orig_seq = getattr(self, field.name)

            if new is not None:
                setattr(
                    self,
                    field.name,
                    type(orig_seq)([*orig_seq[:index], new, *orig_seq[index + 1 :]]),
                )
            else:
                # Remove the old node
                setattr(
                    self,
                    field.name,
                    type(orig_seq)([*orig_seq[:index], *orig_seq[index + 1 :]]),
                )

                # Now shift all the indexes of the children after the removed one
                for c in t.cast(t.Iterable[ASTNode], orig_seq[index + 1 :]):
                    c._set_parent(self, field, t.cast(int, c.parent_index) - 1)
        else:
            # This means the old node was a child in a field
            setattr(self, field.name, new)

        if new is not None:
            new._set_parent(self, field, index)

        # Child changed, so we need to recompute the content id
        # for this parent node and all its parents,
        # but only if child content id changed
        if new is None or old.content_id != new.content_id:
            self._reset_content_id()

    @classmethod
    def get(
        cls: t.Type[ASTNodeType],
        id: str,
        default: ASTNodeType | None = None,
        strict: bool = True,
    ) -> ASTNodeType | None:
        ret = ASTNode._nodes.get(id, default)
        if ret is None:
            return None
        elif strict and not type(ret) == cls:
            return None
        elif not strict and not isinstance(ret, cls):
            return None
        else:
            return t.cast(ASTNodeType, ret)

    @classmethod
    def get_any(cls, id: str, default: ASTNode | None = None) -> ASTNode | None:
        return ASTNode._nodes.get(id, default)

    def attach(self) -> None:
        """Attaches this node and all of it's detached children to the AST registry.

        Raises:
            ASTNodeRegistryCollisionError: If this node's or any of the subtree node id's
                are already in the registry
            ASTNodeParentCollisionError: If any of the subtree nodes are already
                attached to a different parent

        """
        if not self.detached:
            if TRACE_LOGGING:
                logger.debug(f"Tried to attach an attached AST Node <{self.id}> to AST registry")
            return

        self._attach(operation="attach")

    def detach(self, *, only_self: bool = False) -> bool:
        """Removes this node and all of it's children (unless `only_self` is True) from the AST
        registry.

        Args:
            only_self (bool, optional): If True, only this node will be removed from the registry. Defaults to False.

        Returns:
            bool: True if the node was successfully dropped, including if it was not found in the registry. False if it is a sub-tree of a registered node.

        """
        if self.detached:
            if TRACE_LOGGING:
                logger.debug(f"Tried to detach a detached AST Node <{self.id}> from AST registry")
            return True

        # Checking for parent, not for _parent (which is id) to see if it has a registered parent
        if not self.is_attached_root:
            if TRACE_LOGGING:
                logger.debug(
                    f"Tried to detach an attached sub-tree (i.e. has a parent) AST Node <{self.id}> from AST registry"
                )
            return False

        # Walk through children and remove parent reference
        # Although `self` will have object references in its attributes
        # from the attached tree perspective, the children are free floating
        # Unless we do this, we can have a situation after replace
        # where a former child is not among `self.get_child_nodes()`
        # but it's parent property still returns an actual object
        for c in self.get_child_nodes():
            c._clear_parent()

            if not only_self:
                c.detach()

        ASTNode._nodes.pop(self.id)

        return True

    def detach_self(self) -> bool:
        """Removes this node from the AST registry.

        Returns:
            bool: True if the node was successfully dropped, False if it was not found in the registry

        """
        return self.detach(only_self=True)

    def replace(self: ASTNodeType, **changes: t.Any) -> ASTNodeType:
        """This will create and return a new node with changes applied to it.

        if the node was attached to the registry, it will be replaced in the
        AST registry with the new one.

        This works similarly to dataclasses.replace, but with some differences:
        * This will update content_id of the node and all of it's parents if needed.
        * It is not allowed to change the following attributes:
            id, content_id, original_id, id_collision_with, fields with init=False.
        * In addition to creating a new instance, it will also replace the node in the AST registry
            and within it's parent if it has one.

        Currently this function doesn't validate the types of the changes you pass in to be
        compatible with the types of the fields you are trying to change. This may change in the future.

        Args:
            **changes: The changes to apply to the new node

        Raises:
            ASTNodeReplaceError: If you try to change the forbidden attributes or supply
                non-existent attributes
            ASTNodeParentCollisionError: If among changes you pass a new child with
                a different parent assigned (same as init)
            ASTNodeDuplicateChildrenError: same as init
            ASTNodeRegistryCollisionError: same as init

        """
        fields_allowed_for_replace = set(
            f.name
            for f in fields(self)
            if not _is_skip_field(
                f,
                skip_id=True,
                skip_origin=False,
                skip_original_id=True,
                skip_id_collision_with=True,
                skip_hidden=False,
                skip_non_compare=False,
                skip_non_init=True,
            )
        )

        change_keys = set(changes.keys())

        if not change_keys.issubset(fields_allowed_for_replace):
            raise ASTNodeReplaceError(
                node=self,
                changes=changes,
                error_keys=list(change_keys - fields_allowed_for_replace),
            )

        logger.debug(
            f"Replacing attributes <{','.join(changes.keys())}> in an existing AST Node <{self.id}>"
        )

        # remember the parent
        cur_parent = self.parent
        cur_parent_field = self.parent_field
        cur_parent_index = self.parent_index
        if cur_parent is not None:
            # If we have a parent, we need to clear it first,
            # Otherwise the call to detach() will fail because detaching
            # non-root nodes is not allowed. But this is a special case
            self._clear_parent()

        # First we need to detach all children, to make sure that those that are not part of
        # `changes` will not stay incorrectly attached and maintain this parent id
        if not self.detached:
            self.detach_self()

            was_attached = True
        else:
            was_attached = False

        # Now we can safely create a new node. All old children will be re-attached alongside new ones
        try:
            ret = replace(
                self,
                create_detached=not was_attached,
                original_id=None,  # Should not go to init, handled below
                id_collision_with=None,  # Should not go to init, handled below
                **changes,
            )
        except Exception as e:
            # If something went wrong, we need to re-attach the node to the parent
            # and back to the registry
            if was_attached:
                ASTNode._nodes[self.id] = self

            if cur_parent is not None:
                assert cur_parent_field is not None
                self._set_parent(cur_parent, cur_parent_field, cur_parent_index)

            raise e

        if cur_parent is not None:
            # If we had a parent, we need to replace this child with new one in it
            # and re-attach the node to the parent
            assert cur_parent_field is not None
            cur_parent._replace_child(self, cur_parent_field, cur_parent_index, ret)

        # We need to copy the original_id and id_collision_with
        object.__setattr__(ret, "original_id", self.original_id)
        object.__setattr__(ret, "id_collision_with", self.id_collision_with)
        return ret

    def replace_with(self: ASTNode, new: ASTNode | None) -> None:
        """Attempts to replace this node with a new one. If the new node has a different parent of
        if it's type does not match the type declaration of the parent node, the operation will
        raise ASTNodeReplaceWithError.

        Otherwise, the old node is detached and the new one is attached to the
        same parent and registry, and it's ID is changed to the old node ID. It's
        original ID is preserved in the `original_id` attribute.

        Args:
            new (ASTNode | None): The new node to replace this one with. If None,
                this node will be detached from the tree and removed from parent,
                but only if parent field type allows this (sequence or allows None).

        Raises:
            ASTNodeReplaceWithError: if replacement is not possible

        """
        if new and new.is_attached_subtree:
            raise ASTNodeReplaceWithError(
                f"Failed to replace AST Node <{self.id}> with <{new.id}> "
                "because the new node has a parent already",
                node=self,
                new_node=new,
            )

        if self.parent is not None:
            # It is a subtree

            if (
                self.parent_field is None
                or (
                    parent_field_type_info := self.parent.get_child_fields().get(
                        self.parent_field.name
                    )
                )
                is None
            ):
                raise RuntimeError(
                    f"Failed to replace AST Node <{self.id}> with "
                    f"<{new.id if new is not None else 'None'}> because "
                    "the parent node does not have a parent field or field "
                    "type info is missing. This is a bug, please report it."
                )

            # Check if the new node is of the same type as the parent expects
            _, p_type = parent_field_type_info

            if new is None:
                if not p_type.is_optional and p_type.sequence_type is None:
                    raise ASTNodeReplaceWithError(
                        f"Failed to replace AST Node <{self.id}> with <None> "
                        "because parent expects a non-optional node",
                        node=self,
                        new_node=new,
                    )
            else:
                new_type = type(new)
                if not any(issubclass(new_type, t) for t in p_type.types):
                    raise ASTNodeReplaceWithError(
                        f"Failed to replace AST Node <{self.id}> with <{new.id}> because parent "
                        f"expects nodes of type: {', '.join([t.__name__ for t in p_type.types])}",
                        node=self,
                        new_node=new,
                    )

            # remember the parent
            cur_parent = self.parent
            cur_parent_field = self.parent_field
            cur_parent_index = self.parent_index

            # Clear the parent of the old node
            self._clear_parent()

            # Detach the old subtree
            self.detach()

            if new is not None:
                if not new.detached:
                    # Pop the new node from the registry, if it was there (with it's original ID)
                    ASTNode._nodes.pop(new.id, None)

                    new_was_attached = True
                else:
                    new_was_attached = False

                # Change the ID of the new node to the old one, and store the old one in original_id
                object.__setattr__(new, "original_id", new.id)
                object.__setattr__(new, "id", self.id)

                # Make sure the new one is attached with the new ID
                try:
                    new._attach("replace")
                except Exception as e:
                    # If we failed to the attach new node, re-attach the old one
                    # and raise the exception

                    assert cur_parent_field is not None
                    self._set_parent(cur_parent, cur_parent_field, cur_parent_index)

                    self._attach("replace")

                    if new_was_attached:
                        ASTNode._nodes[new.id] = new

                    raise ASTNodeReplaceWithError(
                        f"Failed to attach the new node while replacing AST Node <{self.id}>",
                        node=self,
                        new_node=new,
                    ) from e

            # If we had a parent, we need to replace this child with new one in it
            # and re-attach the node to the parent
            cur_parent._replace_child(self, cur_parent_field, cur_parent_index, new)
        elif new is not None:
            # It is a root or detached node, so we just need to flip ids
            # and possibly attach the new one

            # Detach the old subtree
            if not self.detached:
                self.detach()

                was_attached = True
            else:
                was_attached = False

            if not new.detached:
                # Pop the new node from the registry, if it was there (with it's original ID)
                ASTNode._nodes.pop(new.id, None)

                new_was_attached = True
            else:
                new_was_attached = False

            # Change the ID of the new node to the old one, and store the old one in original_id
            object.__setattr__(new, "original_id", new.id)
            object.__setattr__(new, "id", self.id)

            # Make sure the new one is attached
            try:
                new._attach("replace")
            except Exception as e:
                # If we failed to the attach new node, re-attach the old one
                # and raise the exception
                if was_attached:
                    self._attach("replace")

                if new_was_attached:
                    ASTNode._nodes[new.id] = new

                raise ASTNodeReplaceWithError(
                    f"Failed to attach the new node while replacing AST Node <{self.id}>",
                    node=self,
                    new_node=new,
                ) from e
        else:
            # This is equivalent to detach
            self.detach()

    def duplicate(self: ASTNodeType, as_detached_clone: bool = False) -> ASTNodeType:
        """Creates a new node with the same data as this node but a unique new id."""
        logger.debug(f"Duplicating an existing AST Node <{self.id}>")

        changes: dict[str, t.Any] = {}
        for obj, f in self._iter_child_fields():
            if isinstance(obj, ASTNode):
                changes[f.name] = obj.duplicate(as_detached_clone=as_detached_clone)
            elif isinstance(obj, (list, tuple)):
                changes[f.name] = type(obj)(
                    [
                        c.duplicate(as_detached_clone=as_detached_clone)
                        if isinstance(c, ASTNode)
                        else c
                        for c in obj
                    ]
                )

        ret = replace(
            self,
            create_detached=as_detached_clone,
            original_id=None,  # Should not go to init, handled below
            id_collision_with=None,  # Should not go to init, handled below,
            **changes,
        )
        # We need to assign the original_id and copy id_collision_with
        # But not parent - this is a new node
        object.__setattr__(ret, "id_collision_with", self.id_collision_with)

        if ret.id != self.id:
            # If we duplicated a detached node, id will not change
            # so we treat this as a recreation of a node with the same ID
            object.__setattr__(ret, "original_id", self.id)
        else:
            object.__setattr__(ret, "original_id", self.original_id)

        return ret

    def calculate_xpath(self) -> bool:
        if not self.is_attached_root:
            logger.debug("Cannot calculate xpath for a non-root node")
            return False

        # Calculate the xpath of this node (root)
        xpath = f"/@root[0]{self.__class__.__name__}"

        # Set the xpath of all children
        for child in self.get_child_nodes():
            _set_xpath(child, parent_xpath=xpath)

        # Set the xpath of this node
        object.__setattr__(self, "_xpath", xpath)

        return True

    @property
    def children(self) -> t.List[ASTNode]:
        """Returns a static list with all child ASTNodes.

        Use `get_child_nodes` to iterate over

        """
        return list(self.get_child_nodes())

    @property
    def parent(self) -> ASTNode | None:
        """Returns the parent node of this node or None if it is the root node."""
        # Since we are dynamically assigning the parent fields
        # we need to use getattr to make typing happy
        parent_id = getattr(self, "_parent_id", None)
        if parent_id is None:
            return None
        else:
            return ASTNode.get_any(parent_id)

    @property
    def parent_field(self) -> Field | None:
        """Returns the field that this node is assigned to in the parent node."""
        # Since we are dynamically assigning the parent fields
        # we need to use getattr to make typing happy
        parent_field: Field | None = getattr(self, "_parent_field", None)
        if parent_field is None:
            return None
        else:
            return parent_field

    @property
    def parent_index(self) -> int | None:
        """Returns the index of this node in the parent node's container field (list or tuple)."""
        # Since we are dynamically assigning the parent fields
        # we need to use getattr to make typing happy
        parent_index: int | None = getattr(self, "_parent_index", None)
        if parent_index is None:
            return None
        else:
            return parent_index

    @property
    def xpath(self) -> str | None:
        """Returns the last calculated xpath of this node.

        To calculate the xpath, use `calculate_xpath` or root node.

        Returns None if xpath wasn't calculated.

        """
        # Since we are dynamically assigning the parent fields
        # we need to use getattr to make typing happy
        return t.cast(str | None, getattr(self, "_xpath", None))

    @property
    def detached(self) -> bool:
        """Returns True if this node is detached from the AST registry."""
        return ASTNode._nodes.get(self.id) is not self

    @property
    def is_attached_root(self) -> bool:
        """Returns True if this node is attached and the root node."""
        return self.parent is None and not self.detached

    @property
    def is_attached_subtree(self) -> bool:
        """Returns True if this node is attached and has parent (not root)."""
        return self.parent is not None and not self.detached

    def accept(self, visitor: ASTVisitor[VisitorReturnType]) -> VisitorReturnType:
        """Accepts a visitor by finding and calling a matching visitor method that should have a
        name in a form of visit_{__class__.__name__} or generic_visit if it doesn't exist.

        If the passed in visitor is a strict type, then visit method is matched by
        the exact type match. Otherise accept method walks the mro until it finds
        a visit method matching visit_{__class__.__name__}, which means it matches
        a visitor for the closest super class  of the class of this node.

        Args:
            visitor (ASTVisitor[VisitorReturnType]): The visitor to accept

        Returns:
            VisitorReturnType: The return value of the visitor's visit method

        Example:
            >>> class MyNode(ASTNode):
            ...     pass
            >>> class MyChildNode(MyNode):
            ...     pass
            >>> class MyVisitor(ASTVisitor):
            ...     def visit_MyNode(self, node: MyNode) -> str:
            ...         return "Hello World"
            ...     def generic_visit(self, node: ASTNode) -> str:
            ...         return "Hello World"
            >>> node = MyChildNode()
            >>> visitor = MyVisitor()
            >>> visitor.visit(node)
            "Hello World"

        """
        visitor_method = None

        if visitor.strict:
            visitor_method = getattr(visitor, f"visit_{self.__class__.__name__}", None)
        else:
            mro = getmro(self.__class__)
            for _class in mro[:-1]:
                visitor_method = getattr(visitor, f"visit_{_class.__name__}", None)
                if visitor_method is not None:
                    break

        if visitor_method is None:
            visitor_method = visitor.generic_visit

        return visitor_method(self)

    def get_depth(self, relative_to: ASTNode | None = None, check_ancestor: bool = True) -> int:
        """Returns the depth of this node in the tree either up to root or up to `relative_to` node
        (if it is the ancestor at all).

        Args:
            relative_to (ASTNode | None): The node to stop at. If None, the depth is calculated up to the root node.

        Returns:
            int: The depth of this node in the tree.

        Raises:
            ValueError: If `relative_to` is not an ancestor of this node.

        """
        if relative_to is not None and check_ancestor and not relative_to.is_ancestor(self):
            raise ValueError("relative_to must be an ancestor of this node")

        if self.parent is None:
            return 0
        elif relative_to is not None and self.parent == relative_to:
            return 1
        else:
            return self.parent.get_depth(relative_to=relative_to, check_ancestor=False) + 1

    def ancestors(self) -> t.Iterator[ASTNode]:
        """Iterates over all ancestors of this node."""
        parent = self.parent
        while parent is not None:
            yield parent
            parent = parent.parent

    def get_first_ancestor_of_type(
        self,
        ancestor_class: t.Type[ASTNodeType] | tuple[t.Type[ASTNodeType], ...],
        *,
        exact_type: bool = False,
    ) -> ASTNodeType | None:
        """Returns the first ancestor of this node that is an instance of `ancestor_class`. Or None
        if no such ancestor exists.

        Args:
            ancestor_class (t.Type[ASTNodeType] | tuple[t.Type[ASTNodeType], ...]): The
                ancestor class or tuple of classes to search for.
            exact_type (bool, optional): Whether to search for exact type match,
                or match any subclasses (isintance check). Defaults to False.

        Returns:
            ASTNodeType | None: The ancestor node or None if no such ancestor exists.

        """
        if not isinstance(ancestor_class, tuple):
            ancestor_classes = t.cast(tuple[t.Type[ASTNodeType], ...], (ancestor_class,))
        else:
            ancestor_classes = ancestor_class

        for ancestor in self.ancestors():
            if exact_type and type(ancestor) in ancestor_classes:
                return t.cast(ASTNodeType, ancestor)

            if not exact_type and isinstance(ancestor, ancestor_classes):
                return t.cast(ASTNodeType, ancestor)

        return None

    def is_ancestor(self, node: ASTNode) -> bool:
        """Returns True if this node is an ancestor of `node`."""
        if node.parent is None:
            return False
        elif node.parent == self:
            return True
        else:
            return self.is_ancestor(node.parent)

    def is_equal(self, other: t.Any) -> bool:
        """Returns True if this node is equal to `other`.

        Unlike `==`, this method only compares properties & children and ignores the origin, id,
        parent, etc.

        """
        if not isinstance(other, type(self)):
            return False

        return self.content_id == other.content_id

    def to_properties_dict(self) -> dict[str, t.Any]:
        """Returns a dictionary of all node properties (not children), including only "content",
        i.e. origin, id, origin_id, id_collision, parent_id and hidden attributes are not
        included."""
        d = {}

        for v, f in self.get_properties():
            d[f.name] = v

        return d

    def _ensure_iterable(self, value: t.Any | None) -> t.Iterable[t.Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return value
        else:
            # Dicts, sets are not are not traversed
            return [value]

    def _is_field_child(self, field: Field) -> bool:
        # fields that are not part of init can't be children
        if not field.init:
            return False

        o = getattr(self, field.name)

        if o is None or (isinstance(o, (list, tuple)) and len(o) == 0):
            return field.name in self.get_child_fields()

        if isinstance(o, ASTNode):
            return True
        else:
            for item in self._ensure_iterable(o):
                if isinstance(item, ASTNode):
                    return True

        return False

    def _iter_child_fields(
        self,
    ) -> t.Iterable[tuple[ASTNode | t.Sequence[ASTNode] | None, Field]]:
        for f in fields(self):
            # Skip non-child fields
            if not self._is_field_child(f):
                continue

            yield getattr(self, f.name), f

    def dfs(
        self,
        prune: t.Callable[[ASTNode], bool] | None = None,
        filter: t.Callable[[ASTNode], bool] | None = None,
        bottom_up: bool = False,
        skip_self: bool = False,
    ) -> Generator[ASTNode, None, None]:
        """Returns a generator object which visits all nodes in this tree in the DFS (Depth-first)
        order.

        Args:
            prune (t.Callable[[ASTNode], bool] | None, optional): An optional function which if it returns True will prevent further decent into the children of this element.
            filter (t.Callable[[ASTNode], bool] | None, optional): An optional function which if it returns False will prevent the element from being yielded, but won't interrupt the recursive decent/ascent.
            bottom_up (bool, optional): Enables bottom up traversal. Defaults to False.
            skip_self (bool, optional): Doesn't yield self. Defaults to False.

        Yields:
            Generator[ASTNode, None, None]: A generator object which visits all nodes in this tree in the DFS (Depth-first) order.

        """
        build_queue: t.Deque[ASTNode] = deque([self])
        yield_queue: t.Deque[ASTNode] = deque()

        while build_queue:
            child = build_queue.popleft()

            if not skip_self:
                if filter is None or filter(child):
                    if bottom_up:
                        yield_queue.appendleft(child)
                    else:
                        yield_queue.append(child)

                if prune and prune(child):
                    continue
            else:
                skip_self = False

            # Walk through children
            if bottom_up:
                for c in child.get_child_nodes():
                    build_queue.appendleft(c)
            else:
                for c in reversed(child.children):
                    build_queue.appendleft(c)

        while yield_queue:
            yield yield_queue.popleft()

    def bfs(
        self,
        prune: t.Callable[[ASTNode], bool] | None = None,
        filter: t.Callable[[ASTNode], bool] | None = None,
        skip_self: bool = False,
    ) -> Generator[ASTNode, None, None]:
        """Returns a generator object which visits all nodes in this tree in the BFS (Breadth-first)
        order.

        Args:
            prune (t.Callable[[ASTNode], bool]): An optional function which if it returns True will prevent further decent into the children of this element.
            filter (t.Callable[[ASTNode], bool]): An optional function which if it returns False will prevent the element from being yielded, but won't interrupt the recursive decent.

        Returns:
            the generator object.

        """
        queue: t.Deque[ASTNode] = deque([self])

        while queue:
            child = queue.popleft()

            if not skip_self:
                if filter is None or filter(child):
                    yield child

                if prune and prune(child):
                    continue
            else:
                skip_self = False

            # Walk through children
            queue.extend(child.get_child_nodes())

    def gather(
        self,
        obj_class: t.Type[ASTNodeType] | tuple[t.Type[ASTNodeType], ...],
        *,
        exact_type: bool = False,
        extra_filter: t.Callable[[ASTNode], bool] | None = None,
        prune: t.Callable[[ASTNode], bool] | None = None,
        skip_self: bool = False,
    ) -> Generator[ASTNodeType, None, None]:
        """Shorthand for traversing the tree and gathering all instances of subclasses of
        `obj_class` or exactly `obj_class` if `exact_type` is True.

        Args:
            obj_class (t.Type[ASTNodeType] | tuple[t.Type[ASTNodeType], ...]): any ASTNode subclass or Tuple of classes to gather.
            exact_type (bool, optional): Whether to only gather instances of `obj_class` and not its subclasses. Defaults to False.
            extra_filter (t.Callable[[ASTNode], bool] | None, optional): An optional additional filter to apply when gathering. Defaults to None.
            prune (t.Callable[[ASTNode], bool] | None, optional): Optional function to stop traversal. Defaults to None.
            skip_self (bool, optional): Whether to skip the node that this method is called from. Defaults to False.

        Yields:
            Generator[ASTNodeType, None, None]: An iterator of `obj_class` instances.

        """
        obj_classes: tuple[t.Type[ASTNodeType], ...]
        if not isinstance(obj_class, tuple):
            obj_classes = (obj_class,)
        else:
            obj_classes = obj_class

        if not exact_type:

            def filter_fn(obj: ASTNode) -> bool:
                return isinstance(obj, obj_classes) and (extra_filter is None or extra_filter(obj))

        else:

            def filter_fn(obj: ASTNode) -> bool:
                return type(obj) in obj_classes and (extra_filter is None or extra_filter(obj))

        for elem in self.dfs(prune=prune, filter=filter_fn, bottom_up=False, skip_self=skip_self):
            yield t.cast(ASTNodeType, elem)

    @classmethod
    def get_property_fields(
        cls,
        skip_id: bool = True,
        skip_origin: bool = True,
        skip_original_id: bool = True,
        skip_id_collision_with: bool = True,
        skip_hidden: bool = True,
        skip_non_compare: bool = False,
    ) -> t.Iterable[tuple[str, Field]]:
        """Returns an iterator of all properties (but not child attributes) of this node using
        static type information.

        Args:
            skip_id (bool, optional): Whether to skip the id property. Defaults to True.
            skip_origin (bool, optional): Whether to skip the origin property. Defaults to True.
            skip_original_id (bool, optional): Whether to skip the original_id property. Defaults to True.
            skip_id_collision_with (bool, optional): Whether to skip the id_collision_with property. Defaults to True.
            skip_hidden (bool, optional): Whether to skip properties starting with an underscore. Defaults to True.
            skip_non_compare (bool, optional): Whether to skip properties that are not used in comparison (field.comapre is False). Defaults to False.

        Yields:
            t.Iterable[tuple[str, Field]]: An iterator of tuples of (field name, field).

        """
        if cls._props is None:
            cls._props = {f.name: (f, type_) for f, type_ in get_ast_node_properties(cls).items()}

        for name, (f, _) in cls._props.items():
            if _is_skip_field(
                field=f,
                skip_id=skip_id,
                skip_origin=skip_origin,
                skip_original_id=skip_original_id,
                skip_id_collision_with=skip_id_collision_with,
                skip_hidden=skip_hidden,
                skip_non_compare=skip_non_compare,
                skip_non_init=False,
            ):
                continue

            yield name, f

    @classmethod
    def get_child_fields(
        cls,
    ) -> t.Mapping[str, tuple[Field, ChildFieldTypeInfo]]:
        """Returns an iterator of all child attributes of this node using static type information.

        Returns:
            t.Mapping[str, tuple[Field, ChildFieldTypeInfo]]:
                A mapping of child attribute name to (field, type_info).

        """
        if cls._child_fields is None:
            cls._child_fields = {
                f.name: (f, type_info) for f, type_info in get_ast_node_child_fields(cls).items()
            }

        return cls._child_fields

    def get_properties(
        self,
        skip_id: bool = True,
        skip_origin: bool = True,
        skip_original_id: bool = True,
        skip_id_collision_with: bool = True,
        skip_hidden: bool = True,
        skip_non_compare: bool = False,
    ) -> t.Iterable[tuple[t.Any, Field]]:
        """Returns an iterator of all properties (but not child attributes) of this node.

        Args:
            skip_id (bool, optional): Whether to skip the id property. Defaults to True.
            skip_origin (bool, optional): Whether to skip the origin property. Defaults to True.
            skip_original_id (bool, optional): Whether to skip the original_id property. Defaults to True.
            skip_id_collision_with (bool, optional): Whether to skip the id_collision_with property. Defaults to True.
            skip_hidden (bool, optional): Whether to skip properties starting with an underscore. Defaults to True.
            skip_non_compare (bool, optional): Whether to skip properties that are not used in comparison (field.comapre is False). Defaults to False.

        Yields:
            t.Iterable[tuple[t.Any, Field]]: An iterator of tuples of (value, field).

        """
        for f in fields(self):
            if _is_skip_field(
                field=f,
                skip_id=skip_id,
                skip_origin=skip_origin,
                skip_original_id=skip_original_id,
                skip_id_collision_with=skip_id_collision_with,
                skip_hidden=skip_hidden,
                skip_non_compare=skip_non_compare,
                skip_non_init=False,
            ):
                continue

            # Always skip children
            if self._is_field_child(f):
                continue

            yield getattr(self, f.name), f

    def get_child_nodes(self) -> t.Iterable[ASTNode]:
        for f in fields(self):
            # Skip non-child fields
            if not self._is_field_child(f):
                continue

            objects = self._ensure_iterable(getattr(self, f.name))
            for o in objects:
                assert isinstance(o, ASTNode)
                yield o

    def get_child_nodes_with_field(
        self,
    ) -> t.Iterable[tuple[ASTNode, Field, int | None]]:
        """Returns a generator object which yields all child nodes with their corresponding field
        and index (for lists and tuples)."""
        for f in fields(self):
            # Skip non-child fields
            if not self._is_field_child(f):
                continue

            objects = getattr(self, f.name)
            if isinstance(objects, (list, tuple)):
                for i, o in enumerate(objects):
                    assert isinstance(o, ASTNode)
                    yield o, f, i
            elif objects is not None:
                assert isinstance(objects, ASTNode)
                yield objects, f, None

    def __rich__(self, parent: Tree | None = None) -> Tree:
        """Returns a tree widget for the 'rich' library."""
        return self._rich(parent)

    def _rich(self, parent: Tree | None, field: Field | None = None) -> Tree:
        name = f":deciduous_tree:[bold green]root({self.__class__.__name__})[/bold green]"
        if field:
            name = (
                f":deciduous_tree:[bold green]{field.name}({self.__class__.__name__})[/bold green]"
            )

        if parent:
            tree = parent.add(name)
        else:
            tree = Tree(name)

        if self.origin is not None:
            tree.add(f":round_pushpin: @{escape(str(self.origin))}")

        for p, f in self.get_properties(skip_id=False):
            tree.add(f":spiral_notepad: [yellow]{f.name}[/]={escape(str(p))}")

        for f in fields(self):
            # Skip non-child fields
            if not self._is_field_child(f):
                continue

            child = getattr(self, f.name)
            if isinstance(child, Iterable):
                if not child:
                    tree.add(f":file_folder:[yellow]{f.name}[/]={escape('()')}")
                    continue

                subtree = tree.add(f":file_folder:[yellow]{f.name}[/]")
                for c in child:
                    if isinstance(c, ASTNode):
                        c._rich(subtree, f)
            elif child is None:
                tree.add(f":file_folder:[yellow]{f.name}[/]={escape(str(None))}")
            else:
                child._rich(tree, f)

        return tree

    __hash__ = None  # type: ignore # Make sure even frozen dataclasses will not be hashable

    def __init_subclass__(cls) -> None:
        """Register subclasses and enforce hash function by object ID to all ASTNode subclasses.

        Dataclasses by default will remove hashing by ID, but we want it back for ASTNodes.

        """
        cls.__hash__ = None  # type: ignore # Make sure even frozen dataclasses will not be hashable

        # Make sure each class uses it's own list of child fields & properties
        # We do not set them until first access due forwared references
        # that may not be resolved yet.
        cls._props = None
        cls._child_fields = None

        return super().__init_subclass__()


ASTNodeType = t.TypeVar("ASTNodeType", bound=ASTNode)


class ASTVisitor(t.Generic[VisitorReturnType]):
    """A visitor generic base class for an AST visitor.

    Args:
        t (_type_): Vistior return type

    """

    strict: bool = False
    """Strict visitors match visit methods to nodes by exact type.

    Non-strict visitors will match by isinstance check in MRO order.

    """

    def generic_visit(self, node: ASTNode) -> VisitorReturnType:
        raise NotImplementedError

    def visit(self, node: ASTNode) -> VisitorReturnType:
        """Visits the given node and returns the result."""
        return node.accept(self)

    def __init_subclass__(cls, *, validate: bool = False) -> None:
        """Iterate over new visitor methods and check that names match the node type annotation."""
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

    Note:
        Transformation creates a full copy of the original tree in memory
        if it was an attached tree (and it normally will be).
        Visitor methods operate on a copy, rather than the original nodes.
        The copies are detached, meaning that the visitor method will get nodes
        that do not have assigned parents and thus walking up the tree is not
        possible.

        If transformation didn't raise an exception, the original tree is
        replaced with the transformed tree using ASTNode.replace_with().
        This means that the original tree object becomes fully detached.

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

    def _transform_children(
        self, node: ASTNode
    ) -> t.Mapping[str, ASTNode | None | list[ASTNode] | tuple[ASTNode, ...]]:
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

                new_child = self.transform(child)

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

        # Remove unchanged fields
        unchanged_fields = set(changes.keys()) - field_names_with_changes

        for fname in unchanged_fields:
            changes.pop(fname)

        # Ensure the correct sequence type for changed sequence fields
        for fname in field_names_with_changes:
            _, type_info = t.cast(
                tuple[Field, ChildFieldTypeInfo], node.get_child_fields().get(fname)
            )

            if type_info.sequence_type is not None:
                changes[fname] = type_info.sequence_type(t.cast(list[ASTNode], changes[fname]))

        # Return the changes
        return changes

    def generic_visit(self, node: ASTNode) -> ASTNode | None:
        """Transforms children of the given node and returns a new node with the changes."""

        changes = self._transform_children(node)

        # No changes, return the original node
        if not changes:
            return node

        # Return a new node with the changes
        return node.replace(**changes)

    def visit(self, node: ASTNode) -> ASTNode | None:
        """Overrides the default visit method to ensure that the node is detached before
        transformation.

        Args:
            node (ASTNode): The node to transform

        Returns:
            ASTNode | None: The transformed node or None if the node was
            removed.

        """
        return self.transform(node)

    def transform(self, node: ASTNode) -> ASTNode | None:
        orig_node: ASTNode | None = None

        # If we are transforming an attached tree or subtree
        # we create a fully detached clone, transform it
        # and then replace the original node with the transformed one.
        # but only if transformation was successful.
        if not node.detached:
            orig_node = node
            node = node.duplicate(as_detached_clone=True)

        transformed: ASTNode | None = None
        try:
            transformed = super().visit(node)

            if orig_node is not None:
                orig_node.replace_with(transformed)

            return transformed
        except Exception as e:
            raise ASTTransformError(orig_node=node, transformed_node=transformed) from e


class ASTTransformer(ABC):
    """A transformer base class for AST nodes."""

    def prune(self, node: ASTNode) -> bool:
        """A function used to prune the tree during transformation.

        Must returns whether the given node should prevent further traversal (see `ASTNode.dfs`).

        Default implementation returns False, meaning that the tree will be fully traversed.

        Args:
            node (ASTNode): The current node being transformed.

        """
        return False

    def filter(self, node: ASTNode) -> bool:
        """A function used to filter the tree during transformation.

        Must returns whether the given node should be transformed (see `ASTNode.dfs`).

        Default implementation returns True, meaning that all nodes will be transformed.

        Args:
            node (ASTNode): The current node being considered for transformation.

        """
        return True

    @abstractmethod
    def transform(self, node: ASTNode) -> ASTNode | None:
        """The main function implementing the transformation logic.

        Args:
            node (ASTNode): The node to transform.

        Returns:
            ASTNode: A new transformed node or the original node.

        """
        raise NotImplementedError

    def execute(self, node: ASTNode) -> ASTNode | None:
        """Executes the `transform` function defined on this class against all nodes in subtree
        rooted in `node`, applying filter and pruning function as defined on this class.

        Bottom up traversal is used, so that the children of a node are transformed before the node itself.

        If transform function returns a new node, the original node is replaced with the new node.

        Args:
            node (ASTNode): The node to transform.

        Raises:
            ASTTransformError: If the original node cannot be replaced with the transformed node.

        Returns:
            ASTNode | None: The transformed node.

        """
        for child in node.dfs(bottom_up=True, filter=self.filter, prune=self.prune):
            new_node = self.transform(child)

            if child is node:
                # We hit the root node, return the transformed node
                return new_node

            if new_node is not child:
                # Means that the node was transformed
                if (
                    new_node is None
                    or new_node.id
                    != child.id  # means the new node wasn't created using ASTNode.replace() which would retain the ID
                ):
                    # try to replace the node
                    try:
                        child.replace_with(new_node)
                    except Exception as e:
                        raise ASTTransformError(orig_node=child, transformed_node=new_node) from e

        # Means that this node was pruned
        return node
