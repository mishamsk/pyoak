from __future__ import annotations

import enum
import hashlib
import logging
import weakref
from collections import deque
from collections.abc import Generator, Iterable
from dataclasses import InitVar, dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Deque,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    cast,
)

from rich.markup import escape
from rich.tree import Tree

from . import config
from ._codegen import (
    gen_and_yield_get_child_nodes,
    gen_and_yield_get_child_nodes_with_field,
    gen_and_yield_get_properties,
    gen_and_yield_iter_child_fields,
)
from ._helpers import is_skip_field
from ._methods import repr_fn
from .error import (
    ASTNodeDuplicateChildrenError,
    ASTNodeIDCollisionError,
    ASTNodeParentCollisionError,
    ASTNodeRegistryCollisionError,
    ASTNodeReplaceError,
    ASTNodeReplaceWithError,
    InvalidTypes,
)
from .origin import Origin
from .serialize import TYPE_KEY, DataClassSerializeMixin
from .types import (
    get_cls_all_fields,
    get_cls_allowed_rename_field_names,
    get_cls_child_fields,
    get_cls_props,
)
from .typing import FieldTypeInfo, check_annotations, check_runtime_types, get_type_args_as_tuple

if TYPE_CHECKING:
    from dataclasses import Field as DataClassField

    from .match.xpath import ASTXpath

    # to make mypy happy
    Field = DataClassField[Any]

logger = logging.getLogger(__name__)

_ASTNodeType = TypeVar("_ASTNodeType", bound="ASTNode")


class _UnsetClass(str):
    pass


_UNSET = _UnsetClass()


def _get_next_unique_id(id_: str) -> str:
    """Return a unique ID."""
    i = 1
    original_id = id_
    while ASTNode._nodes.get(id_) is not None:
        id_ = f"{original_id}_{i}"
        i += 1

    return id_


AST_SERIALIZE_DIALECT_KEY = "ast_serialize_dialect"


class ASTSerializationDialects(enum.Enum):
    AST_EXPLORER = enum.auto()
    AST_TEST = enum.auto()


# dataclasses prior to py3.11 didn't support __weakref__ slot
# so in order to make a universally supported base class
# we need to add it manually via a mixin
class _NodeSlots:
    __slots__ = (
        "__weakref__",
        "_parent_id",
        "_parent_field",
        "_parent_index",
    )


@dataclass(slots=True)
class ASTNode(DataClassSerializeMixin, _NodeSlots):
    """A base class for all AST Node classes.

    Provides the following functions:
        - Maintains a weakref dictionary of all nodes, accessible through the class methods `get` and `get_any`
        - Maintains a parent-child relationship between nodes
        - Methods for replacing attributes or whole nodes
        - Various tree walking methods
        - Serialization & deserialization
        - rich console API support (for printing)

    Notes:
        - Subclasses must be dataclasses
        - Subclasses may be slotted, but remember that in multiple inheritance, only one base can have non-empty slots
        - Fields typed as union of subclasses of ASTNode or None as well as tuples of ASTNode subclasses
            are considered children. All other types that have ASTNode subclasses in signature
            will trigger an error.

    Raises:
        ASTNodeError: If there are children with a different parent alredy assigned (every AST node can only have one parent)

    """

    _nodes: ClassVar[weakref.WeakValueDictionary[str, ASTNode]] = weakref.WeakValueDictionary()
    """Registry of all node objects that are considered "attached"."""

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

    id: str = field(default=_UNSET, kw_only=True, compare=False)
    """The unique ID of this node. If not set in the constructor, it will be auto-generated based on
    the node's properties and origin.

    This means that two nodes with the same "content" (i.e. properties and childre)
    but generated from different sources will have different IDs.

    For a stable ID based purely on the node's content, use the `content_id` property.

    """

    content_id: str = field(
        default=_UNSET,
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

        if config.RUNTIME_TYPE_CHECK:
            incorrect_fields = check_runtime_types(
                self,
                {
                    f: finfo
                    for f, finfo in get_cls_all_fields(self.__class__).items()
                    if f.name not in ("id", "content_id")
                },
            )

            if incorrect_fields:
                raise InvalidTypes(incorrect_fields)

        # Calculate ID
        new_id: str
        id_collision_with: str | None = None
        original_id: str | None = None
        if self.id is _UNSET:
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

    def _set_parent(self, parent: ASTNode, field: Field, index: int | None) -> None:
        object.__setattr__(self, "_parent_id", parent.id)
        object.__setattr__(self, "_parent_field", field)
        object.__setattr__(self, "_parent_index", index)

    def __post_serialize__(self, d: dict[str, Any]) -> dict[str, Any]:
        # Run first, otherwise _children will be dropped from the output
        out = super(ASTNode, self).__post_serialize__(d)

        if (
            self._get_serialization_options().get(AST_SERIALIZE_DIALECT_KEY)
            == ASTSerializationDialects.AST_EXPLORER
        ):
            out["_children"] = []
            out["_children"].extend([f.name for f in get_cls_child_fields(self.__class__)])

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
        self, operation: Literal["create", "attach", "replace"]
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

    def _attach(self, operation: Literal["create", "attach", "replace"]) -> None:
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
                for c in cast(Iterable[ASTNode], orig_seq[index + 1 :]):
                    c._set_parent(self, field, cast(int, c.parent_index) - 1)
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
        cls: type[_ASTNodeType],
        id: str,
        default: _ASTNodeType | None = None,
        strict: bool = True,
    ) -> _ASTNodeType | None:
        """Gets a node of this class type from the AST registry.

        Args:
            id (str): The id of the node to get
            default (ASTNodeType | None, optional): The default value to return if the node is not found.
                Defaults to None.
            strict (bool, optional): If True, only a node of this class type is returned.
                Otherwise an instnace if any subclass is allowed. Defaults to True.

        Returns:
            ASTNodeType | None: The node if found, otherwise the default value

        """
        ret = ASTNode._nodes.get(id, default)
        if ret is None:
            return None
        elif strict and not type(ret) == cls:
            return None
        elif not strict and not isinstance(ret, cls):
            return None
        else:
            return cast(_ASTNodeType, ret)

    @classmethod
    def get_any(cls, id: str, default: ASTNode | None = None) -> ASTNode | None:
        """Gets a node of any type from the AST registry by id.

        Args:
            id (str): The id of the node to get
            default (ASTNode | None, optional): The default value to return if the node is not found.
                Defaults to None.

        Returns:
            ASTNode | None: The node if found, otherwise the default value

        """
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
            if config.TRACE_LOGGING:
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
            if config.TRACE_LOGGING:
                logger.debug(f"Tried to detach a detached AST Node <{self.id}> from AST registry")
            return True

        # Checking for parent, not for _parent (which is id) to see if it has a registered parent
        if not self.is_attached_root:
            if config.TRACE_LOGGING:
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

    def replace(self: _ASTNodeType, **changes: Any) -> _ASTNodeType:
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
        fields_allowed_for_replace = get_cls_allowed_rename_field_names(self.__class__)

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
                or (parent_field_type_info := self.parent.get_child_fields().get(self.parent_field))
                is None
            ):
                raise RuntimeError(
                    f"Failed to replace AST Node <{self.id}> with "
                    f"<{new.id if new is not None else 'None'}> because "
                    "the parent node does not have a parent field or field "
                    "type info is missing. This is a bug, please report it."
                )

            # Check if the new node is of the same type as the parent expects
            if new is None:
                if (
                    not parent_field_type_info.is_optional
                    and not parent_field_type_info.is_collection
                ):
                    raise ASTNodeReplaceWithError(
                        f"Failed to replace AST Node <{self.id}> with <None> "
                        "because parent expects a non-optional node",
                        node=self,
                        new_node=new,
                    )
            else:
                new_type = type(new)
                p_field_types = get_type_args_as_tuple(
                    parent_field_type_info.resolved_type, ASTNode
                )
                if not any(issubclass(new_type, t) for t in p_field_types):
                    raise ASTNodeReplaceWithError(
                        f"Failed to replace AST Node <{self.id}> with <{new.id}> because parent "
                        f"expects nodes of type: {', '.join([t.__name__ for t in p_field_types])}",
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

    def duplicate(self: _ASTNodeType, as_detached_clone: bool = False) -> _ASTNodeType:
        """Creates a new node with the same data as this node but a unique new id."""
        logger.debug(f"Duplicating an existing AST Node <{self.id}>")

        changes: dict[str, Any] = {}
        for obj, f in self.iter_child_fields():
            if isinstance(obj, ASTNode):
                changes[f.name] = obj.duplicate(as_detached_clone=as_detached_clone)
            elif isinstance(obj, tuple):
                changes[f.name] = tuple(
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

    @property
    def children(self) -> Sequence[ASTNode]:
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

    def ancestors(self) -> Iterator[ASTNode]:
        """Iterates over all ancestors of this node."""
        parent = self.parent
        while parent is not None:
            yield parent
            parent = parent.parent

    def get_first_ancestor_of_type(
        self,
        ancestor_class: type[_ASTNodeType] | tuple[type[_ASTNodeType], ...],
        *,
        exact_type: bool = False,
    ) -> _ASTNodeType | None:
        """Returns the first ancestor of this node that is an instance of `ancestor_class`. Or None
        if no such ancestor exists.

        Args:
            ancestor_class (type[ASTNodeType] | tuple[type[ASTNodeType], ...]): The
                ancestor class or tuple of classes to search for.
            exact_type (bool, optional): Whether to search for exact type match,
                or match any subclasses (isintance check). Defaults to False.

        Returns:
            ASTNodeType | None: The ancestor node or None if no such ancestor exists.

        """
        if not isinstance(ancestor_class, tuple):
            ancestor_classes = cast(tuple[type[_ASTNodeType], ...], (ancestor_class,))
        else:
            ancestor_classes = ancestor_class

        for ancestor in self.ancestors():
            if exact_type and type(ancestor) in ancestor_classes:
                return cast(_ASTNodeType, ancestor)

            if not exact_type and isinstance(ancestor, ancestor_classes):
                return ancestor

        return None

    def is_ancestor(self, node: ASTNode) -> bool:
        """Returns True if this node is an ancestor of `node`."""
        if node.parent is None:
            return False
        elif node.parent == self:
            return True
        else:
            return self.is_ancestor(node.parent)

    def is_equal(self, other: Any) -> bool:
        """Returns True if this node is equal to `other`.

        Unlike `==`, this method only compares properties & children and ignores the origin, id,
        parent, etc.

        """
        if not isinstance(other, type(self)):
            return False

        return self.content_id == other.content_id

    def to_properties_dict(self) -> dict[str, Any]:
        """Returns a dictionary of all node properties (not children), including only "content",
        i.e. origin, id, origin_id, id_collision, parent_id and hidden attributes are not
        included."""
        d = {}

        for v, f in self.get_properties():
            d[f.name] = v

        return d

    def _ensure_iterable(self, value: Any | None) -> Iterable[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return value
        else:
            # Dicts, sets are not are not traversed
            return [value]

    def iter_child_fields(
        self, *, sort_keys: bool = False
    ) -> Iterable[tuple[ASTNode | tuple[ASTNode] | None, Field]]:
        """Iterates over all child fields of this node, returning the child field value as-is
        (whether it is None, a sequence or a child node) and the field itself.

        Args:
            sort_keys (bool, optional): Whether to yield children by sorted field name.
                Defaults to False, meaning children will be yielded in the order they are defined.

        Returns:
            Iterable[tuple[ASTNode | tuple[ASTNode] | None, Field]]:
                An iterator of tuples of (child field value, field).

        """

        # Dynamicaly generate a specialized function for this class
        yield from gen_and_yield_iter_child_fields(self, sort_keys=sort_keys)

    def dfs(
        self,
        prune: Callable[[ASTNode], bool] | None = None,
        filter: Callable[[ASTNode], bool] | None = None,
        bottom_up: bool = False,
        skip_self: bool = False,
    ) -> Generator[ASTNode, None, None]:
        """Returns a generator object which visits all nodes in this tree in the DFS (Depth-first)
        order.

        Args:
            prune (Callable[[ASTNode], bool] | None, optional): An optional function which if it returns True will prevent further decent into the children of this element.
            filter (Callable[[ASTNode], bool] | None, optional): An optional function which if it returns False will prevent the element from being yielded, but won't interrupt the recursive decent/ascent.
            bottom_up (bool, optional): Enables bottom up traversal. Defaults to False.
            skip_self (bool, optional): Doesn't yield self. Defaults to False.

        Yields:
            Generator[ASTNode, None, None]: A generator object which visits all nodes in this tree in the DFS (Depth-first) order.

        """
        build_queue: Deque[ASTNode] = deque([self])
        yield_queue: Deque[ASTNode] = deque()

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
        prune: Callable[[ASTNode], bool] | None = None,
        filter: Callable[[ASTNode], bool] | None = None,
        skip_self: bool = False,
    ) -> Generator[ASTNode, None, None]:
        """Returns a generator object which visits all nodes in this tree in the BFS (Breadth-first)
        order.

        Args:
            prune (Callable[[ASTNode], bool]): An optional function which if it returns True will prevent further decent into the children of this element.
            filter (Callable[[ASTNode], bool]): An optional function which if it returns False will prevent the element from being yielded, but won't interrupt the recursive decent.

        Returns:
            the generator object.

        """
        queue: Deque[ASTNode] = deque([self])

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
        obj_class: type[_ASTNodeType] | tuple[type[_ASTNodeType], ...],
        *,
        exact_type: bool = False,
        extra_filter: Callable[[ASTNode], bool] | None = None,
        prune: Callable[[ASTNode], bool] | None = None,
        skip_self: bool = False,
    ) -> Generator[_ASTNodeType, None, None]:
        """Shorthand for traversing the tree and gathering all instances of subclasses of
        `obj_class` or exactly `obj_class` if `exact_type` is True.

        Args:
            obj_class (type[ASTNodeType] | tuple[type[ASTNodeType], ...]): any ASTNode subclass or Tuple of classes to gather.
            exact_type (bool, optional): Whether to only gather instances of `obj_class` and not its subclasses. Defaults to False.
            extra_filter (Callable[[ASTNode], bool] | None, optional): An optional additional filter to apply when gathering. Defaults to None.
            prune (Callable[[ASTNode], bool] | None, optional): Optional function to stop traversal. Defaults to None.
            skip_self (bool, optional): Whether to skip the node that this method is called from. Defaults to False.

        Yields:
            Generator[ASTNodeType, None, None]: An iterator of `obj_class` instances.

        """
        obj_classes: tuple[type[_ASTNodeType], ...]
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
            yield cast(_ASTNodeType, elem)

    def find(self, xpath: str | ASTXpath) -> ASTNode | None:
        """Finds a node by xpath.

        Args:
            xpath (str | ASTXpath): The xpath to find.

        Returns:
            ASTNode | None: The node if found, otherwise None.

        Raises:
            ASTXpathDefinitionError: If the xpath is invalid.

        """
        from .match.xpath import ASTXpath

        if isinstance(xpath, str):
            xpath = ASTXpath(xpath)

        try:
            return next(xpath.findall(self))
        except StopIteration:
            return None

    def findall(self, xpath: str | ASTXpath) -> Generator[ASTNode, None, None]:
        """Finds all nodes by xpath.

        Args:
            xpath (str | ASTXpath): The xpath to find.

        Returns:
            Generator[ASTNode, None, None]: An iterator of nodes.

        Raises:
            ASTXpathDefinitionError: If the xpath is invalid.

        """
        from .match.xpath import ASTXpath

        if isinstance(xpath, str):
            xpath = ASTXpath(xpath)

        yield from xpath.findall(self)

    @classmethod
    def get_property_fields(
        cls,
        skip_id: bool = True,
        skip_origin: bool = True,
        skip_original_id: bool = True,
        skip_id_collision_with: bool = True,
        skip_hidden: bool = True,
        skip_non_compare: bool = False,
        skip_non_init: bool = False,
    ) -> Iterable[Field]:
        """Returns an iterator of all properties (but not child attributes) of this node using
        static type information.

        Args:
            skip_id (bool, optional): Whether to skip the id property. Defaults to True.
            skip_origin (bool, optional): Whether to skip the origin property. Defaults to True.
            skip_original_id (bool, optional): Whether to skip the original_id property. Defaults to True.
            skip_id_collision_with (bool, optional): Whether to skip the id_collision_with property. Defaults to True.
            skip_hidden (bool, optional): Whether to skip properties starting with an underscore. Defaults to True.
            skip_non_compare (bool, optional): Whether to skip properties that are not used in comparison (field.comapre is False). Defaults to False.
            skip_non_init (bool, optional): Whether to skip properties that are not initialized. Defaults to False.

        Yields:
            Iterable[tuple[str, Field]]: An iterator of tuples of (field name, field).

        """
        for f in get_cls_props(cls):
            if is_skip_field(
                field=f,
                skip_id=skip_id,
                skip_origin=skip_origin,
                skip_original_id=skip_original_id,
                skip_id_collision_with=skip_id_collision_with,
                skip_hidden=skip_hidden,
                skip_non_compare=skip_non_compare,
                skip_non_init=skip_non_init,
            ):
                continue

            yield f

    @classmethod
    def get_child_fields(
        cls,
    ) -> Mapping[Field, FieldTypeInfo]:
        """Returns an iterator of all child attributes of this node using static type information.

        Returns:
            Mapping[Field, ChildFieldTypeInfo]:
                A mapping of child attribute name to (field, type_info).

        """
        return get_cls_child_fields(cls)

    def get_properties(
        self,
        skip_id: bool = True,
        skip_origin: bool = True,
        skip_original_id: bool = True,
        skip_id_collision_with: bool = True,
        skip_hidden: bool = True,
        skip_non_compare: bool = False,
        skip_non_init: bool = False,
        *,
        sort_keys: bool = False,
    ) -> Iterable[tuple[Any, Field]]:
        """Returns an iterator of all properties (but not child attributes) of this node.

        Args:
            skip_id (bool, optional): Whether to skip the id property. Defaults to True.
            skip_origin (bool, optional): Whether to skip the origin property. Defaults to True.
            skip_original_id (bool, optional): Whether to skip the original_id property. Defaults to True.
            skip_id_collision_with (bool, optional): Whether to skip the id_collision_with property. Defaults to True.
            skip_hidden (bool, optional): Whether to skip properties starting with an underscore. Defaults to True.
            skip_non_compare (bool, optional): Whether to skip properties that are not used in comparison (field.comapre is False). Defaults to False.
            skip_non_init (bool, optional): Whether to skip properties that are not initialized. Defaults to False.
            sort_keys (bool, optional): Whether to yield properties by sorted field name.

        Yields:
            Iterable[tuple[Any, Field]]: An iterator of tuples of (value, field).

        """
        # Dynamicaly generate a specialized function for this class
        yield from gen_and_yield_get_properties(
            self,
            skip_id,
            skip_origin,
            skip_original_id,
            skip_id_collision_with,
            skip_hidden,
            skip_non_compare,
            skip_non_init,
            sort_keys=sort_keys,
        )

    def get_child_nodes(self, *, sort_keys: bool = False) -> Iterable[ASTNode]:
        """Returns a generator object which yields all child nodes.

        Args:
            sort_keys (bool, optional): Whether to yield children by sorted field name.
                Defaults to False, meaning children will be yielded in the order they are defined.

        """
        # Dynamicaly generate a specialized function for this class
        yield from gen_and_yield_get_child_nodes(self, sort_keys=sort_keys)

    def get_child_nodes_with_field(
        self, *, sort_keys: bool = False
    ) -> Iterable[tuple[ASTNode, Field, int | None]]:
        """Returns a generator object which yields all child nodes with their corresponding field
        and index (for tuples).

        Args:
            sort_keys (bool, optional): Whether to yield children by sorted field name.
                Defaults to False, meaning children will be yielded in the order they are defined.

        """

        # Dynamicaly generate a specialized function for this class
        yield from gen_and_yield_get_child_nodes_with_field(self, sort_keys=sort_keys)

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

        for child, f in self.iter_child_fields():
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
    __repr__ = repr_fn

    def __init_subclass__(cls) -> None:
        # Make sure even frozen dataclasses will not be hashable
        cls.__hash__ = None  # type: ignore[assignment]
        cls.__repr__ = repr_fn  # type: ignore[assignment]

        # Make sure subclass will use the functions that generate specialized
        # methods on the fly for each class
        # Instead of possibly triggering base classes methods that has already
        # been replaced with generated actual methods
        cls.iter_child_fields = gen_and_yield_iter_child_fields  # type: ignore[method-assign]
        cls.get_properties = gen_and_yield_get_properties  # type: ignore[method-assign]
        cls.get_child_nodes = gen_and_yield_get_child_nodes  # type: ignore[method-assign]
        cls.get_child_nodes_with_field = gen_and_yield_get_child_nodes_with_field  # type: ignore[method-assign]

        # Try checking type annotations now
        # At this point not all forward references may be resolved
        # so it may fail (i.e. skipped)
        if not check_annotations(cls, ASTNode) and config.TRACE_LOGGING:
            logger.debug(
                f"Annotations for {cls.__name__} could not be checked"
                " due to unresolved forward references."
            )

        return super(ASTNode, cls).__init_subclass__()
