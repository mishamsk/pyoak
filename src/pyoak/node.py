from __future__ import annotations

import enum
import hashlib
import logging
from collections import deque
from collections.abc import Generator, Iterable
from dataclasses import InitVar, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Deque, Mapping, NamedTuple, Sequence, TypeVar, cast

from rich.markup import escape
from rich.tree import Tree

from . import config
from ._methods import eq_fn, hash_fn
from .codegen import (
    gen_and_yield_get_child_nodes,
    gen_and_yield_get_child_nodes_with_field,
    gen_and_yield_get_properties,
    gen_and_yield_iter_child_fields,
)
from .error import ASTRefCollisionError, InvalidTypes
from .origin import NO_ORIGIN, Origin
from .registry import _get_node, _pop_node, _register, _register_with_ref
from .serialize import TYPE_KEY, DataClassSerializeMixin
from .types import get_cls_all_fields, get_cls_child_fields, get_cls_props
from .typing import Field, FieldTypeInfo, check_annotations, check_runtime_types

if TYPE_CHECKING:
    from .match.xpath import ASTXpath
    from .tree import Tree as PyOakTree


logger = logging.getLogger(__name__)


_ASTNodeType = TypeVar("_ASTNodeType", bound="ASTNode")


# Named Tuple for tree traversal functions
class NodeTraversalInfo(NamedTuple):
    node: ASTNode
    parent: ASTNode
    field: Field
    findex: int | None = None


AST_SERIALIZE_DIALECT_KEY = "ast_serialize_dialect"


class ASTSerializationDialects(enum.Enum):
    AST_EXPLORER = enum.auto()
    AST_TEST = enum.auto()


_REF_ATTR = "__node_ref__"


def _attach(node: ASTNode, ref_id: str) -> None:
    if not _register_with_ref(node, ref_id):
        raise ASTRefCollisionError(ref_id)

    object.__setattr__(node, _REF_ATTR, ref_id)


# dataclasses prior to py3.11 didn't support __weakref__ slot
# so in order to make a universally supported base class
# we need to add it manually via a mixin
class _NodeSlots:
    __slots__ = ("__weakref__", _REF_ATTR)


@dataclass(frozen=True, slots=True)
class ASTNode(DataClassSerializeMixin, _NodeSlots):
    """A base class for all AST Node classes.

    Provides the following functions:
        - Maintains a weakref dictionary of all nodes, accessible through the class methods `get` and `get_any`
        - Various tree walking methods
        - Visitor API (accept method)
        - Serialization & deserialization
        - rich console API support (for printing)

    Notes:
        - Subclasses must be frozen dataclasses
        - Subclasses may be slotted, but remember that in multiple inheritance, only one base can have non-empty slots
        - Fields typed as union of subclasses of ASTNode or None as well as tuples of ASTNode subclasses
            are considered children. All other types that have ASTNode subclasses in signature
            will trigger an error.
    """

    id: str = field(init=False, compare=False)
    """The unique ID of this node. It will be auto-generated based on the
    node's properties and origin. Uniqueness is ensured for all nodes in the
    registry, thus as long as an identical node is in memory, a new node will
    get a different ID.

    Two nodes with the same "content" (i.e. properties and children)
    but generated with a different origin will have get different IDs.

    For a stable (but non-unique) ID based purely on the node's content,
    use the `content_id` property.

    Note:
        Although ID may be provided manually, it is not recommended to do so.
        Instead use properties to store natural (e.g. source system) identifiers of the node.
    """

    content_id: str = field(init=False, compare=False)
    """The ID of this node based on it's content (i.e. properties and
    children).

    Unlike the `id` property, this ID is stable and will be the same for two nodes
    with the same content, regardless of their origin or existence of multiple
    copies in memory.

    Naturally, this id can't be used to identify a node in the registry, since
    it is by definition non-unique.
    """

    origin: Origin = field(default=NO_ORIGIN, kw_only=True)
    """The origin of this node.

    Provides information of the source (such as text file) and it's
    path, as well as position within the source (e.g. char index)
    """

    attached: InitVar[bool] = field(default=False, kw_only=True)

    def __post_init__(self, attached: bool) -> None:
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

        # Calculate ID's

        # Build common data for content_id & id
        cid_data = ""

        # Property values are encoded in the same way for both id's
        for val, f in self.get_properties(
            skip_id=True,
            skip_origin=True,
            skip_content_id=True,
            skip_non_compare=True,
            sort_keys=True,
        ):
            cid_data += f":{f.name}="
            cid_data += f"{type(val)}({val!s})"

        # Full ID must include origin's (current node and children)
        id_data = f"{self.__class__.__name__}@{self.origin.fqn}{cid_data}"
        # Content ID - just use class
        cid_data = self.__class__.__name__ + cid_data

        for c, f, i in self.get_child_nodes_with_field(sort_keys=True):
            resolved_index = i or -1
            cid_data += f":{f.name}[{resolved_index}]="
            cid_data += f"{c.content_id}"
            id_data += f":{f.name}[{resolved_index}]="
            id_data += f"{c.content_id}@{c.origin.fqn}"

        # Assign content_id
        object.__setattr__(
            self,
            "content_id",
            hashlib.blake2b(
                cid_data.encode("utf-8"), digest_size=config.ID_DIGEST_SIZE
            ).hexdigest(),
        )

        # Assign ID
        object.__setattr__(
            self,
            "id",
            hashlib.blake2b(id_data.encode("utf-8"), digest_size=config.ID_DIGEST_SIZE).hexdigest(),
        )

        # Attach to registry if needed
        if attached:
            object.__setattr__(self, _REF_ATTR, _register(self))

    @property
    def ref(self) -> str | None:
        """The registry reference of this node if it is attached."""
        return getattr(self, _REF_ATTR, None)

    @property
    def ref_or_raise(self) -> str:
        """The registry reference of this node if it is attached or raise
        AttributeError if not."""
        return cast(str, getattr(self, _REF_ATTR))

    @classmethod
    def _deserialize(cls, value: dict[str, Any]) -> ASTNode:
        # Get an optional ref value (to be non-destructive, mashumaro allows extra fields)
        ref_id = value.get(_REF_ATTR, None)

        # create a new node as usual
        new_obj = super(ASTNode, cls)._deserialize(value)

        # If the node was serialized with ref, try to register it
        if ref_id is not None:
            _attach(new_obj, ref_id)

        return new_obj

    def __post_serialize__(self, d: dict[str, Any]) -> dict[str, Any]:
        # Run first, otherwise _children will be dropped from the output
        out = super(ASTNode, self).__post_serialize__(d)

        # Save ref value if it exists
        if self.ref is not None:
            out[_REF_ATTR] = self.ref

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

    def to_attached(self: _ASTNodeType) -> _ASTNodeType:
        """Returns an attached copy of this node.

        If the node is already attached, returns itself.
        """
        if self.ref is not None:
            return self

        return replace(self, attached=True)

    @classmethod
    def get(
        cls: type[_ASTNodeType],
        ref: str | None,
        default: _ASTNodeType | None = None,
        strict: bool = True,
    ) -> _ASTNodeType | None:
        """Gets a node of this class type from the AST registry.

        Only nodes that were previously added to the registry via `ASTNode.get_ref` method can be retrieved.

        Args:
            ref (str | None): The ref value of the node to get. None is for convenience, since nodes may not have a ref value.
            default (ASTNodeType | None, optional): The default value to return if the node is not found.
                Defaults to None.
            strict (bool, optional): If True, only a node of this class type is returned.
                Otherwise an instnace of any subclass is allowed. Defaults to True.

        Returns:
            ASTNodeType | None: The node if found, otherwise the default value
        """
        if ref is None:
            return default

        ret = _get_node(ref)
        if ret is None:
            return default
        elif strict and not type(ret) == cls:
            return default
        elif not strict and not isinstance(ret, cls):
            return default
        else:
            return cast(_ASTNodeType, ret)

    @classmethod
    def get_any(cls, ref: str | None, default: ASTNode | None = None) -> ASTNode | None:
        """Gets a node of any type from the AST registry by ref id.

        Only nodes that were previously added to the registry via `ASTNode.get_ref` method can be retrieved.

        Args:
            ref (str | None): The ref value of the node to get. None is for convenience, since nodes may not have a ref value.
            default (ASTNode | None, optional): The default value to return if the node is not found.
                Defaults to None.

        Returns:
            ASTNode | None: The node if found, otherwise the default value
        """
        if ref is None:
            return default

        return _get_node(ref, default)

    def detach(self) -> None:
        """Removes this node and and the whole tree rooted with this node from
        the registry."""
        if _pop_node(self) is not None:
            object.__delattr__(self, _REF_ATTR)

        for ni in self.dfs():
            if _pop_node(ni.node) is not None:
                object.__delattr__(ni.node, _REF_ATTR)

    def detach_self(self) -> bool:
        """Removes this node from the registry.

        Returns:
            bool: True if the node was removed, False if it was not in the registry
        """
        if _pop_node(self) is not None:
            object.__delattr__(self, _REF_ATTR)
            return True

        return False

    def replace(self: _ASTNodeType, **kwargs: Any) -> _ASTNodeType:
        """Creates a new node by replacing values from `kwargs` and replacing
        this node in the registry with a new one if it was previously
        registered via `ASTNode.get_ref`.

        The difference vs. `dataclasses.replace`:
            - This method will remove the original node from the registry, if it was there
                but only if replace succeeds.

        Args:
            **kwargs (Any): The fields to replace (see dataclasses.replace
                for more details)

        Raises:
            ValueError: see dataclasses.replace
            ASTRefCollisionError: Unlikely, but may be raised if ref value collision occurs

        Returns:
            ASTNodeType: The new node
        """
        new_node = replace(self, **kwargs)

        if self.ref is not None:
            # Remember the original ref value
            ref = self.ref

            # Detach the original node
            self.detach_self()

            # Register the new node with the same ref value
            _attach(new_node, ref)

        return new_node

    def replace_with(self, other: _ASTNodeType) -> _ASTNodeType:
        """Replaces this node with another one by creating an attached copy of
        the `other` node with the same ref as this node and replacing it in the
        registry.

        If this node is not in the registry, returns the other node without
        modification.

        Args:
            other (ASTNode): The node to replace this one with.

        Raises:
            ASTRefCollisionError: Unlikely, but may be raised if ref value collision occurs
        """
        if self.ref is None:
            return other

        # make a fresh copy of the other node
        other = replace(other)

        # Remember the original ref value
        ref = self.ref

        # Detach the original node
        self.detach_self()

        # Register the new node with the same ref value
        _attach(other, ref)

        return other

    @property
    def children(self) -> Sequence[ASTNode]:
        """Returns a static sequence with all child ASTNodes.

        Use `get_child_nodes` to iterate over
        """
        return list(self.get_child_nodes())

    def is_equal(self, other: Any) -> bool:
        """Returns True if this node is equal to `other`.

        Unlike `==`, this method only compares properties & children and
        ignores the origin and compares by content_id.
        """
        if type(other) is not type(self):
            return False

        return self.content_id == cast(ASTNode, other).content_id

    def to_properties_dict(self) -> dict[str, Any]:
        """Returns a dictionary of all node properties (not children),
        including only "content", i.e. origin, id, origin_id, id_collision,
        parent_id and hidden attributes are not included."""
        d = {}

        for v, f in self.get_properties():
            d[f.name] = v

        return d

    def to_tree(self) -> PyOakTree:
        """Returns a pyoak.tree.Tree object with this node as root."""

        from .tree import Tree as PyOakTree

        return PyOakTree(self)

    def iter_child_fields(
        self, *, sort_keys: bool = False
    ) -> Iterable[tuple[ASTNode | tuple[ASTNode] | None, Field]]:
        """Iterates over all child fields of this node, returning the child
        field value as-is (whether it is None, a sequence or a child node) and
        the field itself.

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
        prune: Callable[[NodeTraversalInfo], bool] | None = None,
        filter: Callable[[NodeTraversalInfo], bool] | None = None,
        bottom_up: bool = False,
    ) -> Generator[NodeTraversalInfo, None, None]:
        """Returns a generator object which yields all nodes in a tree, with
        this node as root in the DFS (Depth-first) order. It doesn't yield the
        node itself.

        Args:
            prune (Callable[[NodeTraversalInfo], bool] | None, optional):
                An optional function which if it returns True will prevent
                further decent into the children of this element.
            filter (Callable[[NodeTraversalInfo], bool] | None, optional):
                An optional function which if it returns False will prevent
                the element from being yielded, but won't interrupt the recursive decent/ascent.
            bottom_up (bool, optional): Enables bottom up traversal. Defaults to False.

        Yields:
            Generator[NodeTraversalInfo, None, None]:
                A generator object which yields all sub-nodes in the DFS (Depth-first) order.
        """

        build_stack: list[NodeTraversalInfo] = []
        yield_queue: Deque[NodeTraversalInfo] = deque()

        if bottom_up:
            appender = yield_queue.appendleft
        else:
            appender = yield_queue.append

        children_info = list(self.get_child_nodes_with_field())

        if not bottom_up:
            children_info.reverse()

        for c, f, i in children_info:
            build_stack.append(NodeTraversalInfo(c, self, f, i))

        while build_stack:
            child_info = build_stack.pop()

            if filter is None or filter(child_info):
                appender(child_info)

            if prune and prune(child_info):
                continue

            children_info = list(child_info.node.get_child_nodes_with_field())

            if not bottom_up:
                children_info.reverse()

            for c, f, i in children_info:
                build_stack.append(NodeTraversalInfo(c, child_info.node, f, i))

        while yield_queue:
            yield yield_queue.popleft()

    def bfs(
        self,
        prune: Callable[[NodeTraversalInfo], bool] | None = None,
        filter: Callable[[NodeTraversalInfo], bool] | None = None,
    ) -> Generator[NodeTraversalInfo, None, None]:
        """Returns a generator object which visits all nodes in this tree in
        the BFS (Breadth-first) order.

        Args:
            prune (Callable[[NodeTraversalInfo], bool]): An optional function which if it returns True will prevent further decent into the children of this element.
            filter (Callable[[NodeTraversalInfo], bool]): An optional function which if it returns False will prevent the element from being yielded, but won't interrupt the recursive decent.

        Returns:
            the generator object.
        """

        queue: Deque[NodeTraversalInfo] = deque(
            (NodeTraversalInfo(c, self, f, i) for c, f, i in self.get_child_nodes_with_field())
        )

        while queue:
            child = queue.popleft()

            if filter is None or filter(child):
                yield child

            if prune and prune(child):
                continue

            # Walk through children
            queue.extend(
                NodeTraversalInfo(c, child.node, f, i)
                for c, f, i in child.node.get_child_nodes_with_field()
            )

    def gather(
        self,
        obj_class: type[_ASTNodeType] | tuple[type[_ASTNodeType], ...],
        *,
        exact_type: bool = False,
        extra_filter: Callable[[NodeTraversalInfo], bool] | None = None,
        prune: Callable[[NodeTraversalInfo], bool] | None = None,
    ) -> Generator[_ASTNodeType, None, None]:
        """Shorthand for traversing the tree and gathering all instances of
        subclasses of `obj_class` or exactly `obj_class` if `exact_type` is
        True.

        This function will not yield the node itself.

        Args:
            obj_class (type[ASTNodeType] | tuple[type[ASTNodeType], ...]): any ASTNode subclass or Tuple of classes to gather.
            exact_type (bool, optional): Whether to only gather instances of `obj_class` and not its subclasses. Defaults to False.
            extra_filter (Callable[[NodeTraversalInfo], bool] | None, optional): An optional additional filter to apply when gathering. Defaults to None.
            prune (Callable[[NodeTraversalInfo], bool] | None, optional): Optional function to stop traversal. Defaults to None.
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

            def filter_fn(node_info: NodeTraversalInfo) -> bool:
                return isinstance(node_info.node, obj_classes) and (
                    extra_filter is None or extra_filter(node_info)
                )

        else:

            def filter_fn(node_info: NodeTraversalInfo) -> bool:
                return type(node_info.node) in obj_classes and (
                    extra_filter is None or extra_filter(node_info)
                )

        for n_info in self.dfs(prune=prune, filter=filter_fn, bottom_up=False):
            yield cast(_ASTNodeType, n_info.node)

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
        skip_content_id: bool = True,
        skip_non_compare: bool = False,
        skip_non_init: bool = False,
    ) -> Iterable[Field]:
        """Returns an iterator of all properties (but not child attributes) of
        this node using static type information.

        Args:
            skip_id (bool, optional): Whether to skip the id property. Defaults to True.
            skip_origin (bool, optional): Whether to skip the origin property. Defaults to True.
            skip_content_id (bool, optional): Whether to skip the content_id property. Defaults to True.
            skip_non_compare (bool, optional): Whether to skip properties that are not used in comparison (field.comapre is False). Defaults to False.

        Yields:
            Iterable[tuple[str, Field]]: An iterator of tuples of (field name, field).
        """

        for f in get_cls_props(cls):
            # Skip id
            if (f.name == "id") and skip_id:
                continue

            # Skip content_id
            if f.name == "content_id" and skip_content_id:
                continue

            # Skip origin
            if f.name == "origin" and skip_origin:
                continue

            # Skip non-comparable fields
            if not f.compare and skip_non_compare:
                continue

            # Skip non-init fields
            if not f.init and skip_non_init:
                continue

            yield f

    @classmethod
    def get_child_fields(
        cls,
    ) -> Mapping[Field, FieldTypeInfo]:
        """Returns an iterator of all child attributes of this node using
        static type information.

        Returns:
            Mapping[Field, ChildFieldTypeInfo]:
                A mapping of child attribute name to (field, type_info).
        """
        return get_cls_child_fields(cls)

    def get_properties(
        self,
        skip_id: bool = True,
        skip_origin: bool = True,
        skip_content_id: bool = True,
        skip_non_compare: bool = False,
        skip_non_init: bool = False,
        *,
        sort_keys: bool = False,
    ) -> Iterable[tuple[Any, Field]]:
        """Returns an iterator of all properties (but not child attributes) of
        this node.

        Args:
            skip_id (bool, optional): Whether to skip the id property. Defaults to True.
            skip_origin (bool, optional): Whether to skip the origin property. Defaults to True.
            skip_content_id (bool, optional): Whether to skip the content_id property. Defaults to True.
            skip_non_compare (bool, optional): Whether to skip properties that are not used in comparison (field.comapre is False). Defaults to False.
            skip_non_init (bool, optional): Whether to skip properties that are not used in initialization (field.init is False). Defaults to False.
            sort_keys (bool, optional): Whether to sort the properties by field name. Defaults to False.

        Yields:
            Iterable[tuple[Any, Field]]: An iterator of tuples of (value, field).
        """
        # Dynamicaly generate a specialized function for this class
        yield from gen_and_yield_get_properties(
            self,
            skip_id,
            skip_origin,
            skip_content_id,
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
        """Returns a generator object which yields all child nodes with their
        corresponding field and index (for tuples).

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
                    c._rich(subtree, f)
            elif child is None:
                tree.add(f":file_folder:[yellow]{f.name}[/]={escape(str(None))}")
            else:
                child._rich(tree, f)

        return tree

    __hash__ = hash_fn

    def __init_subclass__(cls) -> None:
        # Make sure subclasses use the same hash, eq functions
        # instead of the standard slow dataclass approach
        cls.__hash__ = hash_fn  # type: ignore[assignment]
        cls.__eq__ = eq_fn  # type: ignore[assignment]

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
