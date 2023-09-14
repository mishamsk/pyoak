from __future__ import annotations

import enum
import hashlib
import logging
import sys
import weakref
from collections import deque
from collections.abc import Generator, Iterable
from dataclasses import InitVar, dataclass, field, replace
from inspect import getmro
from typing import TYPE_CHECKING, Any, Callable, Deque, Mapping, NamedTuple, Sequence, TypeVar, cast

from rich.markup import escape
from rich.tree import Tree

from . import config
from .codegen import (
    gen_and_yield_get_child_nodes,
    gen_and_yield_get_child_nodes_with_field,
    gen_and_yield_get_properties,
    gen_and_yield_iter_child_fields,
)
from .error import InvalidTypes
from .origin import NO_ORIGIN, Origin
from .serialize import TYPE_KEY, DataClassSerializeMixin
from .types import get_cls_all_fields, get_cls_child_fields, get_cls_props
from .typing import Field, FieldTypeInfo, check_annotations, is_instance

if TYPE_CHECKING:
    from .match.xpath import ASTXpath
    from .tree import Tree as PyOakTree
    from .visitor import ASTVisitor

_dataclass_extra_args: dict[str, bool] = {}
if sys.version_info >= (3, 11):
    # We rely on weakref slot, which is only available in py3.11+ for dataclasses
    # Also slots are only possible with mashumaro supporting SerializableType with slots
    from mashumaro.types import SerializableType

    if hasattr(SerializableType, "__slots__"):
        _dataclass_extra_args = dict(slots=True, weakref_slot=True)

logger = logging.getLogger(__name__)


_VRT = TypeVar("_VRT")


# A sentinel object to detect if a parameter is supplied or not.
# Use a subclass of str to make typing happy.
class _UNSET_ID_TYPE(str):
    pass


_UNSET_ID = _UNSET_ID_TYPE()


# Alternative implementations of dunder methods
def _hash_fn(node: ASTNode) -> int:
    return hash(node.id)


def _eq_fn(self: ASTNode, other: ASTNode) -> bool:
    # Other typed as ASTNode to make mypy happy
    if other.__class__ is self.__class__:
        return self.content_id == other.content_id and self.origin == other.origin

    return False


# Hack to make dataclasses InitVar work with future annotations
# See https://stackoverflow.com/questions/70400639/how-do-i-get-python-dataclass-initvar-fields-to-work-with-typing-get-type-hints
InitVar.__call__ = lambda *args: None  # type: ignore


def _check_runtime_types(node: ASTNode, type_map: Mapping[Field, FieldTypeInfo]) -> Sequence[Field]:
    incorrect_fields: list[Field] = []

    for f, type_info in type_map.items():
        val = getattr(node, f.name)
        if not is_instance(val, type_info.resolved_type):
            incorrect_fields.append(f)

    return incorrect_fields


NODE_REGISTRY: weakref.WeakValueDictionary[str, ASTNode] = weakref.WeakValueDictionary()
"""Registry of all node objects."""


def _get_next_unique_id(id_: str) -> str:
    """Return a unique ID."""
    i = 1
    original_id = id_
    while NODE_REGISTRY.get(id_) is not None:
        id_ = f"{original_id}_{i}"
        i += 1

    return id_


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


@dataclass(frozen=True, **_dataclass_extra_args)
class ASTNode(DataClassSerializeMixin):
    """A base class for all AST Node classes.

    Provides the following functions:
        - Maintains a weakref dictionary of all nodes, accessible through the class methods `get` and `get_any`
        - Various tree walking methods
        - Visitor API (accept method)
        - Serialization & deserialization
        - rich console API support (for printing)

    Notes:
        - Subclasses must be frozen dataclasses
        - Subclasses may be slotted, but this would prevent using multiple inheritance
        - Fields typed as union of subclasses of ASTNode or None as well as tuples of ASTNode subclasses
            are considered children. All other types that have ASTNode subclasses in signature
            will trigger an error.
    """

    id: str = field(default=_UNSET_ID, init=False, compare=False)
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

    content_id: str = field(
        default=_UNSET_ID,
        init=False,
        compare=False,
    )
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

    def __post_init__(self) -> None:
        if config.RUNTIME_TYPE_CHECK:
            incorrect_fields = _check_runtime_types(
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
        for val, f in sorted(
            self.get_properties(
                skip_id=True,
                skip_origin=True,
                skip_content_id=True,
                skip_non_compare=True,
            ),
            key=lambda x: x[1].name,
        ):
            cid_data += f":{f.name}="
            cid_data += f"{type(val)}({val!s})"

        # Full ID must include origin's (current node and children)
        id_data = f"{self.__class__.__name__}@{self.origin.fqn}{cid_data}"
        # Content ID - just use class
        cid_data = self.__class__.__name__ + cid_data

        for c, f, i in sorted(
            self.get_child_nodes_with_field(), key=lambda x: (x[1].name, x[2] or -1)
        ):
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

        # Generate ID
        new_id = hashlib.blake2b(
            id_data.encode("utf-8"), digest_size=config.ID_DIGEST_SIZE
        ).hexdigest()

        if existing_node := NODE_REGISTRY.get(new_id):
            # Node with the same ID already exists
            # This may mean two things:
            # 1. The same node (equality wise) is already in the registry
            # 2. Hash collision
            # For the latter case, we won't to rehash with a longer digest
            # to better preserve stable IDs. This is due to the fact
            # that the simple counting approach used in _get_next_unique_id
            # will depend on the order and number of collided nodes, thus
            # prone to more fluctuations based on the order of insertion.

            digest_increment = 1
            while existing_node is not None and not existing_node.is_equal(self):
                new_id = hashlib.blake2b(
                    id_data.encode("utf-8"),
                    # Do not bother to check digest is within allowed size
                    # since it allmost impossible to run out of digest size
                    digest_size=config.ID_DIGEST_SIZE + digest_increment,
                ).hexdigest()

                existing_node = NODE_REGISTRY.get(new_id)
                digest_increment += 1

            if existing_node is not None:
                # By now this is only possible if we are dealing with duplicate nodes
                new_id = _get_next_unique_id(new_id)

        # Assign ID
        object.__setattr__(self, "id", new_id)

        # Register in the registry
        NODE_REGISTRY[new_id] = self

    @classmethod
    def _deserialize(cls, value: dict[str, Any]) -> ASTNode:
        # Id must exist in the serialized data, otherwise
        # it is a corrupt data
        existing_node = NODE_REGISTRY.get(value["id"])

        # If >1 node was serialized with the same ID
        # it must have been the same object in memory
        # thus we want to deserialize it as the same object
        if existing_node is not None:
            return existing_node

        # Otherwise, we need to create a new node
        new_obj = super(ASTNode, cls)._deserialize(value)

        # If the node was serialized with ID that had collision
        # After recreating the node, it may not have the collision
        # yet and thus will have a different ID. We need to force
        # the ID to be the same as the serialized one and replace
        # the node in the registry
        if new_obj.id != value["id"]:
            NODE_REGISTRY.pop(new_obj.id)
            object.__setattr__(new_obj, "id", value["id"])
            NODE_REGISTRY[value["id"]] = new_obj

        return new_obj

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

    @classmethod
    def get(
        cls: type[ASTNodeType],
        id: str,
        default: ASTNodeType | None = None,
        strict: bool = True,
    ) -> ASTNodeType | None:
        """Gets a node of this class type from the AST registry.

        Args:
            id (str): The id of the node to get
            default (ASTNodeType | None, optional): The default value to return if the node is not found.
                Defaults to None.
            strict (bool, optional): If True, only a node of this class type is returned.
                Otherwise an instnace of any subclass is allowed. Defaults to True.

        Returns:
            ASTNodeType | None: The node if found, otherwise the default value
        """
        ret = NODE_REGISTRY.get(id)
        if ret is None:
            return default
        elif strict and not type(ret) == cls:
            return default
        elif not strict and not isinstance(ret, cls):
            return default
        else:
            return cast(ASTNodeType, ret)

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
        return NODE_REGISTRY.get(id, default)

    def detach(self) -> bool:
        """Removes this node from the registry.

        Returns:
            bool: True if the node was removed, False if it was not in the registry
        """
        return NODE_REGISTRY.pop(self.id, None) is not None

    def replace(self: ASTNodeType, **kwargs: Any) -> ASTNodeType:
        """Replaces this node in the registry with a new one with the given
        fields replaced.

        The key differences vs. using `dataclasses.replace`:
            - This method will remove the original node from the registry,
                but only if replace succeeds.
            - If the new node ID collides with the original node ID,
                the new node will preserve it, because the original
                node is purged from the registry before the new one
                is added.

        Args:
            **kwargs (Any): The fields to replace (see dataclasses.replace
                for more details)

        Raises:
            ValueError: see dataclasses.replace

        Returns:
            ASTNodeType: The new node
        """
        ori_n = NODE_REGISTRY.pop(self.id, None)

        try:
            new_node = replace(self, **kwargs)
        except Exception as e:
            if ori_n is not None:
                NODE_REGISTRY[ori_n.id] = ori_n

            raise e

        return new_node

    def duplicate(self: ASTNodeType) -> ASTNodeType:
        """Creates a full duplicate of the given node, recursively duplicating
        all children."""
        if config.TRACE_LOGGING:
            logger.debug(f"Duplicating an existing AST Node <{self.id}>")

        changes: dict[str, Any] = {}
        for obj, f in self.iter_child_fields():
            if isinstance(obj, ASTNode):
                changes[f.name] = obj.duplicate()
            elif isinstance(obj, tuple):
                changes[f.name] = tuple(c.duplicate() for c in obj)

        return replace(
            self,
            **changes,
        )

    @property
    def children(self) -> Sequence[ASTNode]:
        """Returns a static sequence with all child ASTNodes.

        Use `get_child_nodes` to iterate over
        """
        return list(self.get_child_nodes())

    def accept(self, visitor: ASTVisitor[_VRT]) -> _VRT:
        """Accepts a visitor by finding and calling a matching visitor method
        that should have a name in a form of visit_{__class__.__name__} or
        generic_visit if it doesn't exist.

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
        self,
    ) -> Iterable[tuple[ASTNode | tuple[ASTNode] | None, Field]]:
        """Iterates over all child fields of this node, returning the child
        field value as-is (whether it is None, a sequence or a child node) and
        the field itself.

        Returns:
            Iterable[tuple[ASTNode | tuple[ASTNode] | None, Field]]:
                An iterator of tuples of (child field value, field).
        """

        # Dynamicaly generate a specialized function for this class
        yield from gen_and_yield_iter_child_fields(self)

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
        obj_class: type[ASTNodeType] | tuple[type[ASTNodeType], ...],
        *,
        exact_type: bool = False,
        extra_filter: Callable[[NodeTraversalInfo], bool] | None = None,
        prune: Callable[[NodeTraversalInfo], bool] | None = None,
    ) -> Generator[ASTNodeType, None, None]:
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
        obj_classes: tuple[type[ASTNodeType], ...]
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
            yield cast(ASTNodeType, n_info.node)

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
    ) -> Iterable[tuple[Any, Field]]:
        """Returns an iterator of all properties (but not child attributes) of
        this node.

        Args:
            skip_id (bool, optional): Whether to skip the id property. Defaults to True.
            skip_origin (bool, optional): Whether to skip the origin property. Defaults to True.
            skip_content_id (bool, optional): Whether to skip the content_id property. Defaults to True.
            skip_non_compare (bool, optional): Whether to skip properties that are not used in comparison (field.comapre is False). Defaults to False.

        Yields:
            Iterable[tuple[Any, Field]]: An iterator of tuples of (value, field).
        """
        # Dynamicaly generate a specialized function for this class
        yield from gen_and_yield_get_properties(
            self, skip_id, skip_origin, skip_content_id, skip_non_compare, skip_non_init
        )

    def get_child_nodes(self) -> Iterable[ASTNode]:
        """Returns a generator object which yields all child nodes."""

        # Dynamicaly generate a specialized function for this class
        yield from gen_and_yield_get_child_nodes(self)

    def get_child_nodes_with_field(
        self,
    ) -> Iterable[tuple[ASTNode, Field, int | None]]:
        """Returns a generator object which yields all child nodes with their
        corresponding field and index (for tuples)."""

        # Dynamicaly generate a specialized function for this class
        yield from gen_and_yield_get_child_nodes_with_field(self)

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

    __hash__ = _hash_fn

    def __init_subclass__(cls) -> None:
        # Make sure subclasses use the same hash, eq functions
        # instead of the standard slow dataclass approach
        cls.__hash__ = _hash_fn  # type: ignore[assignment]
        cls.__eq__ = _eq_fn  # type: ignore[assignment]

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


ASTNodeType = TypeVar("ASTNodeType", bound=ASTNode)
