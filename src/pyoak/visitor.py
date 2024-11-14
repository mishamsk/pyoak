import sys
from abc import ABC, abstractmethod
from functools import singledispatch
from inspect import Parameter, getmembers, isfunction, signature
from typing import Any, Callable, ClassVar, Generic, Mapping, TypeVar

from .error import ASTTransformError
from .node import ASTNode

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

_VRT = TypeVar("_VRT")
_VIT = TypeVarTuple("_VIT")


class ASTVisitor(Generic[_VRT, Unpack[_VIT]], ABC):
    """A visitor generic base class for an AST visitor.

    Subclasses must implement a generic_visit method that will be called
    when no matching visit method is found.

    Subclasses can also implement visitor methods:

    >>> dev visit_any_suffix(self, node: NodeType, *args: *_VIT) -> _VRT:
    ...     pass

    which will be called when a matching node is visited.

    Methods are matched by the node type annotation of the second argument. If none
    is found a TypeError is raised. Type annotations must be simple subclass of
    ASTNode (or an ABC in non-strict mode), otherwise an exception is raised at class creation.

    Methods are matched differently based on the strict flag:
    - If strict is True, the method is matched by exact type only.
    - If strict is False, singledispatch is used - see stdlib functools.singledispatch.

    Note that visitor method dispatch logic is cached at class creation time (for
    method functions) and bound methods are cached on instances. Thus monkey patching
    visitor methods will not work.

    Optionally, when subclassing, you can set the `validate` class var to True.
    This will cause the visitor to validate that the method name matches the
    node type annotation. E.g.

    >>> class MyVisitor(ASTVisitor[None], validate=True):
    ...     def visit_MyNode(self, node: MyNode) -> None:
    ...         pass
    ...     def generic_visit(self, node: ASTNode) -> None:
    ...         pass
    ...     def visit_WrongNamedNode(self, node: MyOtherNode) -> None:
    ...         pass

    will raise a TypeError because the last method name doesn't match the node
    type annotation.

    Defaul `visit` method accepts node to visit and arbitrary positional args.
    Types of the extra args can be specified with a type annotation on the class
    itself. E.g.

    >>> class MyVisitor(ASTVisitor[str, int]):
    ...     def visit_MyNode(self, node: MyNode, arg: int) -> str:
    ...         pass

    will accept an additional integer argument when visiting a MyNode. Due to the
    limitations of Python's type system, only positional only arguments are supported.

    If you need a more complicated `visit` method signature, you can override the
    `generic_visit`, `visit` methods and use the `_dispatch_visit_method` in
    your `visit` implementation to get a matching visitor method for a given node.
    Note that this will raise type errors in static type checkers if additional
    arguments do not have default values.

    Args:
        t (_type_): Vistior return type

    """

    # Classvar to store the dispatcher for the visit method
    # when single dispatch is used.
    __dispatcher__: ClassVar[Any]

    # Classvar to store the node type to method mapping. Automatically
    # populated by __init_subclass__.
    __visit_methods_registry__: ClassVar[Mapping[type[ASTNode], Callable[..., _VRT]]]  # type: ignore[misc]

    # This is a class level cache for dispatching to unbound visit methods.
    __unbound_visitor_dispatch_cache__: ClassVar[dict[type[ASTNode], Callable[..., _VRT]]]  # type: ignore[misc]

    strict: ClassVar[bool] = False
    """Strict visitors match visit methods to nodes by exact type.

    Non-strict visitors will match using singledispatch, which means that the method for a node type
    will be found by walking the extneded MRO of the node type until a matching method is found.

    See functools.singledispatch for more information.

    """

    def __init__(self) -> None:
        # Create an instance cache for dispatching which will have
        # bound methods instead of class functions.
        self.__bound_visitor_dispatch_cache__: dict[type[ASTNode], Callable[..., _VRT]] = {}

    @classmethod
    def __dispatch_visit_method(cls, node_type: type[ASTNode]) -> Callable[..., _VRT]:
        """Returns an unbounded visit method for a given node type.

        Dispatching is done based on a registry of methods, looked up on the class.
        All methods that look like: `visit[arbitrary suffix](self, node: ASTNodeType, ...)`
        are inspected and the second argument's type annotation is used to determine
        which method to call.

        If the visitor `strict` class var is True, then visit method is matched by
        the exact type match. Otherwise stdlib's singledispatch is used.

        """

        if cls.strict:
            # Check if we have cached unbound method for this node type
            visitor_method = cls.__unbound_visitor_dispatch_cache__.get(node_type)

            if visitor_method is not None:
                # We have a cached undound method, return it
                return visitor_method

            # Strict mode, match by exact type only
            visitor_method = cls.__visit_methods_registry__.get(node_type)

            if visitor_method is None:
                # If we didn't find a visitor method, use generic_visit
                visitor_method = cls.generic_visit

            # Cache the undound method (this may just rewrite the existing one)
            cls.__unbound_visitor_dispatch_cache__[node_type] = visitor_method

            return visitor_method

        # Non-strict mode, match by singledispatch
        return cls.__dispatcher__.dispatch(node_type)  # type: ignore[no-any-return]

    def _dispatch_visit(self, node: ASTNode) -> Callable[..., _VRT]:
        """Returns a bound visit method for a given node.

        You can use it when overriding visit method directly, if you want to extend visit method
        signatures in a subclass.

        """

        # Check if we already have a bound method for this node type
        visitor_bound_method = self.__bound_visitor_dispatch_cache__.get(node.__class__)

        if visitor_bound_method is not None:
            # We have a bound method, return it
            return visitor_bound_method

        # We don't have a bound method. First find an undound visitor method
        visitor_method = self.__dispatch_visit_method(node.__class__)

        # Create a bound method and cache it
        self.__bound_visitor_dispatch_cache__[node.__class__] = visitor_bound_method = (
            visitor_method.__get__(self, self.__class__)
        )

        return visitor_bound_method

    @abstractmethod
    def generic_visit(self, node: ASTNode, *args: Unpack[_VIT]) -> _VRT:
        raise NotImplementedError

    def visit(self, node: ASTNode, *args: Unpack[_VIT]) -> _VRT:
        """Visits the given node by finding and calling a matching visitor method or generic_visit
        if it doesn't exist.

        Args:
            node (ASTNode): The node to visit.

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

        return self._dispatch_visit(node)(node, *args)

    def __init_subclass__(cls, *, validate: bool = False) -> None:
        """Iterate over new visitor methods and check that names match the node type annotation."""

        cls.__dispatcher__ = singledispatch(cls.visit)
        cls.__dispatcher__.register(object, cls.generic_visit)

        # Make sure each subclass has its own dispatch cache
        cls.__unbound_visitor_dispatch_cache__ = {}

        visit_method_registry: dict[type[ASTNode], Callable[..., _VRT]] = {}

        errors: list[tuple[str, str]] = []
        for method_name, method in getmembers(cls, isfunction):
            if method_name.startswith("visit_"):
                try:
                    sig = signature(method, eval_str=True)
                except Exception as e:
                    errors.append(
                        (
                            method_name,
                            f"Invalid signature or a string annotation that can't be resolved: {e}",
                        )
                    )
                    continue

                if len(sig.parameters) < 2:
                    errors.append(
                        (method_name, "Method must have at least two parameters: self and node")
                    )
                    continue

                node_arg_type = list(sig.parameters.values())[1].annotation

                if node_arg_type is Parameter.empty:
                    errors.append((method_name, "Node type annotation is missing"))
                    continue

                try:
                    if not issubclass(node_arg_type, (ASTNode, ABC)):
                        errors.append(
                            (
                                method_name,
                                "Node type annotation must be a subclass of ASTNode or an ABC",
                            )
                        )
                        continue
                except TypeError:
                    errors.append(
                        (method_name, "Node type annotation must be a single ASTNode subclass")
                    )
                    continue

                if node_arg_type in visit_method_registry:
                    errors.append(
                        (
                            method_name,
                            f"Node type '{node_arg_type.__name__}' already has a visit method "
                            f"'{visit_method_registry[node_arg_type].__name__}'. "
                            "Multiple visit methods for the same node type are not allowed",
                        )
                    )
                    continue

                if validate:
                    expected_node_type = method_name[6:]
                    if node_arg_type.__name__ != expected_node_type:
                        errors.append(
                            (
                                method_name,
                                "Method name doesn't match the second argument type annotation",
                            )
                        )
                        continue

                visit_method_registry[node_arg_type] = method
                cls.__dispatcher__.register(node_arg_type, method)

        if errors:
            raise TypeError(
                f"Visitor class '{cls.__name__}' method(s) have invalid signature(s):\n  - "
                + "\n  - ".join(
                    f"'{method_name}': {error}"
                    for method_name, error in sorted(errors, key=lambda x: x[0])
                )
            )

        cls.__visit_methods_registry__ = visit_method_registry

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

    def _transform_children(self, node: ASTNode) -> Mapping[str, Any]:
        """Transforms the children of a given node and returns a mapping of field names to changes.

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
        """Transforms children of the given node and returns a new node with the changes."""

        changes = self._transform_children(node)

        # No changes, return the original node
        if not changes:
            return node

        # Return a new node with the changes
        return node.replace(**changes)

    def transform(self, node: ASTNode) -> ASTNode | None:
        """Transforms a given node and returns the transformed node or None if the node was removed.

        Args:
            node (ASTNode): The node to transform

        Returns:
            ASTNode | None: The transformed node or None if the node was removed.

        Raises:
            ASTTransformError: If the transformation fails with the original exception
                as context.

        """
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
