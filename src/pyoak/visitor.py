import weakref
from abc import ABC, abstractmethod
from inspect import Parameter, getmembers, getmro, isfunction, signature
from typing import Any, Callable, Generic, Mapping, TypeVar

from .error import ASTTransformError
from .node import ASTNode

_VRT = TypeVar("_VRT")


class ASTVisitor(Generic[_VRT], ABC):
    """A visitor generic base class for an AST visitor.

    Subclasses must implement a generic_visit method that will be called
    when no matching visit method is found.

    Subclasses can also implement visit_{node_type}(self, node: NodeType) methods
    that will be called when a matching node is visited.

    Methods are matched by the node type annotation of the second argument. If none
    is found a TypeError is raised. Type annotations must be simple subclasses of
    ASTNode, otherwise an exception is raised.

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

    This will raise a TypeError because the last method name doesn't match the node
    type annotation.

    Defaul `visit` method assumes only one argument, the node to visit. If you need
    to pass additional arguments to the visitor methods, you can override the
    `generic_visit`, `visit` methods and use the `_dispatch_visit_method` in
    your `visit` implementation to get a matching visitor method for a given node.
    Note that this will raise type errors in static type checkers if additional
    arguments do not have default values.

    Args:
        t (_type_): Vistior return type

    """

    # This is both a class var for initial cache built at class creation
    # and an instance var for dispatching to bound methods.
    __dispatch_cache__: weakref.WeakKeyDictionary[type[ASTNode], Callable[..., _VRT]]

    strict: bool = False
    """Strict visitors match visit methods to nodes by exact type.

    Non-strict visitors will match by isinstance check in MRO order.

    """

    def __init__(self) -> None:
        # Create an instance cache for dispatching which will have
        # bound methods instead of class functions.
        self.__dispatch_cache__ = weakref.WeakKeyDictionary()

    def _dispatch_visit_method(self, node: ASTNode) -> Callable[..., _VRT]:
        """Returns a visit method for a given node. Usefull for overriding visit method signature in
        subclasses.

        Dispatching is done based on `visit_{__class__.__name__}(self, node, ...)`
        second argument type annotation. By default method name itself is ignored,
        unless `validate` is set to True when subclassing the visitor.

        If the visitor `strict` class var is True, then visit method is matched by
        the exact type match. Otherise the node mro is walked in reverse order until
        until an exact match is found. Unlike stdlib singledispatch, we are not
        checking for abstract (virtual) base classes.

        """

        visitor_bound_method = self.__dispatch_cache__.get(node.__class__)

        if visitor_bound_method is not None:
            return visitor_bound_method

        visitor_method = None

        if self.strict:
            visitor_method = self.__class__.__dispatch_cache__.get(node.__class__)
        else:
            mro = getmro(node.__class__)
            for _class in mro[:-1]:
                visitor_method = self.__class__.__dispatch_cache__.get(_class, None)
                if visitor_method is not None:
                    break

        if visitor_method is not None:
            visitor_bound_method = visitor_method.__get__(self, self.__class__)
        else:
            visitor_bound_method = self.generic_visit

        self.__dispatch_cache__[node.__class__] = visitor_bound_method

        return visitor_bound_method

    @abstractmethod
    def generic_visit(self, node: ASTNode) -> _VRT:
        raise NotImplementedError

    def visit(self, node: ASTNode) -> _VRT:
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

        return self._dispatch_visit_method(node)(node)

    def __init_subclass__(cls, *, validate: bool = False) -> None:
        """Iterate over new visitor methods and check that names match the node type annotation."""
        cls.__dispatch_cache__ = weakref.WeakKeyDictionary()

        errors: list[tuple[str, str]] = []
        for method_name, method in getmembers(cls, isfunction):
            if method_name.startswith("visit_"):
                sig = signature(method, eval_str=True)

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
                    if not issubclass(node_arg_type, ASTNode):
                        errors.append(
                            (method_name, "Node type annotation must be a subclass of ASTNode")
                        )
                        continue
                except TypeError:
                    errors.append(
                        (method_name, "Node type annotation must be a single ASTNode subclass")
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

                cls.__dispatch_cache__[node_arg_type] = method

        if errors:
            raise TypeError(
                f"Visitor class '{cls.__name__}' method(s) have invalid signature(s):\n  - "
                + "\n  - ".join(
                    f"'{method_name}': {error}"
                    for method_name, error in sorted(errors, key=lambda x: x[0])
                )
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
