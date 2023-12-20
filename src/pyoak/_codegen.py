from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping

from pyoak.types import get_cls_child_fields, get_cls_props

from . import config

if TYPE_CHECKING:
    from dataclasses import Field as DataClassField

    from .node import ASTNode
    from .typing import FieldTypeInfo

    Field = DataClassField[Any]


_IND = " " * 4


def _indent(txt: str, times: int = 1) -> str:
    return "\n".join([_IND * times + line if line.strip() else line for line in txt.split("\n")])


def _gen_func(
    clz: type[ASTNode],
    fname: str,
    ret_type: str,
    body: str,
    local_vars: dict[str, Any],
    *,
    extra_args: str = "",
) -> None:
    """Generate a specialized function and assing to the class."""

    # The code of the actual function
    if not extra_args:
        def_stat = f"{_IND}def {fname}(self) -> {ret_type}:\n"
    else:
        def_stat = f"{_IND}def {fname}(self, {extra_args}) -> {ret_type}:\n"

    inner = def_stat + _indent(body, 2)

    # Print the code if debug is enabled
    if config.CODEGEN_DEBUG:
        print(inner)

    # From stdlib.dataclasses. Create a function in a new namespace
    # that will create a closure and return the function
    txt = f"def __create_fn__({', '.join(local_vars.keys())}):\n{inner}\n{_IND}return {fname}"
    ns: dict[str, Any] = {}
    exec(txt, None, ns)

    # Execute the function to create the closure
    new_f = ns["__create_fn__"](**local_vars)

    # Set qualname to the proper function name
    new_f.__qualname__ = f"{clz.__qualname__}.{new_f.__name__}"

    # Set the function to the class
    setattr(clz, new_f.__name__, new_f)


def _gen_get_child_nodes_func(
    clz: type[ASTNode], child_fields: Mapping[Field, FieldTypeInfo]
) -> None:
    """Generate a specialized function for getting all child nodes."""
    fname = "get_child_nodes"

    # Hols closure vars with Field objects
    local_vars = {}

    # The code of the actual function
    ret_type = "Iterable[ASTNode]"

    # If a node doesn't have child fields, the above loop will not produce any code
    # but we need to make the function a generator, so we yield nothing
    if len(child_fields) == 0:
        body = "yield from ()\n"
    else:
        # Iterate over all child fields and create sorted & unsroted versions

        # First the sorted version
        body = "if sort_keys:\n"

        def _build_body(f: Field, type_info: FieldTypeInfo) -> None:
            nonlocal body

            if type_info.is_collection:
                # Collection creates an for loop
                body += f"{_IND}for o in self.{f.name}:\n"
                body += f"{_IND*2}yield o\n"
            else:
                # Non-collection yield if child is not None
                body += f"{_IND}if self.{f.name}:\n"
                body += f"{_IND*2}yield self.{f.name}\n"

        for f, type_info in sorted(child_fields.items(), key=lambda x: x[0].name):
            # Store the field object in the closure
            local_vars[f"_fld_{f.name}"] = f

            _build_body(f, type_info)

        # Now the unsorted version
        body += "else:\n"

        for f, type_info in child_fields.items():
            _build_body(f, type_info)

    _gen_func(clz, fname, ret_type, body, local_vars, extra_args="sort_keys: bool = False")


def _gen_get_child_nodes_with_field_func(
    clz: type[ASTNode], child_fields: Mapping[Field, FieldTypeInfo]
) -> None:
    """Generate a specialized function for getting child nodes with their corresponding field and
    index."""
    fname = "get_child_nodes_with_field"

    # Hols closure vars with Field objects
    local_vars = {}

    # The code of the actual function
    ret_type = "Iterable[tuple[ASTNode, Field, int | None]]"

    # If a node doesn't have child fields, the above loop will not produce any code
    # but we need to make the function a generator, so we yield nothing
    if len(child_fields) == 0:
        body = "yield from ()\n"
    else:
        # Iterate over all child fields and create sorted & unsroted versions

        # First the sorted version
        body = "if sort_keys:\n"

        def _build_body(f: Field, type_info: FieldTypeInfo) -> None:
            nonlocal body

            if type_info.is_collection:
                # Collection creates an for loop
                body += f"{_IND}for i, o in enumerate(self.{f.name}):\n"
                body += f"{_IND*2}yield o, _fld_{f.name}, i\n"
            else:
                # Non-collection yield id child is not None
                body += f"{_IND}if self.{f.name}:\n"
                body += f"{_IND*2}yield self.{f.name}, _fld_{f.name}, None\n"

        for f, type_info in sorted(child_fields.items(), key=lambda x: x[0].name):
            # Store the field object in the closure
            local_vars[f"_fld_{f.name}"] = f

            _build_body(f, type_info)

        # Now the unsorted version
        body += "else:\n"

        for f, type_info in child_fields.items():
            _build_body(f, type_info)

    _gen_func(clz, fname, ret_type, body, local_vars, extra_args="sort_keys: bool = False")


def _gen_iter_child_fields_func(
    clz: type[ASTNode], child_fields: Mapping[Field, FieldTypeInfo]
) -> None:
    """Generate a specialized `iter_child_fields` function."""
    fname = "iter_child_fields"

    # Hols closure vars with Field objects
    local_vars = {}

    # The code of the actual function
    ret_type = "Iterable[tuple[ASTNode | tuple[ASTNode] | None, Field]]"

    # If a node doesn't have child fields, the above loop will not produce any code
    # but we need to make the function a generator, so we yield nothing
    if len(child_fields) == 0:
        body = "yield from ()\n"
    else:
        # Iterate over all child fields and create sorted & unsroted versions

        # First the sorted version
        body = "if sort_keys:\n"

        def _build_body(f: Field) -> None:
            nonlocal body

            # Simply yield value and field
            body += f"{_IND}yield self.{f.name}, _fld_{f.name}\n"

        for f in sorted(child_fields.keys(), key=lambda f: f.name):
            # Store the field object in the closure
            local_vars[f"_fld_{f.name}"] = f

            _build_body(f)

        # Now the unsorted version
        body += "else:\n"

        for f in child_fields.keys():
            _build_body(f)

    _gen_func(clz, fname, ret_type, body, local_vars, extra_args="sort_keys: bool = False")


def _gen_get_properties_func(clz: type[ASTNode], props: Mapping[Field, FieldTypeInfo]) -> None:
    """Generate a specialized `get_properties` function."""
    fname = "get_properties"

    # Hols closure vars with Field objects
    local_vars = {}

    # The code of the actual function
    ret_type = "Iterable[tuple[Any, Field]]"

    # Iterate over all child fields and create sorted & unsroted versions

    # First the sorted version
    body = "if sort_keys:\n"

    def _build_body(f: Field) -> None:
        nonlocal body

        if f.name == "id" or f.name == "content_id":
            body += f"{_IND}if not skip_id:\n"
            body += f"{_IND*2}yield self.{f.name}, _fld_{f.name}\n"
            return

        if f.name == "origin":
            body += f"{_IND}if not skip_origin:\n"
            body += f"{_IND*2}yield self.{f.name}, _fld_{f.name}\n"
            return

        if f.name == "original_id":
            body += f"{_IND}if not skip_original_id:\n"
            body += f"{_IND*2}yield self.{f.name}, _fld_{f.name}\n"
            return

        if f.name == "id_collision_with":
            body += f"{_IND}if not skip_id_collision_with:\n"
            body += f"{_IND*2}yield self.{f.name}, _fld_{f.name}\n"
            return

        if f.name.startswith("_"):
            body += f"{_IND}if not skip_hidden:\n"
            body += f"{_IND*2}yield self.{f.name}, _fld_{f.name}\n"
            return

        if not f.compare:
            body += f"{_IND}if not skip_non_compare:\n"
            body += f"{_IND*2}yield self.{f.name}, _fld_{f.name}\n"
            return

        if not f.init:
            body += f"{_IND}if not skip_non_init:\n"
            body += f"{_IND*2}yield self.{f.name}, _fld_{f.name}\n"
            return

        body += f"{_IND}yield self.{f.name}, _fld_{f.name}\n"

    for f in sorted(props.keys(), key=lambda f: f.name):
        # Store the field object in the closure
        local_vars[f"_fld_{f.name}"] = f

        _build_body(f)

    # Now the unsorted version
    body += "else:\n"

    for f in props.keys():
        _build_body(f)

    extra_args = """
        skip_id: bool = True,
        skip_origin: bool = True,
        skip_original_id: bool = True,
        skip_id_collision_with: bool = True,
        skip_hidden: bool = True,
        skip_non_compare: bool = False,
        skip_non_init: bool = False,
        *,
        sort_keys: bool = False,
    """.strip().replace("\n", "")

    _gen_func(clz, fname, ret_type, body, local_vars, extra_args=extra_args)


def gen_and_yield_iter_child_fields(
    self: ASTNode, *, sort_keys: bool = False
) -> Iterable[tuple[ASTNode | tuple[ASTNode] | None, Field]]:
    """A special function that will generate a specialized method for the class of the given node on
    the fly and also yield from it to make sure the first call to the method also works."""

    # Dynamicaly generate a specialized function for this class
    _gen_iter_child_fields_func(self.__class__, get_cls_child_fields(self.__class__))

    # At this point this will call a specialized function
    yield from self.iter_child_fields(sort_keys=sort_keys)


def gen_and_yield_get_properties(
    self: ASTNode,
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
    """A special function that will generate a specialized method for the class of the given node on
    the fly and also yield from it to make sure the first call to the method also works."""
    # Dynamicaly generate a specialized function for this class
    _gen_get_properties_func(self.__class__, get_cls_props(self.__class__))

    # At this point this will call a specialized function
    yield from self.get_properties(
        skip_id,
        skip_origin,
        skip_original_id,
        skip_id_collision_with,
        skip_hidden,
        skip_non_compare,
        skip_non_init,
        sort_keys=sort_keys,
    )


def gen_and_yield_get_child_nodes(self: ASTNode, *, sort_keys: bool = False) -> Iterable[ASTNode]:
    """A special function that will generate a specialized method for the class of the given node on
    the fly and also yield from it to make sure the first call to the method also works."""

    # Dynamicaly generate a specialized function for this class
    _gen_get_child_nodes_func(self.__class__, get_cls_child_fields(self.__class__))

    # At this point this will call a specialized function
    yield from self.get_child_nodes(sort_keys=sort_keys)


def gen_and_yield_get_child_nodes_with_field(
    self: ASTNode, *, sort_keys: bool = False
) -> Iterable[tuple[ASTNode, Field, int | None]]:
    """A special function that will generate a specialized method for the class of the given node on
    the fly and also yield from it to make sure the first call to the method also works."""

    # Dynamicaly generate a specialized function for this class
    _gen_get_child_nodes_with_field_func(self.__class__, get_cls_child_fields(self.__class__))

    # At this point this will call a specialized function
    yield from self.get_child_nodes_with_field(sort_keys=sort_keys)
