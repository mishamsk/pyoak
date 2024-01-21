from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
from typing import Any, Mapping, Type

from ..node import ASTNode


def check_and_get_ast_node_type(
    class_name: str, types: Mapping[str, type[Any]]
) -> tuple[Type[ASTNode] | None, str]:
    if class_name not in types:
        return None, f"Unknown AST type: <{class_name}>"

    type_ = types[class_name]

    if not issubclass(type_, ASTNode):
        return None, f"<{class_name}> is not an AST type"
    else:
        return type_, ""


@lru_cache(maxsize=None)
def get_dataclass_field_names(type_: type[Any]) -> set[str]:
    return {field.name for field in fields(type_)}


def point_at_index(input_string: str, index: int, length: int = 1) -> str:
    """Add a pointer to the string at the given index.

    Args:
        input_string (str): string to add a pointer to
        index (int): _description_
        length (int, optional): _description_. Defaults to 1.

    Raises:
        ValueError: _description_

    Returns:
        str: _description_

    """
    if not input_string:
        return input_string

    if index >= len(input_string):
        raise ValueError("Index is out of range")

    # Get the line where the index is located
    lines = input_string.splitlines(keepends=True)

    # min is needed to support a single line string with a trailing newline
    target_line_index = min(len(lines), input_string.count("\n", 0, index))
    line = lines[target_line_index]

    # Remember if the line has a trailing newline
    # but remove it for now. We should not point to the newline
    has_trailing_newline = line.endswith("\n")
    if has_trailing_newline:
        line = line[:-1]

    # Shift the index to the start of the line
    target_line_start_index = sum(len(pl) for pl in lines[:target_line_index])
    index -= target_line_start_index

    # Calculate the pointer
    arrow = "-" * max(0, min(3, index)) + "^" * min(length, max(0, len(line) - index))

    # add ws before
    arrow = " " * max(0, (index - 3)) + arrow

    # add ws after
    arrow += " " * max(0, (len(line) - len(arrow)))

    # Add the pointer to the line
    updated_line = f"{line}\n{arrow}"

    if has_trailing_newline:
        # Restore the trailing newline
        updated_line += "\n"

    return "".join([*lines[:target_line_index], updated_line, *lines[target_line_index + 1 :]])
