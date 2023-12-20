from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataclasses import Field as DataClassField

    # to make mypy happy
    Field = DataClassField[Any]


def is_skip_field(
    field: "Field",
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
