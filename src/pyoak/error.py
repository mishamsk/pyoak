from dataclasses import Field as DataClassField
from typing import Any, Sequence

# to make mypy happy
Field = DataClassField[Any]


class ASTNodeError(Exception):
    """Base class for all ASTNode errors."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message})"


class ASTRefCollisionError(ASTNodeError):
    """Raised when a ref value collision is detected."""

    def __init__(self, ref: str) -> None:
        self.ref = ref
        message = f"Ref value collision for {ref}"
        super().__init__(message, ref)


class InvalidFieldAnnotations(ASTNodeError):
    """Raised when an ASTNode has fields with mixed ASTNode subclasses and
    regular types or unsupported child fileds types (e.g. mutable collections
    of ASTNode's)."""

    def __init__(self, invalid_annotations: Sequence[tuple[str, str, type]]) -> None:
        self.invalid_annotations = invalid_annotations
        message = f"The following field annotations are not valid: {', '.join(f'{name} ({type_}): {reason}' for name, reason, type_ in invalid_annotations)}"
        super().__init__(message, invalid_annotations)


class InvalidTypes(ASTNodeError):
    """Raised at runtime when a field is assigned an invalid type and
    config.RUNTIME_TYPE_CHECK is True."""

    def __init__(self, invalid_fields: Sequence[Field]) -> None:
        self.invalid_fields = invalid_fields
        message = (
            "The values for following fields have incorrect types: "
            f"{', '.join(f'{f.name} of type<{f.type}>' for f in invalid_fields)}"
        )
        super().__init__(message, invalid_fields)
