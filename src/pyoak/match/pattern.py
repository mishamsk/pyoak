import logging
from typing import TYPE_CHECKING, Any, Mapping

from pyoak import config

from .error import ASTXpathOrPatternDefinitionError

if TYPE_CHECKING:
    from .matcher import BaseMatcher

logger = logging.getLogger(__name__)


def validate_pattern(
    pattern_def: str, types: Mapping[str, type[Any]] | None = None
) -> tuple[bool, str]:
    """Validate a single pattern definition in a form of `pattern`.

    Args:
        pattern_def: The pattern definition to validate.
        types: An optional mapping of AST class names to their types. If not provided,
            the default mapping from `pyoak.serialize` is used.

    Returns:
        A tuple of a boolean indicating whether the pattern definition is valid and a string
        containing the error message if the pattern definition is invalid.

    """
    if types is None:
        # Only import if needed
        from ..serialize import TYPES

        types = TYPES

    # Import here to avoid circular imports
    from .parser import Parser

    try:
        _ = Parser(types).parse_pattern(pattern_def)
    except ASTXpathOrPatternDefinitionError as e:
        return (False, str(e))
    except Exception:
        return False, "Incorrect pattern definition. Unexpected error"

    return True, "Valid pattern definition"


_PATTERN_CACHE: dict[int, "BaseMatcher"] = {}


def _make_key(pattern_def: str, types: Mapping[str, type[Any]]) -> int:
    """Create a key for the LRU cache based on pattern definition and types."""
    return hash((pattern_def, tuple(types.items())))


def from_pattern(pattern_def: str, types: Mapping[str, type[Any]] | None = None) -> "BaseMatcher":
    """Create a Matcher from a pattern definition.

    Args:
        pattern_def: The pattern definition to parse.
        types: An optional mapping of AST class names to their types. If not provided,
            the default mapping from `pyoak.serialize` is used.

    Returns:
        A BaseMatcher instance.

    Raises:
        ASTXpathOrPatternDefinitionError: Raised if the pattern definition is incorrect

    """
    if types is None:
        # Only import if needed
        from ..serialize import TYPES

        types = TYPES

    # Check cache
    key = _make_key(pattern_def, types)

    matcher = _PATTERN_CACHE.get(key, None)

    if matcher is not None:
        return matcher

    # Import here to avoid circular imports
    from .parser import Parser

    try:
        matcher = Parser(types).parse_pattern(pattern_def)
    except ASTXpathOrPatternDefinitionError:
        raise
    except Exception as e:
        if config.TRACE_LOGGING:
            logger.debug(f"Unexpected error during pattern definition grammar generation: {e}")

        raise ASTXpathOrPatternDefinitionError(
            "Failed to parse a tree pattern due to internal error. Please report it!"
        ) from e

    _PATTERN_CACHE[key] = matcher

    return matcher
