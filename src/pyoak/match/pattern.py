from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Sequence,
)

from pyoak import config

from ..node import ASTNode
from .error import ASTXpathOrPatternDefinitionError

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)

_Vars = Mapping[str, Any]
_MatchRes = tuple[bool, _Vars]


@dataclass(frozen=True, slots=True)
class BaseMatcher(ABC):
    """Base class for all matchers.

    Use `from_pattern` to create a matcher from a pattern definition.

    """

    name: str | None = field(default=None, kw_only=True)

    @abstractmethod
    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        raise NotImplementedError

    def match(self, value: Any, ctx: _Vars | None = None) -> _MatchRes:
        if ctx is None:
            ctx = {}

        ok, new_vars = self._match(value, ctx)

        if not ok:
            return (False, {})

        if self.name is None:
            return (True, new_vars)

        return (True, {self.name: value, **new_vars})

    @staticmethod
    @lru_cache(maxsize=None)
    def from_pattern(pattern_def: str, types: Mapping[str, type[Any]] | None = None) -> BaseMatcher:
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

        return matcher


@dataclass(frozen=True, slots=True)
class AnyMatcher(BaseMatcher):
    """Matcher that matches any value (`*` in pattern DSL)."""

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        return (True, {})


@dataclass(frozen=True, slots=True)
class ValueMatcher(BaseMatcher):
    """Matcher that matches against a constant value.

    In pattern DSL, it is used to match None & empty sequences only.

    """

    value: Any

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        if isinstance(self.value, ASTNode):
            # Using content based equality for ASTNodes
            return (self.value.is_equal(value), {})

        if value == self.value:
            return (True, {})

        return (False, {})


@dataclass(frozen=True, slots=True)
class RegexMatcher(BaseMatcher):
    """Matcher that matches a value against a regex.

    In pattern DSL, this is represented as a double quoted escaped string.

    """

    _re_str: str
    pattern: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern", re.compile(self._re_str))

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        if self.pattern.match(str(value)) is not None:
            return (True, {})

        return (False, {})


@dataclass(frozen=True, slots=True)
class VarMatcher(BaseMatcher):
    """Matcher that matches a value against a previously captured value.

    In pattern DSL, this is represented as `$var_name`.

    """

    var_name: str

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        if self.var_name not in ctx:
            raise ASTXpathOrPatternDefinitionError(
                f"Pattern uses match variable {self.var_name} before it was captured"
            )

        var_value = ctx[self.var_name]

        if isinstance(var_value, ASTNode):
            # Using content based equality for ASTNodes
            return (var_value.is_equal(value), {})

        # Using value based equality for other types
        return (var_value == value, {})


@dataclass(frozen=True, slots=True)
class SequenceMatcher(BaseMatcher):
    """Matcher that matches a sequence of values against a sequence of matchers.

    In pattern DSL, this is represented as `[...]`, a comma separated list of matchers
    in square brackets with an optional any (`*`) tail matcher.

    """

    matchers: tuple[BaseMatcher, ...]
    tail_matcher: AnyMatcher | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if len(self.matchers) == 0:
            raise RuntimeError(
                "SequenceMatcher must have at least one matcher."
                " Use ValueMatcher with empty tuple instead."
            )

        if isinstance(self.matchers[-1], AnyMatcher):
            object.__setattr__(self, "tail_matcher", self.matchers[-1])
            object.__setattr__(self, "matchers", self.matchers[:-1])

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        if not isinstance(value, Sequence):
            return (False, {})

        any_tail = self.tail_matcher is not None

        # If we have any tail, we can match sequence of any length
        # but if we don't have any tail, we can exit early if the lengths
        # don't match
        if (not any_tail and len(value) != len(self.matchers)) or (
            any_tail and len(value) < len(self.matchers)
        ):
            return (False, {})

        # Create local mutable context
        local_ctx = dict(ctx)
        ret_vars: dict[str, Any] = {}

        # Start with non-tail matchers
        for matcher, val in zip(self.matchers, value, strict=False):
            ok, new_vars = matcher.match(val, local_ctx)
            if not ok:
                return (False, {})

            local_ctx.update(new_vars)
            ret_vars.update(new_vars)

        # If we have any tail, match it against the rest of the sequence
        if self.tail_matcher:
            # Any always matches so we only doing this for possible captures
            _, new_vars = self.tail_matcher.match(value[len(self.matchers) :], local_ctx)

            ret_vars.update(new_vars)

        return (True, ret_vars)


@dataclass(frozen=True, slots=True)
class AlternativeMatcher(BaseMatcher):
    """Matcher that matches a value against a set of alternatives in order. Only the first matching
    alternative is used, and thus vlues are captured.

    In pattern DSL, this is represented as `matcher | matcher | ...`.

    """

    matchers: tuple[BaseMatcher, ...]

    def __post_init__(self) -> None:
        if len(self.matchers) == 0:
            raise RuntimeError("AlternativeMatcher must have at least one matcher.")

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        for matcher in self.matchers:
            ok, new_vars = matcher.match(value, ctx)
            if ok:
                return (True, new_vars)

        return (False, {})


@dataclass(frozen=True, slots=True)
class NodeMatcher(BaseMatcher):
    types: tuple[type[ASTNode], ...]
    # Using tuple because stdlib doesn't have immutable mapping
    content: tuple[tuple[str, BaseMatcher], ...]

    def __post_init__(self) -> None:
        # Remove duplicates
        object.__setattr__(self, "types", tuple(set(self.types)))

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        if not isinstance(value, self.types):
            return (False, {})

        # Create local mutable context
        local_ctx = dict(ctx)
        ret_vars: dict[str, Any] = {}

        for fname, submatcher in self.content:
            if not hasattr(value, fname):
                return (False, {})

            ok, new_vars = submatcher.match(getattr(value, fname), local_ctx)
            if not ok:
                return (False, {})

            local_ctx.update(new_vars)
            ret_vars.update(new_vars)

        return (True, ret_vars)


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
