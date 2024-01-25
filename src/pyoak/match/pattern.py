from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Sequence,
)

from pyoak import config
from pyoak.match.helpers import get_dataclass_field_names

from ..node import ASTNode
from .error import ASTXpathOrPatternDefinitionError

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)

_Vars = Mapping[str, Any]
_MatchRes = tuple[bool, _Vars]
_SeqMatchRes = tuple[bool, _Vars, int]

_PATTERN_CACHE: dict[int, BaseMatcher] = {}


def _make_key(pattern_def: str, types: Mapping[str, type[Any]]) -> int:
    """Create a key for the LRU cache based on pattern definition and types."""
    return hash((pattern_def, tuple(types.items())))


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


@dataclass(frozen=True, slots=True)
class BaseMatcher(ABC):
    """Base class for all matchers.

    Use `from_pattern` to create a matcher from a pattern definition.

    """

    name: str | None = field(default=None, kw_only=True)
    """Name of the matcher.

    If not None, the matcher will capture the matched value under this name in the context (which is
    eventually returned as the match dictionary).

    """

    append_to_match: bool = field(default=False, kw_only=True)
    """If True, the matcher will append the matched value to the list under the name in the context
    instead of overwriting it."""

    @abstractmethod
    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        """Internal API to be implemented by concrete matchers.

        Match a value against the pattern and returns whether it had match as well as a dictionary
        of all captured values by the submatchers. There is no need to add the captured value of
        this matcher to the dictionary as it will be added by the caller.

        """
        raise NotImplementedError

    def match(self, value: Any, ctx: _Vars | None = None) -> _MatchRes:
        """Match a value against the pattern."""

        if ctx is None:
            ctx = {}

        ok, new_vars = self._match(value, ctx)

        if not ok:
            return (False, {})

        if self.name is None:
            return (True, new_vars)

        if self.append_to_match:
            vals_in_ctx = ctx.get(self.name, ())
            vals_in_new_vars = new_vars.get(self.name, ())

            if not isinstance(vals_in_ctx, Sequence) or not isinstance(vals_in_new_vars, Sequence):
                raise ValueError(
                    f"Matcher {self.name} is set to append to a match but the captured value "
                    "so far is not a sequence. This should have been caught during pattern "
                    "definition parsing. Please report a bug!"
                )

            return (True, {**new_vars, self.name: (*vals_in_ctx, *vals_in_new_vars, value)})

        return (True, {**new_vars, self.name: value})

    from_pattern = staticmethod(from_pattern)


class WildcardMatcher(BaseMatcher, ABC):
    def _match_seq(self, value: Sequence[Any], ctx: _Vars) -> _SeqMatchRes:
        """Internal API to be implemented by concrete matchers.

        Unlike regular _match, matches a sequence of values by consuming
        as many values as needed from the start of the sequence.

        Returns:
            A tuple of a match result, the number of items "consumed".

        """
        raise NotImplementedError

    def match_seq(self, value: Sequence[Any], ctx: _Vars | None = None) -> _SeqMatchRes:
        """Match a sequence of values against the pattern and return the remaining values."""
        if not isinstance(value, Sequence):
            raise ValueError("match_seq can only be called on sequences")

        if ctx is None:
            ctx = {}

        ok, new_vars, consumed = self._match_seq(value, ctx)

        if not ok:
            return (False, {}, 0)

        store_value = value[:consumed]

        if self.name is None:
            return (True, new_vars, consumed)

        if self.append_to_match:
            vals_in_ctx = ctx.get(self.name, ())
            vals_in_new_vars = new_vars.get(self.name, ())

            if not isinstance(vals_in_ctx, Sequence) or not isinstance(vals_in_new_vars, Sequence):
                raise ValueError(
                    f"Matcher {self.name} is set to append to a match but the captured value "
                    "so far is not a sequence. This should have been caught during pattern "
                    "definition parsing. Please report a bug!"
                )

            # In multi capture we extend the list of values
            return (
                True,
                {**new_vars, self.name: (*vals_in_ctx, *vals_in_new_vars, *store_value)},
                consumed,
            )

        # In single capture we overwrite the value, so name will map to a sequence
        return (True, {**new_vars, self.name: store_value}, consumed)


@dataclass(frozen=True, slots=True)
class AnyMatcher(WildcardMatcher):
    """Matcher that matches any value (`*` in pattern DSL)."""

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        return (True, {})

    def _match_seq(self, value: Sequence[Any], ctx: _Vars) -> _SeqMatchRes:
        # Consumes the entire sequence
        return True, {}, len(value)


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

    def __post_init__(self) -> None:
        if len(self.matchers) == 0:
            raise RuntimeError(
                "SequenceMatcher must have at least one matcher."
                " Use ValueMatcher with empty tuple instead."
            )

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        if not isinstance(value, Sequence):
            return (False, {})

        wildcards = any(isinstance(m, WildcardMatcher) for m in self.matchers)

        # Short circuit by comparing lengths
        # If we have no wildcards, we can exit early if the lengths
        # don't match
        if not wildcards and len(value) != len(self.matchers):
            return (False, {})

        # Create local mutable context
        local_ctx = dict(ctx)
        ret_vars: dict[str, Any] = {}

        tail = value
        # Iterate over all matchers
        for matcher in self.matchers:
            if isinstance(matcher, WildcardMatcher):
                ok, new_vars, shift = matcher.match_seq(tail, local_ctx)
            elif tail:
                ok, new_vars = matcher.match(tail[0], local_ctx)
                shift = 1
            else:
                # Still have matchers but no values left
                return (False, {})

            if not ok:
                return (False, {})

            tail = tail[shift:]

            local_ctx.update(new_vars)
            ret_vars.update(new_vars)

        return (True, ret_vars)


@dataclass(frozen=True, slots=True)
class QualifierMatcher(WildcardMatcher):
    """Matcher that implements regex-like wildcard qualifiers to a value matcher inside of a
    sequence matcher.

    Specifically:
        - `*` matches 0 or more times (min=0, max=None)
        - `+` matches 1 or more times (min=1, max=None)
        - `?` matches 0 or 1 times (min=0, max=1)
        - `{n}` matches exactly n times (min=n, max=n)
        - `{m,n}` matches between m and n times, inclusive (min=m, max=n)

    Matching is greedy and doesn't backtrack. I.e. the following pattern
    can never match: `(* @foo=[(*)* (OtherNode)])` because the first wildcard
    will consume the entire sequence.

    In pattern DSL, this is represented as a matcher followed by one of the
    above quantifiers.

    """

    matcher: BaseMatcher
    min: int
    max: int | None

    def __post_init__(self) -> None:
        if self.max is not None and self.min > self.max:
            raise ValueError("min must be <= max")

        if self.min < 0:
            raise ValueError("min must be >= 0")

        if self.max is not None and self.max < 0:
            raise ValueError("max must be >= 0")

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        raise NotImplementedError

    def _match_seq(self, value: Sequence[Any], ctx: _Vars) -> _SeqMatchRes:
        # Create local mutable context
        local_ctx = dict(ctx)
        ret_vars: dict[str, Any] = {}

        matched = 0

        for val in value:
            ok, new_vars = self.matcher.match(val, local_ctx)
            if not ok:
                break

            local_ctx.update(new_vars)
            ret_vars.update(new_vars)
            matched += 1

            if self.max is not None and matched == self.max:
                break

        if matched < self.min:
            return False, {}, 0

        return True, ret_vars, matched


@dataclass(frozen=True, slots=True)
class AlternativeMatcher(BaseMatcher):
    """Matcher that matches a value against a set of alternatives in order. Only the first matching
    alternative is used, and thus vlues are captured.

    In pattern DSL, this is represented as `matcher | matcher | ...`.

    """

    matchers: tuple[NodeMatcher | PatternRefMatcher, ...]

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
    """Tuple of ASTNode types to match against.

    If empty, matches any class.

    """

    # Using tuple because stdlib doesn't have immutable mapping
    content: tuple[tuple[str, BaseMatcher], ...]
    """Tuple of tuples of field names and matchers to match against."""

    def __post_init__(self) -> None:
        # Remove duplicates
        object.__setattr__(self, "types", tuple(set(self.types)))

        # Validate types have necessary attributes
        err_types: dict[type[ASTNode], set[str]] = {}

        expected_fields = {fname for fname, _ in self.content}

        for type_ in self.types:
            missing = expected_fields - get_dataclass_field_names(type_)

            if missing:
                err_types[type_] = missing

        if err_types:
            pretty_list = "\n  - ".join(
                f"{t.__name__}: {', '.join(sorted(m))}"
                for t, m in sorted(err_types.items(), key=lambda x: x[0].__name__)
            )
            raise ASTXpathOrPatternDefinitionError(
                f"Pattern uses match types with missing attributes:\n  - {pretty_list}"
            )

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        if self.types and not isinstance(value, self.types):
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


@dataclass(frozen=True, slots=True)
class PatternRefMatcher(BaseMatcher):
    """Matcher that matches a value against a previously defined pattern.

    In pattern DSL, this is represented as `#pattern_name`.

    """

    pattern_name: str

    # We intentionally use mutable mapping here to allow
    # for forward/circular references. Existance of the pattern
    # in a map is checked during matching (but also during parsing).
    # At init time, we don't know all the patterns yet.
    pattern_alias_map: dict[str, AlternativeMatcher | NodeMatcher]

    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        if self.pattern_name not in self.pattern_alias_map:
            raise KeyError(
                f"Unknown pattern reference <{self.pattern_name}>. Did you construct the "
                "matcher by hand? Make sure to update the pattern alias map!"
            )

        return self.pattern_alias_map[self.pattern_name].match(value, ctx)


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
