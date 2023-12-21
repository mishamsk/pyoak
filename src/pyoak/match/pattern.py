from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Sequence,
)

from ..node import ASTNode
from .error import ASTXpathOrPatternDefinitionError

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)

_Vars = Mapping[str, Any]
_MatchRes = tuple[bool, _Vars]


@dataclass(frozen=True, slots=True)
class BaseMatcher(ABC):
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


@dataclass(frozen=True, slots=True)
class AnyMatcher(BaseMatcher):
    def _match(self, value: Any, ctx: _Vars) -> _MatchRes:
        return (True, {})


@dataclass(frozen=True, slots=True)
class ValueMatcher(BaseMatcher):
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

        # For future - operate on a local context
        # in case / when we'll have backtracking
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


_MATCHER_CACHE: dict[str, NodeMatcher] = {}


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

        # For future - operate on a local context
        # in case / when we'll have backtracking
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

    @classmethod
    def from_pattern(
        cls, pattern_def: str, types: Mapping[str, type[Any]] | None = None
    ) -> NodeMatcher:
        """Create a NodeMatcher from a pattern definition.

        Args:
            pattern_def: The pattern definition to parse.
            types: An optional mapping of AST class names to their types. If not provided,
                the default mapping from `pyoak.serialize` is used.

        Returns:
            A NodeMatcher instance.

        Raises:
            ASTXpathOrPatternDefinitionError: Raised if the pattern definition is incorrect

        """
        if pattern_def in _MATCHER_CACHE:
            return _MATCHER_CACHE[pattern_def]

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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Unexpected error during pattern definition grammar generation: {e}")

            raise ASTXpathOrPatternDefinitionError(
                "Failed to parse a tree pattern due to internal error. Please report it!"
            ) from e

        _MATCHER_CACHE[pattern_def] = matcher
        return matcher


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


class MultiPatternMatcher:
    def __init__(
        self, pattern_defs: Sequence[tuple[str, str]], types: Mapping[str, type[Any]] | None = None
    ) -> None:
        """A tree pattern matcher.

        Args:
            pattern_defs (Sequence[tuple[str, str]]): A sequence of tuples of pattern name
                and pattern definition. Pattern names must be unqiue and are used
                identify the pattern in the match result.
            types: An optional mapping of AST class names to their types. If not provided,
                the default mapping from `pyoak.serialize` is used.

        Raises:
            ASTXpathOrPatternDefinitionError: Raised if the pattern definition is incorrect

        """

        if len({pd[0] for pd in pattern_defs}) != len(pattern_defs):
            raise ASTXpathOrPatternDefinitionError("Pattern names must be unique")

        self._name_to_matcher: dict[str, NodeMatcher] = {}

        incorrect_patterns: list[tuple[str, str]] = []
        for pattern_name, pattern_def in pattern_defs:
            try:
                matcher = NodeMatcher.from_pattern(pattern_def, types)
            except Exception as e:
                incorrect_patterns.append((pattern_name, str(e)))

            self._name_to_matcher[pattern_name] = matcher

        if incorrect_patterns:
            raise ASTXpathOrPatternDefinitionError(
                "Incorrect pattern definitions:\n"
                + "\n".join(
                    [
                        f"Pattern '{pattern_name}': {pattern_def}"
                        for pattern_name, pattern_def in incorrect_patterns
                    ]
                )
            )

    def match(
        self, node: ASTNode, rules: Iterable[str] | None = None
    ) -> tuple[str, Mapping[str, Any]] | None:
        """Match a node against the pattern definitions.

        Args:
            node: The node to match against the pattern definitions.
            rules: The rules to match against (in order). If None, all rules will be checked.

        Returns:
            A tuple of the matched rule name and a dictionary of the matched values.
            If no rule matches, None is returned.

        """
        if rules is None:
            rules = self._name_to_matcher.keys()

        for rule in rules:
            ok, capture_dict = self._name_to_matcher[rule].match(node)

            if ok:
                return rule, capture_dict

        return None
