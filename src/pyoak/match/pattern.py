from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Mapping,
    Sequence,
    cast,
)

from lark import Lark, Tree, UnexpectedInput
from lark.visitors import Interpreter

from ..node import ASTNode
from .error import ASTPatternDefinitionError
from .grammar import PATTERN_DEF_GRAMMAR
from .helpers import check_and_get_ast_node_type

if TYPE_CHECKING:
    pass

pattern_def_parser = Lark(grammar=PATTERN_DEF_GRAMMAR, start="tree", parser="lalr")


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
    _instance: ClassVar[AnyMatcher | None] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> AnyMatcher:
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

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
            raise ASTPatternDefinitionError(
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
            any_tail and len(value) < len(self.matchers) - 1
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
    def from_pattern(cls, pattern_def: str) -> tuple[NodeMatcher | None, str]:
        """Create a NodeMatcher from a pattern definition."""
        if pattern_def in _MATCHER_CACHE:
            return _MATCHER_CACHE[pattern_def], "Cached matcher"

        try:
            parsed_pattern_defs = pattern_def_parser.parse(pattern_def)
        except UnexpectedInput as e:
            return (
                None,
                f"Incorrect pattern definition. Context:\n{e.get_context(pattern_def)}",
            )
        except Exception:
            return None, "Incorrect pattern definition. Unexpected error"

        try:
            matcher = cast(
                NodeMatcher, PatternDefInterpreter().visit(cast(Tree[str], parsed_pattern_defs))
            )
        except ASTPatternDefinitionError as e:
            return None, e.message
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Unexpected error during pattern definition grammar generation: {e}")
            return None, "Incorrect pattern definition. Unexpected error"

        _MATCHER_CACHE[pattern_def] = matcher
        return matcher, "Valid pattern definition"


class PatternDefInterpreter(Interpreter[str, BaseMatcher]):
    def __init__(self) -> None:
        super().__init__()

        self._captures_seen: set[str] = set()

    def _check_unique_and_get_capture(self, child: str | Tree[str]) -> str | None:
        if not isinstance(child, Tree) or child.data != "capture":
            return None

        name = str(child.children[0])

        if name in self._captures_seen:
            raise ASTPatternDefinitionError(f"Capture name <{name}> used more than once")

        self._captures_seen.add(name)

        return name

    def reset(self) -> None:
        self._captures_seen = set()

    def tree(self, tree: Tree[str]) -> NodeMatcher:
        # First parse class_spec which will have 1+ ASTNode types or ANY
        assert isinstance(tree.children[0], Tree)
        assert tree.children[0].data == "class_spec"

        match_types: list[type[ASTNode]] = []
        class_names = cast(list[str], tree.children[0].children)

        for class_name in class_names:
            if class_name == "*":
                match_types = [ASTNode]
                break

            type_, msg = check_and_get_ast_node_type(class_name)
            if type_ is None:
                raise ASTPatternDefinitionError(msg)

            match_types.append(type_)

        # Then parse field_spec which will have 1+ field names and matchers
        content: list[tuple[str, BaseMatcher]] = []

        for child in tree.children[1:]:
            if isinstance(child, Tree):
                assert child.data == "field_spec"
                content.append((str(child.children[0]), self.visit(child)))
            else:
                raise RuntimeError(f"Unexpected child in tree rule: {child}")

        return NodeMatcher(types=tuple(match_types), content=tuple(content))

    def field_spec(self, tree: Tree[str]) -> BaseMatcher:
        if len(tree.children) == 1:
            # Any value without capture
            return AnyMatcher()

        # Second may be capture or value
        name = self._check_unique_and_get_capture(tree.children[1])

        if name is not None:
            # Any value with capture
            return AnyMatcher(name=name)

        val = tree.children[1]
        assert isinstance(val, Tree)

        matcher = self.visit(val)

        if len(tree.children) > 2:
            # Has capture
            name = self._check_unique_and_get_capture(tree.children[2])

            if name is None:
                raise RuntimeError("Unexpected child in field_spec rule")

            return replace(matcher, name=name)

        return matcher

    def sequence(self, tree: Tree[str]) -> SequenceMatcher | ValueMatcher:
        matchers: list[BaseMatcher] = []
        last_matcher: BaseMatcher | None = None

        for child in tree.children:
            if isinstance(child, Tree):
                # Value or capture
                capture_name = self._check_unique_and_get_capture(child)

                if capture_name is not None:
                    if last_matcher is None:
                        raise RuntimeError("Unexpected capture in sequence rule.")

                    last_matcher = replace(last_matcher, name=capture_name)
                    continue

                if last_matcher is not None:
                    matchers.append(last_matcher)

                last_matcher = self.visit(child)
            else:
                # Must be ANY
                assert child == "*"

                if last_matcher is not None:
                    matchers.append(last_matcher)

                last_matcher = AnyMatcher()

        if last_matcher is not None:
            matchers.append(last_matcher)

        if len(matchers) == 0:
            return ValueMatcher(value=())

        return SequenceMatcher(matchers=tuple(matchers))

    def value(self, tree: Tree[str]) -> NodeMatcher | ValueMatcher | RegexMatcher | VarMatcher:
        assert len(tree.children) == 1
        val = tree.children[0]

        if isinstance(val, Tree):
            if val.data == "tree":
                return self.tree(val)

            if val.data == "var":
                var_name = str(val.children[0])

                if var_name not in self._captures_seen:
                    raise ASTPatternDefinitionError(
                        f"Pattern uses match variable {var_name} before it was captured"
                    )

                return VarMatcher(var_name=var_name)

            raise RuntimeError(f"Unexpected child Tree in value rule: {val}")

        if val == "None":
            return ValueMatcher(value=None)

        # Must be a string
        return RegexMatcher(_re_str=str(val[1:-1]))


def validate_pattern(pattern_def: str) -> tuple[bool, str]:
    """Validate a single pattern definition in a form of `pattern`.

    Args:
        pattern_def: The pattern definition to validate.

    Returns:
        A tuple of a boolean indicating whether the pattern definition is valid and a string
        containing the error message if the pattern definition is invalid.
    """
    try:
        parsed_pattern_defs = pattern_def_parser.parse(pattern_def)
    except UnexpectedInput as e:
        return (
            False,
            f"Incorrect pattern definition. Context:\n{e.get_context(pattern_def)}",
        )
    except Exception:
        return False, "Incorrect pattern definition. Unexpected error"

    try:
        _ = PatternDefInterpreter().visit(cast(Tree[str], parsed_pattern_defs))
    except ASTPatternDefinitionError as e:
        return False, e.message
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Unexpected error during pattern definition grammar generation: {e}")
        return False, "Incorrect pattern definition. Unexpected error"

    return True, "Valid pattern definition"


class MultiPatternMatcher:
    def __init__(self, pattern_defs: Sequence[tuple[str, str]]) -> None:
        """A tree pattern matcher.

        Args:
            pattern_defs (Sequence[tuple[str, str]]): A sequence of tuples of pattern name
                and pattern definition. Pattern names must be unqiue and are used
                identify the pattern in the match result.

        Raises:
            ASTPatternDefinitionError: Raised if the pattern definition is incorrect
        """

        if len({pd[0] for pd in pattern_defs}) != len(pattern_defs):
            raise ASTPatternDefinitionError("Pattern names must be unique")

        self._name_to_matcher: dict[str, NodeMatcher] = {}

        incorrect_patterns: list[tuple[str, str]] = []
        for pattern_name, pattern_def in pattern_defs:
            matcher, msg = NodeMatcher.from_pattern(pattern_def)

            if matcher is None:
                incorrect_patterns.append((pattern_name, msg))
                continue

            self._name_to_matcher[pattern_name] = matcher

        if incorrect_patterns:
            raise ASTPatternDefinitionError(
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
