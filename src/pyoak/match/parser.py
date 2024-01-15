"""This module implements a simple recursive decent parser.

It is used to parse the xpath-like & pattern matching syntax used by pyoak.match.

It was originally built with Lark, hece the formal grammar below is using Lark's
grammar syntax. But it is now a fully bespoke implementation.

xpath: element* self

element: "/" field_spec? index_spec? class_spec?

self: "/" field_spec? index_spec? class_spec

field_spec: "@" CNAME

index_spec: "[" DIGIT* "]"

class_spec: CNAME | tree

pattern_alt: tree ("|" tree)*

tree: "(" pattern_class_spec pattern_field_spec* ")"

pattern_class_spec: ANY | CLASS ("|" CLASS)*

pattern_field_spec: "@" FIELD_NAME ("=" (sequence | value))? capture?

sequence: "[" (value capture? ","?)* (ANY capture?)? "]"

value: pattern_alt | var | NONE | ESCAPED_STRING

capture: "->" CNAME

var: "$" CNAME

NONE: "None"

CLASS: CNAME

FIELD_NAME: CNAME

ANY: "*"

%import common.ESCAPED_STRING
%import common.WS
%import common.CNAME
%import common.LCASE_LETTER
%import common.DIGIT

%ignore WS

"""
import logging
import re
from contextlib import contextmanager
from dataclasses import replace
from enum import Enum, auto
from typing import (
    Any,
    Generator,
    Generic,
    Iterable,
    Mapping,
    NamedTuple,
    NoReturn,
    Sequence,
    TypeVar,
    cast,
)

from pyoak import config
from pyoak.match.error import ASTXpathOrPatternDefinitionError
from pyoak.match.helpers import check_and_get_ast_node_type
from pyoak.node import ASTNode

from .element import ASTXpathElement
from .pattern import (
    AlternativeMatcher,
    AnyMatcher,
    BaseMatcher,
    NodeMatcher,
    RegexMatcher,
    SequenceMatcher,
    ValueMatcher,
    VarMatcher,
)

_IT = TypeVar("_IT")

logger = logging.getLogger(__name__)


# Polyfill, instead of depending on typing-extensions
def _assert_never(arg: NoReturn, /) -> NoReturn:
    raise AssertionError(f"Unhandled type: {type(arg).__name__!r}")


class NoMatchError(Exception, Generic[_IT]):
    def __init__(self, expected: _IT, actual: _IT | None) -> None:
        self.expected = expected
        self.actual = actual

        super().__init__(f"Expected {expected}, got {actual}")


class LookaheadQueue(Generic[_IT]):
    def __init__(self, init_items: Iterable[_IT]) -> None:
        self._hit_eoq = False
        self._items = list(init_items)
        self._pos = 0
        self._len = len(self._items)

    def _next_item(self) -> _IT | None:
        """Internal API to fetch/read the next item into the queue.

        Implement this method in a subclass.

        Returns:
            _IT | None: the next item or None if there is no next item

        """
        return None

    @property
    def pos(self) -> int:
        """Get the current position in the queue.

        Returns:
            int: the position

        """
        return self._pos

    @property
    def len(self) -> int:
        """Get the length of the queue.

        Returns:
            int: the length

        """
        return self._len

    @property
    def items(self) -> tuple[_IT, ...]:
        """Get a read-only view of the items queue.

        Returns:
            tuple[_IT]: the items

        """
        return tuple(self._items)

    @property
    def hit_eoq(self) -> bool:
        """Check if we are at the end of the queue. Meaning there are no more items to consume.

        Returns:
            bool: True if we are at the end of the queue, False otherwise

        """
        return self._hit_eoq and self._pos >= self._len

    def feed(self) -> bool:
        """Feed the queue with one more item.

        Returns:
            bool: True if there is a new item, False otherwise

        """
        new_item = self._next_item()

        if new_item is None:
            self._hit_eoq = True
            return False

        self._items.append(new_item)
        self._len += 1
        return True

    def fill(self, count: int = -1) -> bool:
        """Fill the queue such that the `count` items are available.

        Args:
            count (int, optional): number of items to fill.
                Defaults to -1. If -1, fill all remaining items.

        Returns:
            bool: True if the requested number of items was filled,
                False otherwise

        """
        if count == -1:
            while self.feed():
                pass

            return self._pos < self._len

        while self._pos + count > self._len:
            if not self.feed():
                return False

        return True

    def peek(self, la: int = 1) -> _IT | None:
        """Peek at the lookeahed item in the queue.

        Args:
            la (int, optional): numer of items to look ahead. Defaults to 1.

        Returns:
            _IT | None: the item or None if there is no item

        """
        # Make sure we have enough items
        if not self.fill(la):
            return None

        p_index = self._pos + la - 1
        if p_index >= self._len:
            return None

        return self._items[p_index]

    la = peek
    """Alias for peek."""

    def lb(self, la: int = 1) -> _IT | None:
        """Look behind at the item in the queue.

        Args:
            la (int, optional): numer of items to look behind.
                Defaults to 1, meaning the last consumed item.

        Returns:
            _IT | None: the item or None if there is no item

        """
        lb_index = self._pos - la
        if lb_index < 0:
            return None

        return self._items[lb_index]

    def last(self) -> _IT | None:
        """Get the last consumed item. Shortcut for `lb(1)`.

        Returns:
            _IT | None: the item or None if there is no item

        """
        return self.lb(1)

    def consume(self) -> _IT | None:
        """Consume the next item in the queue.

        Returns:
            _IT | None: the item or None if there is no item

        """

        # Make sure we have at least one unconsumed item
        if not self.fill(1):
            return None

        self._pos += 1

        return self._items[self._pos - 1]

    def match(self, value: Any) -> _IT:
        """Consume the next item in the queue if it matches the given value. Or raise a
        NoMatchError.

        Args:
            value (_IT): the value to match

        Returns:
            _IT: the item or None if there is no item

        """
        item = self.peek()
        if item != value:
            raise NoMatchError(value, item)

        return cast(_IT, self.consume())


class TokenType(Enum):
    def __new__(cls, re_str: str) -> "TokenType":
        value = len(cls.__members__) + 1

        # Check if the member already exists
        for member in cls:
            if member._re_str == re_str:  # type: ignore[attr-defined]
                value = member.value

        obj = object.__new__(cls)
        obj._value_ = value
        obj._re_str = re_str  # type: ignore[attr-defined]
        return obj

    @property
    def re_str(self) -> str:
        return cast(str, self._re_str)  # type: ignore[attr-defined]

    _EOF = r"$"
    WS = r"\s+"
    CNAME = r"[_a-zA-Z][_a-zA-Z0-9]*"
    ESCAPED_STRING = r'"(?:[^"\\]|\\.)*"'
    DIGITS = r"[0-9]+"
    CAPTURE_START = r"->"
    CAPTURE_KEY = CNAME
    NONE = CNAME
    STAR = r"\*"
    LPAREN = r"\("
    RPAREN = r"\)"
    DOLLAR = r"\$"
    AT = r"@"
    LBRACKET = r"\["
    RBRACKET = r"\]"
    FSLASH = r"/"
    COMMA = r","
    EQUALS = r"="
    PIPE = r"\|"


def _pretty_print_tok_type(tok_type: TokenType) -> str:
    """Outputs a human readable version of the token type."""
    match tok_type:
        case TokenType._EOF:
            return "end of text"
        case TokenType.WS:
            return "whitespace"
        case TokenType.CNAME:
            return "a name (identifier)"
        case TokenType.ESCAPED_STRING:
            return "an escaped string (possibly a regex)"
        case TokenType.DIGITS:
            return "a number"
        case TokenType.CAPTURE_START:
            return "-> (capture indicator)"
        # This & None will never be hit, due to python enum aliases
        # being the same object as the original enum member
        case TokenType.CAPTURE_KEY:
            return "a capture name"
        case TokenType.NONE:
            return "'None'"
        case TokenType.STAR:
            return "'*' (any class / sequence tail marker)"
        case TokenType.LPAREN:
            return "'(' (pattern start)"
        case TokenType.RPAREN:
            return "')' (pattern end)"
        case TokenType.DOLLAR:
            return "'$' (variable indicator)"
        case TokenType.AT:
            return "'@' (field indicator)"
        case TokenType.LBRACKET:
            return "'[' (sequence start)"
        case TokenType.RBRACKET:
            return "']' (sequence end)"
        case TokenType.FSLASH:
            return "'/' (xpath element separator)"
        case TokenType.COMMA:
            return "',' (sequence item separator)"
        case TokenType.EQUALS:
            return "'=' (item value separator)"
        case TokenType.PIPE:
            return "'|' (alternative separator)"
        case _ as unreachable:
            _assert_never(unreachable)


def _get_lexer_re(
    *,
    include_tok_types: set[TokenType] | None = None,
    exclude_tok_types: set[TokenType] | None = None,
) -> re.Pattern[str]:
    """Get the regular expression for the lexer.

    Args:
        exclude_tok_types (Iterable[TokenType]): the token types to exclude

    Returns:
        re.Pattern[str]: the regular expression

    """
    if include_tok_types is None:
        include_tok_types = set(TokenType)

    if exclude_tok_types is None:
        exclude_tok_types = set()

    # Always exclude EOF, it is not for matching really, but for error reporting
    exclude_tok_types.add(TokenType._EOF)

    toks_to_include = include_tok_types - exclude_tok_types

    re_str = "|".join(
        f"(?P<{token.name}>{token.re_str})" for token in TokenType if token in toks_to_include
    )

    return re.compile(re_str)


_SEQ_OR_VALUE_FOLLOW_SET = (
    TokenType.RBRACKET,
    TokenType.LPAREN,
    TokenType.DOLLAR,
    TokenType.NONE,
    TokenType.ESCAPED_STRING,
    TokenType.STAR,
)


class Token(NamedTuple):
    type: TokenType
    value: str
    start: int
    stop: int
    line: int
    column: int
    source: str

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.start}, {self.stop}, {self.line}, {self.column}, source len={len(self.source)})"

    def __eq__(self, value: object) -> bool:
        # Allow "matching" with TokenType by type only
        # match by enum value to support aliases
        if isinstance(value, TokenType):
            return self.type == value
            # return cast(int, self.type.value) == cast(int, value.value)

        if isinstance(value, Token):
            return tuple(self) == tuple(value)

        return NotImplemented

    def __ne__(self, value: object) -> bool:
        # Allow "matching" with TokenType by type only
        # match by enum value to support aliases
        if isinstance(value, TokenType):
            return self.type != value
            # return cast(int, self.type.value) != cast(int, value.value)

        if isinstance(value, Token):
            return tuple(self) != tuple(value)

        return NotImplemented


_LEXER_REGULAR_RE = _get_lexer_re()


class LexerMode(Enum):
    """Lexer mode.

    Left for future use, to allow different token sets to be allowed in different contexts.

    The idea is to have a different regex for each mode, and switch between them using a context
    manager on the lexer.

    """

    REGULAR = auto()


class UnexpectedCharactersError(Exception):
    def __init__(
        self,
        text: str,
        text_pos: int,
        cur_line: int,
        cur_column: int,
        previous_tokens: Sequence[Token],
        mode: LexerMode,
    ) -> None:
        self.text = text
        self.text_pos = text_pos
        self.cur_line = cur_line
        self.cur_column = cur_column
        self.previous_tokens = previous_tokens
        self.mode = mode

        super().__init__(str(self))

    def __str__(self) -> str:
        msg = f"No matching token found at line {self.cur_line}, column {self.cur_column}\n\n"

        prev_tokens = self.previous_tokens[-5:]

        if prev_tokens:
            msg += "Previous tokens:\n"
            msg += "\n".join(f"{token!r}" for token in prev_tokens)
            msg += "\n\n"

        msg += f"Text content:\n{self.text[self.text_pos-20:self.text_pos+20]}\n"

        return msg


class Lexer(LookaheadQueue[Token]):
    def __init__(self, text: str) -> None:
        self._text = text
        self._text_pos = 0
        self._cur_line = 1
        self._cur_column = 0
        self._mode_stack = [LexerMode.REGULAR]

        super().__init__([])

    @property
    def text(self) -> str:
        """Get the text being lexed.

        Returns:
            str: the text

        """
        return self._text

    @property
    def text_pos(self) -> int:
        """Get the current text position.

        Returns:
            int: the position

        """
        return self._text_pos

    @contextmanager
    def mode(self, mode: LexerMode) -> Generator[None, None, None]:
        """Context manager to switch lexer state.

        It is not currently used, but left here for future use.
        See `LexerMode` for more details.

        Returns:
            AbstractContextManager[None]: the context manager

        """
        self._mode_stack.append(mode)

        try:
            yield
        finally:
            self._mode_stack.pop()

    def _advance_pos(self, token: Token) -> None:
        """Advance the current text, line and column positions.

        Args:
            token (Token): the token

        """
        for c in token.value:
            if c == "\n":
                self._cur_line += 1
                self._cur_column = 0
            else:
                self._cur_column += 1

        self._text_pos = token.stop

    def _next_item(self) -> Token | None:
        """Internal API to fetch/read the next item into the queue.

        Implement this method in a subclass.

        Returns:
            _IT | None: the next item or None if there is no next item

        """

        while self._text_pos < len(self._text):
            m = _LEXER_REGULAR_RE.match(self._text, self._text_pos)

            if m is None:
                raise UnexpectedCharactersError(
                    self._text,
                    self._text_pos,
                    self._cur_line,
                    self._cur_column,
                    self.items,
                    self._mode_stack[-1],
                )

            group, value = m.lastgroup, m.group()

            if group is None or value is None:
                raise RuntimeError("Internal error: no group matched")

            try:
                type_ = TokenType[group]
            except KeyError:
                raise RuntimeError(
                    f"Internal error: unknown token re-group matched {group}"
                ) from None

            token = Token(
                type=type_,
                value=value,
                start=m.start(),
                stop=m.end(),
                line=self._cur_line,
                column=self._cur_column,
                source=self._text,
            )

            self._advance_pos(token)

            if type_ == TokenType.ESCAPED_STRING:
                # Unescape string
                token = token._replace(value=token.value[1:-1])

            if type_ == TokenType.WS:
                # Ignore whitespace
                continue

            return token

        # Reached end of text
        return None


class Parser:
    def __init__(self, types: Mapping[str, type[Any]]) -> None:
        self._types = types

        self._reset("")

    def _reset(self, text: str) -> None:
        self._lexer = Lexer(text)
        self._captures_seen: set[str] = set()

    def _get_grammar_error_exception(
        self,
        msg: str,
        expected: Sequence[TokenType],
        actual: TokenType | None,
        *,
        omit_context: bool = False,
    ) -> ASTXpathOrPatternDefinitionError:
        if omit_context:
            return ASTXpathOrPatternDefinitionError(msg)

        if expected:
            expected_str = ", ".join(_pretty_print_tok_type(tok_type) for tok_type in expected)

            msg += f"\nExpected: {expected_str}"

        if actual is not None:
            msg += f", got: {_pretty_print_tok_type(actual)}"

        prev_tokens = self._lexer.items[-5:]

        if prev_tokens:
            msg += "\n\nPrevious tokens:\n"
            msg += "\n".join(f"{token!r}" for token in prev_tokens)

        msg += (
            "\n\nText content:\n"
            f"{self._lexer.text[self._lexer.text_pos-20:self._lexer.text_pos+20]}"
        )

        return ASTXpathOrPatternDefinitionError(msg)

    def _match_or_raise(self, token_type: TokenType, msg: str) -> Token:
        """Match the next token or raise an error.

        Args:
            token_type (TokenType): the token type to match

        Raises:
            NoMatchError: if the next token does not match the given type

        Returns:
            Token: the matched token

        """
        try:
            token = self._lexer.match(token_type)
        except NoMatchError as e:
            raise self._get_grammar_error_exception(
                msg, [e.expected], e.actual or TokenType._EOF
            ) from None

        return token

    def _check_and_save_capture_key(self, capture_key: str) -> None:
        # We didn't allow capture keys to be used more than once previously
        # This code is left temporarily, in case we want to re-enable this
        # if capture_key in self._captures_seen:
        #     raise self._get_grammar_error_exception(
        #         f"Capture name <{capture_key}> used more than once", [], None
        #     )

        self._captures_seen.add(capture_key)

    def _xpath(self) -> Sequence[ASTXpathElement]:
        """xpath: element* self"""
        elements = []

        # Collect elements. Since hit_eoq will only work (i.e. show that there no more tokens)
        # after the first call to peek, which in turn only happens inside `_element` call
        # this will effectively check that xpath is not empty, because `_element` will raise
        # an error if there are no more tokens
        while not self._lexer.hit_eoq:
            elements.append(self._element())

        # Check that the last item (self) is not empty
        _, _, ast_class_or_pattern = elements[-1]

        if ast_class_or_pattern is None:
            raise ASTXpathOrPatternDefinitionError(
                "Incorrect xpath definition. Last element must be an AST node type or a pattern."
            )

        ret: list[ASTXpathElement] = []

        # The following logic parses the xpath in reverse order
        elements_iter = reversed(elements)
        for el in elements_iter:
            parent_field, parent_index, ast_class_or_pattern = el

            # Check if this is from plain "//"
            is_any = el == (None, None, None)

            # If not make sure that ast class is implicitly set to ASTNode
            if not is_any:
                ast_class_or_pattern = ast_class_or_pattern or ASTNode

            while is_any:
                next_el = next(elements_iter, None)
                if next_el is None:
                    # We are at the very beginning of the xpath
                    # and it starts with // (or more)
                    # change last element to anywhere and return
                    ret[-1] = ASTXpathElement(
                        ret[-1].ast_class_or_pattern,
                        ret[-1].parent_field,
                        ret[-1].parent_index,
                        True,
                    )
                    return ret

                # Change last element to anywhere
                ret[-1] = ASTXpathElement(
                    ret[-1].ast_class_or_pattern, ret[-1].parent_field, ret[-1].parent_index, True
                )

                parent_field, parent_index, ast_class_or_pattern = next_el

                # Check if this is from plain "//"
                is_any = next_el == (None, None, None)

                # If not make sure that ast class is implicitly set to ASTNode
                if not is_any:
                    ast_class_or_pattern = ast_class_or_pattern or ASTNode

            ret.append(
                ASTXpathElement(
                    ast_class_or_pattern=cast(type[ASTNode] | NodeMatcher, ast_class_or_pattern),
                    parent_field=parent_field,
                    parent_index=parent_index,
                    anywhere=False,
                )
            )

        return ret

    def _element(self) -> tuple[str | None, int | None, type[ASTNode] | NodeMatcher | None]:
        """element: "/" field_spec? index_spec? class_spec?"""

        type_or_pattern: type[ASTNode] | NodeMatcher | None = None
        parent_field: str | None = None
        parent_index: int | None = None

        self._match_or_raise(TokenType.FSLASH, "Incorrect xpath element definition.")

        while not self._lexer.hit_eoq:
            match self._lexer.peek():
                case TokenType.AT:
                    parent_field = self._field_spec()
                    continue
                case TokenType.LBRACKET:
                    parent_index = self._index_spec()
                    continue
                case TokenType.CNAME:
                    type_or_pattern = self._class_spec()
                    continue
                case TokenType.LPAREN:
                    type_or_pattern = self._tree()
                    continue
                case _:
                    break

        return parent_field, parent_index, type_or_pattern

    def _field_spec(self) -> str:
        """field_spec: "@" CNAME"""
        self._match_or_raise(TokenType.AT, "Incorrect field specification in xpath.")
        return self._match_or_raise(
            TokenType.CNAME, "Incorrect field specification in xpath."
        ).value

    def _index_spec(self) -> int | None:
        """index_spec: "[" DIGIT* "]" """
        self._match_or_raise(
            TokenType.LBRACKET, "Incorrect definition of sequence positin in xpath."
        )

        if self._lexer.peek() == TokenType.RBRACKET:
            self._lexer.consume()
            return None

        value = int(
            self._match_or_raise(
                TokenType.DIGITS, "Incorrect definition of sequence positin in xpath."
            ).value
        )

        self._match_or_raise(
            TokenType.RBRACKET, "Incorrect definition of sequence positin in xpath."
        )

        return value

    def _class_spec(self) -> type[ASTNode] | NodeMatcher:
        """class_spec: CNAME | tree"""
        class_name = self._match_or_raise(
            TokenType.CNAME, "Incorrect definition of class name in xpath."
        ).value

        type_, msg = check_and_get_ast_node_type(class_name, self._types)

        if type_ is None:
            raise self._get_grammar_error_exception(msg, [], None, omit_context=True)

        return type_

    def _pattern_alt(self) -> AlternativeMatcher | NodeMatcher:
        """pattern_alt: tree ("|" tree)*"""
        matchers: list[NodeMatcher] = []

        while not self._lexer.hit_eoq:
            matchers.append(self._tree())

            if self._lexer.peek() != TokenType.PIPE:
                break

            self._lexer.consume()

        # Check for empty pattern. This is not allowed
        if not matchers:
            raise ASTXpathOrPatternDefinitionError(
                "Empty pattern definition. Make sure to include at least one tree pattern."
            )

        if len(matchers) == 1:
            return matchers[0]

        return AlternativeMatcher(matchers=tuple(matchers))

    def _tree(self) -> NodeMatcher:
        """tree: "(" class_spec field_spec* ")" """
        err_msg = "Incorrect definition of a tree pattern."
        self._match_or_raise(TokenType.LPAREN, err_msg)

        match_types: list[type[ASTNode]] = self._pattern_class_spec()

        # Then parse field_spec which will have 0+ field names and matchers
        content: list[tuple[str, BaseMatcher]] = []

        while not self._lexer.hit_eoq:
            match self._lexer.peek():
                case TokenType.RPAREN:
                    # not consuming here, since we match at the end of the loop
                    break
                case TokenType.AT:
                    content.append(self._pattern_field_spec())
                    continue
                case _:
                    break

        self._match_or_raise(TokenType.RPAREN, err_msg)

        return NodeMatcher(types=tuple(match_types), content=tuple(content))

    def _pattern_class_spec(self) -> list[type[ASTNode]]:
        """pattern_class_spec: ANY | CLASS ("|" CLASS)*"""

        match_types: list[type[ASTNode]] = []
        class_names: list[str] = []

        # First check if this is ANY
        if self._lexer.peek() == TokenType.STAR:
            self._lexer.consume()
            return [ASTNode]

        # Ok, now we may only have class names
        # And we must have at least one
        expect_class_name = True

        while not self._lexer.hit_eoq:
            if expect_class_name:
                class_names.append(
                    self._match_or_raise(
                        TokenType.CNAME, "Incorrect definition of class name in a tree pattern."
                    ).value
                )

                expect_class_name = False
                continue

            match self._lexer.peek():
                case TokenType.PIPE:
                    self._lexer.consume()
                    expect_class_name = True
                    continue
                case _:
                    break

        if not class_names:
            raise self._get_grammar_error_exception(
                "Incorrect definition of class name in a tree pattern. ",
                [TokenType.CNAME, TokenType.STAR],
                None,
            )

        for class_name in class_names:
            type_, msg = check_and_get_ast_node_type(class_name, self._types)
            if type_ is None:
                raise self._get_grammar_error_exception(msg, [], None, omit_context=True)

            match_types.append(type_)

        return match_types

    def _pattern_field_spec(self) -> tuple[str, BaseMatcher]:
        """pattern_field_spec: "@" FIELD_NAME ("=" (sequence | value))? capture?"""

        self._match_or_raise(TokenType.AT, "Incorrect definition of field name in a tree pattern.")

        field_name = self._match_or_raise(
            TokenType.CNAME, "Incorrect definition of field name in a tree pattern."
        ).value

        matcher: BaseMatcher = AnyMatcher()
        has_value = False
        capture_key: str | None = None

        match self._lexer.peek():
            case TokenType.EQUALS:
                self._lexer.consume()
                matcher = self._sequence_or_value()
                has_value = True
            case TokenType.CAPTURE_START:
                self._lexer.consume()
                capture_key = self._parse_capture_key()
            case _:
                # Nothing, just a field name
                pass

        # Now try matching again, as we may have a capture key after a value
        if has_value:
            match self._lexer.peek():
                case TokenType.CAPTURE_START:
                    self._lexer.consume()
                    capture_key = self._parse_capture_key()
                case _:
                    # Nothing, just a field name
                    pass

        if capture_key is not None:
            self._check_and_save_capture_key(capture_key)

            return field_name, replace(matcher, name=capture_key)

        return field_name, matcher

    def _parse_capture_key(self) -> str:
        try:
            capture_key = self._match_or_raise(
                TokenType.CAPTURE_KEY,
                "Incorrect definition of capture key in a tree pattern.",
            ).value
        except UnexpectedCharactersError as e:
            raise ASTXpathOrPatternDefinitionError(
                f"Incorrect definition of capture key in a tree pattern.\n{e!s}"
            ) from None

        return capture_key

    def _sequence_or_value(self) -> BaseMatcher:
        """(sequence | value)"""

        try:
            next_tok = self._lexer.peek()
        except UnexpectedCharactersError as e:
            raise ASTXpathOrPatternDefinitionError(
                f"Incorrect definition of field value in a tree pattern.\n{e!s}"
            ) from None

        match next_tok:
            case TokenType.LBRACKET:
                return self._sequence()
            case TokenType.LPAREN:
                return self._pattern_alt()
            case TokenType.DOLLAR | TokenType.CNAME | TokenType.ESCAPED_STRING:
                # TokenType.CNAME here is in place of NONE
                return self._value()
            case TokenType.STAR:
                raise self._get_grammar_error_exception(
                    "Incorrect definition of field value in a tree pattern. "
                    "Sequence tail marker '*' must be inside [].",
                    [TokenType.LBRACKET],
                    None,
                )
            case _ as tok:
                raise self._get_grammar_error_exception(
                    "Incorrect definition of field value in a tree pattern. ",
                    [
                        TokenType.LBRACKET,
                        TokenType.LPAREN,
                        TokenType.DOLLAR,
                        TokenType.NONE,
                        TokenType.ESCAPED_STRING,
                    ],
                    tok.type if tok else None,
                )

    def _sequence(self) -> SequenceMatcher | ValueMatcher:
        """sequence: "[" (value capture? ","?)* (ANY capture?)? "]" """

        err_msg = "Incorrect definition of sequence in a tree pattern."
        self._match_or_raise(TokenType.LBRACKET, err_msg)

        matchers: list[BaseMatcher] = []
        hit_any: bool = False
        expect_comma: bool = False

        next_tok = self._lexer.peek()

        while next_tok and next_tok != TokenType.RBRACKET:
            match next_tok:
                case TokenType.LPAREN:
                    matchers.append(self._pattern_alt())
                    expect_comma = True
                case TokenType.DOLLAR | TokenType.CNAME | TokenType.ESCAPED_STRING:
                    matchers.append(self._value())
                    expect_comma = True
                case TokenType.COMMA:
                    if not expect_comma:
                        raise self._get_grammar_error_exception(
                            err_msg,
                            _SEQ_OR_VALUE_FOLLOW_SET,
                            TokenType.COMMA,
                        )

                    self._lexer.consume()
                    expect_comma = False

                    # We can't have capture after a comma
                    # So we skip the logic after this match
                    next_tok = self._lexer.peek()
                    continue
                case TokenType.STAR:
                    # There could only be one
                    if hit_any:
                        raise self._get_grammar_error_exception(
                            f"{err_msg}. Multiple '*' sequence tail specifiers.",
                            [TokenType.RBRACKET],
                            TokenType.STAR,
                        )

                    self._lexer.consume()
                    matchers.append(AnyMatcher())
                    expect_comma = False
                case _ as tok:
                    raise self._get_grammar_error_exception(
                        err_msg,
                        _SEQ_OR_VALUE_FOLLOW_SET,
                        tok.type if tok else None,
                    )

            next_tok = self._lexer.peek()

            if next_tok == TokenType.CAPTURE_START:
                self._lexer.consume()
                capture_key = self._parse_capture_key()

                self._check_and_save_capture_key(capture_key)

                matchers[-1] = replace(matchers[-1], name=capture_key)
                next_tok = self._lexer.peek()

        self._match_or_raise(
            TokenType.RBRACKET, "Incorrect definition of sequence in a tree pattern."
        )

        if len(matchers) == 0:
            return ValueMatcher(value=())

        return SequenceMatcher(matchers=tuple(matchers))

    def _value(self) -> ValueMatcher | RegexMatcher | VarMatcher:
        """value: tree | var | NONE | ESCAPED_STRING

        But tree is already handled by _sequence_or_value
        """
        match self._lexer.peek():
            case TokenType.DOLLAR:
                self._lexer.consume()
                var_name = self._parse_capture_key()

                if var_name not in self._captures_seen:
                    raise self._get_grammar_error_exception(
                        f"Pattern uses match variable <{var_name}> before it was captured",
                        [],
                        None,
                    )

                return VarMatcher(var_name=var_name)
            case TokenType.CNAME as tok:
                if tok.value == "None":
                    self._lexer.consume()
                    return ValueMatcher(value=None)

                # Error
                raise self._get_grammar_error_exception(
                    "Incorrect definition of field value in a tree pattern.",
                    [TokenType.NONE],
                    tok.type,
                )
            case TokenType.ESCAPED_STRING:
                tok = self._lexer.consume()
                return RegexMatcher(_re_str=tok.value)
            case _ as tok:
                raise self._get_grammar_error_exception(
                    "Incorrect definition of field value in a tree pattern.",
                    [TokenType.DOLLAR, TokenType.NONE, TokenType.ESCAPED_STRING],
                    tok.type if tok else None,
                )

    def parse_xpath(self, xpath: str) -> Sequence[ASTXpathElement]:
        """Public API to parse an xpath.

        Args:
            xpath (str): the xpath

        Returns:
            ASTXpath: the parsed xpath

        """
        if not xpath.startswith("/"):
            # Relative path is the same as absolute path starting with "anywehere"
            # It also allows us to simplify the grammar
            xpath = "//" + xpath

        self._reset(xpath)

        try:
            return self._xpath()
        except UnexpectedCharactersError as e:
            # This will catch lexer errors that are not checked in the rules
            raise ASTXpathOrPatternDefinitionError(
                f"Incorrect xpath definition. Context:\n{e!s}"
            ) from None
        except ASTXpathOrPatternDefinitionError:
            raise
        except Exception as e:
            if config.TRACE_LOGGING:
                logger.debug("Internal error parsing xpath", exc_info=True)
            raise ASTXpathOrPatternDefinitionError(
                "Failed to parse Xpath due to internal error. Please report it!"
            ) from e

    def parse_pattern(self, pattern: str) -> AlternativeMatcher | NodeMatcher:
        """Public API to parse a tree pattern.

        Args:
            pattern (str): the pattern

        Returns:
            ASTXpath: the parsed pattern

        """

        self._reset(pattern)

        try:
            ret = self._pattern_alt()

            # This is the top-level alternative, so it must be the last thing in the pattern
            if not self._lexer.hit_eoq and (next_tok := self._lexer.peek()):
                raise self._get_grammar_error_exception(
                    "Incorrect tree pattern definition.",
                    [TokenType._EOF],
                    next_tok.type,
                )

            return ret
        except UnexpectedCharactersError as e:
            # This will catch lexer errors that are not checked in the rules
            raise ASTXpathOrPatternDefinitionError(
                f"Incorrect tree pattern definition. Context:\n{e!s}"
            ) from None
        except ASTXpathOrPatternDefinitionError:
            raise
        except Exception as e:
            if config.TRACE_LOGGING:
                logger.debug("Internal error parsing tree pattern", exc_info=True)
            raise ASTXpathOrPatternDefinitionError(
                "Failed to parse a tree pattern due to internal error. Please report it!"
            ) from e
