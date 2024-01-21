from dataclasses import dataclass
from typing import Iterable

import pytest
from pyoak.match.error import ASTXpathOrPatternDefinitionError
from pyoak.match.parser import (
    Lexer,
    LexerMode,
    LookaheadQueue,
    NoMatchError,
    Parser,
    Token,
    TokenType,
)
from pyoak.node import ASTNode


class IntLAQueue(LookaheadQueue[int]):
    def __init__(self, init_items: Iterable[int], feed_n: int) -> None:
        super().__init__(init_items)

        self._feed_n = feed_n

    def _next_item(self) -> int | None:
        if self._feed_n == 0:
            return None

        self._feed_n -= 1
        return self._len + 1


def test_lookahead_queue_initialization():
    queue = IntLAQueue([1, 2, 3], 0)
    assert queue.items == (1, 2, 3)
    assert queue.pos == 0
    assert queue.len == 3
    assert not queue.hit_eoq
    assert not queue.feed()


def test_lookahead_queue_feed():
    queue = IntLAQueue([1, 2, 3], 1)

    # Test will feed only 1 additional item
    assert queue.feed()
    assert queue.items == (1, 2, 3, 4)
    assert queue.pos == 0
    assert queue.len == 4

    assert not queue.hit_eoq
    assert not queue.feed()

    # Should not feed any more items
    assert not queue.feed()
    assert queue.items == (1, 2, 3, 4)
    assert queue.pos == 0
    assert queue.len == 4


def test_lookahead_queue_fill():
    queue = IntLAQueue([1, 2, 3], 5)

    # Test fill all items
    assert queue.fill()
    assert queue.items == (1, 2, 3, 4, 5, 6, 7, 8)
    assert queue.pos == 0
    assert queue.len == 8

    assert not queue.hit_eoq
    assert not queue.feed()

    # Consecutive fills should not change anything
    assert queue.fill()
    assert queue.items == (1, 2, 3, 4, 5, 6, 7, 8)

    # Recreate queue
    queue = IntLAQueue([1, 2, 3], 5)

    # Test fill only 6 items (3 initial + 3 additional)
    assert queue.fill(6)
    assert queue.items == (1, 2, 3, 4, 5, 6)
    assert queue.pos == 0
    assert queue.len == 6

    assert not queue.hit_eoq
    assert queue.feed()  # this will force feed another item

    # Now one more item should be available
    assert queue.fill(7)
    assert queue.items == (1, 2, 3, 4, 5, 6, 7)
    assert queue.pos == 0
    assert queue.len == 7

    assert not queue.hit_eoq

    # Try filling more items than available
    # This will still fill all items but return False
    assert not queue.fill(10)
    assert queue.items == (1, 2, 3, 4, 5, 6, 7, 8)
    assert queue.pos == 0
    assert queue.len == 8

    assert not queue.hit_eoq
    assert not queue.feed()


def test_lookahead_queue_peek_la():
    queue = IntLAQueue([1, 2, 3], 0)

    assert queue.peek() == queue.la() == 1
    assert queue.peek(1) == queue.la(1) == 1
    assert queue.peek(2) == queue.la(2) == 2
    assert queue.peek(3) == queue.la(3) == 3
    assert queue.peek(4) == queue.la(4) is None


def test_lookahead_queue_lb():
    queue = IntLAQueue([1, 2, 3], 0)

    assert queue.lb() is None

    # Consume all items
    while _ := queue.consume():
        pass

    assert queue.lb() == 3
    assert queue.lb(1) == 3
    assert queue.lb(2) == 2
    assert queue.lb(3) == 1
    assert queue.lb(4) is None


def test_lookahead_queue_last():
    queue = IntLAQueue([1, 2, 3], 0)

    assert queue.last() is None

    for i in range(3):
        queue.consume()
        assert queue.last() == i + 1


def test_lookahead_queue_consume():
    queue = IntLAQueue([1, 2, 3], 0)
    assert queue.consume() == 1
    assert queue.consume() == 2
    assert queue.consume() == 3
    assert queue.consume() is None


def test_lookahead_queue_match():
    queue = IntLAQueue([1, 2, 3], 0)

    assert queue.match(1) == 1

    with pytest.raises(NoMatchError) as exc_info:
        # 3 is not next item
        queue.match(3)

    assert exc_info.value.expected == 3
    assert exc_info.value.actual == 2

    assert queue.match(2) == 2

    with pytest.raises(NoMatchError) as exc_info:
        # try a different type
        queue.match("2")

    assert exc_info.value.expected == "2"
    assert exc_info.value.actual == 3


def test_lookahead_queue_match_any():
    queue = IntLAQueue([1, 2, 3], 0)

    assert queue.match_any([1, 2, 3]) == 1

    with pytest.raises(NoMatchError) as exc_info:
        # 3 is not next item
        queue.match_any([3])

    assert exc_info.value.expected == [3]
    assert exc_info.value.actual == 2

    assert queue.match_any([2]) == 2

    with pytest.raises(NoMatchError) as exc_info:
        # try a different type
        queue.match_any(["2"])

    assert exc_info.value.expected == ["2"]
    assert exc_info.value.actual == 3


def test_token_equality():
    token1 = Token(TokenType.CNAME, "test", 0, 4, 1, 0, "test text")
    token2 = Token(TokenType.CNAME, "test", 2, 6, 1, 2, "test text")
    token3 = Token(TokenType.CNAME, "test", 0, 4, 1, 0, "test text")
    token4 = Token(TokenType.ESCAPED_STRING, "test", 0, 4, 1, 0, "test text")

    assert token1 != token2
    assert token1 == token3
    assert token1 == TokenType.CNAME
    assert token1 != token4
    assert token1 != "not a token"


def test_init():
    lexer = Lexer("test text")
    assert lexer._text == "test text"
    assert lexer._text_pos == 0
    assert lexer._cur_line == 1
    assert lexer._cur_column == 0
    assert lexer._mode_stack == [LexerMode.REGULAR]


def test_mode():
    lexer = Lexer("test text")
    with lexer.mode(LexerMode.REGULAR):
        assert lexer._mode_stack == [LexerMode.REGULAR, LexerMode.REGULAR]
    assert lexer._mode_stack == [LexerMode.REGULAR]


def test_advance_pos():
    text = "test text"
    lexer = Lexer(text)
    token = Token(TokenType.CNAME, "test", 0, 4, 1, 0, text)
    lexer._advance_pos(token)
    assert lexer._text_pos == 4
    assert lexer._cur_line == 1
    assert lexer._cur_column == 4

    # Try WS with newlines
    text = "test\n\t\n text"
    lexer = Lexer(text)
    token = Token(TokenType.WS, "\n\t\n ", 4, 8, 1, 4, text)
    lexer._advance_pos(token)
    assert lexer._text_pos == 8
    assert lexer._cur_line == 3
    assert lexer._cur_column == 1


ALL_LEXEMS = '($None"test\\"test"[as,with,none,assuf,withsuf'


@pytest.mark.parametrize(
    "text,expected,mode",
    [
        ("test", [Token(TokenType.CNAME, "test", 0, 4, 1, 0, "test")], LexerMode.REGULAR),
        (
            "test1 test2",
            [
                Token(TokenType.CNAME, "test1", 0, 5, 1, 0, "test1 test2"),
                Token(TokenType.CNAME, "test2", 6, 11, 1, 6, "test1 test2"),
            ],
            LexerMode.REGULAR,
        ),
        (
            "(test)",
            [
                Token(TokenType.LPAREN, "(", 0, 1, 1, 0, "(test)"),
                Token(TokenType.CNAME, "test", 1, 5, 1, 1, "(test)"),
                Token(TokenType.RPAREN, ")", 5, 6, 1, 5, "(test)"),
            ],
            LexerMode.REGULAR,
        ),
        (
            "[test]",
            [
                Token(TokenType.LBRACKET, "[", 0, 1, 1, 0, "[test]"),
                Token(TokenType.CNAME, "test", 1, 5, 1, 1, "[test]"),
                Token(TokenType.RBRACKET, "]", 5, 6, 1, 5, "[test]"),
            ],
            LexerMode.REGULAR,
        ),
        (
            '"test\\"test"',
            [Token(TokenType.ESCAPED_STRING, 'test\\"test', 0, 12, 1, 0, '"test\\"test"')],
            LexerMode.REGULAR,
        ),
        (
            '($None"test\\"test"[as,with,none,assuf,withsuf',
            [
                Token(TokenType.LPAREN, "(", 0, 1, 1, 0, ALL_LEXEMS),
                Token(TokenType.DOLLAR, "$", 1, 2, 1, 1, ALL_LEXEMS),
                Token(TokenType.NONE, "None", 2, 6, 1, 2, ALL_LEXEMS),
                Token(TokenType.ESCAPED_STRING, 'test\\"test', 6, 18, 1, 6, ALL_LEXEMS),
                Token(TokenType.LBRACKET, "[", 18, 19, 1, 18, ALL_LEXEMS),
                Token(TokenType.AS, "as", 19, 21, 1, 19, ALL_LEXEMS),
                Token(TokenType.COMMA, ",", 21, 22, 1, 21, ALL_LEXEMS),
                Token(TokenType.WITH, "with", 22, 26, 1, 22, ALL_LEXEMS),
                Token(TokenType.COMMA, ",", 26, 27, 1, 26, ALL_LEXEMS),
                Token(TokenType.CNAME, "none", 27, 31, 1, 27, ALL_LEXEMS),
                Token(TokenType.COMMA, ",", 31, 32, 1, 31, ALL_LEXEMS),
                Token(TokenType.CNAME, "assuf", 32, 37, 1, 32, ALL_LEXEMS),
                Token(TokenType.COMMA, ",", 37, 38, 1, 37, ALL_LEXEMS),
                Token(TokenType.CNAME, "withsuf", 38, 45, 1, 38, ALL_LEXEMS),
            ],
            LexerMode.REGULAR,
        ),
        # TODO: more tests
    ],
)
def test_correct_lexing(text: str, expected: list[Token], mode: LexerMode):
    lexer = Lexer(text)

    with lexer.mode(mode):
        all_tokens = []
        while token := lexer.consume():
            all_tokens.append(token)

        assert len(all_tokens) == len(expected)

        # Tokens are equal purely based on type
        # so we need to check the rest of the attributes
        assert all(tok._asdict() == exp_tok._asdict() for tok, exp_tok in zip(all_tokens, expected))


def test_error_reporting():
    @dataclass
    class TestType(ASTNode):
        pass

    parser = Parser({"TestType": TestType})

    with pytest.raises(ASTXpathOrPatternDefinitionError) as excinfo:
        parser.parse_xpath("//(TestType ])")

    assert excinfo.value.message.startswith(
        """Incorrect definition of a tree pattern.
Expected one of: ')' (pattern end), '@' (field indicator). Got: ']' (sequence end)

Text context:
//(TestType ])
         ---^"""
    )

    with pytest.raises(ASTXpathOrPatternDefinitionError) as excinfo:
        parser.parse_pattern(
            "(TestType @attr=(TestType @attr -> cap)) |"
            "(TestType @attr=(TestType @attr -> (* @id -> id)))"
        )

    assert excinfo.value.message.startswith(
        """Incorrect definition of capture key in a tree pattern.
Expected: a name (identifier). Got: '(' (pattern start)

Text context:
)) |(TestType @attr=(TestType @attr -> (* @id -> id)))
                                    ---^"""
    )

    with pytest.raises(ASTXpathOrPatternDefinitionError) as excinfo:
        parser.parse_pattern("(TestType @attr=(TestType @attr cap))")

    assert excinfo.value.message.startswith(
        """Incorrect definition of a tree pattern.
Expected one of: ')' (pattern end), '@' (field indicator). Got: a name (identifier)

Text context:
(TestType @attr=(TestType @attr cap))
                             ---^^^"""
    )
