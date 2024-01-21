from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pyoak.match.pattern import BaseMatcher, validate_pattern
from pyoak.node import ASTNode
from pyoak.origin import NO_ORIGIN

origin = NO_ORIGIN


@dataclass
class PTestChild1(ASTNode):
    foo: str


@dataclass
class PTestChild2(ASTNode):
    bar: str


@dataclass
class PTestParent(ASTNode):
    foo: str = "foo_val"
    bar: str = "bar_val"
    child1: PTestChild1 | None = None
    child_any: PTestChild2 | PTestChild1 | None = None
    sub_parent: PTestParent | None = None
    child_tuple: tuple[ASTNode, ...] = field(default_factory=tuple)
    child_tuple2: tuple[ASTNode, ...] = field(default_factory=tuple)


@dataclass
class Expr(ASTNode):
    pass


@dataclass
class BinOp(Expr):
    left: Expr
    op: str
    right: Expr


@dataclass
class UnOp(Expr):
    op: str
    expr: Expr


@dataclass
class Lit(Expr):
    val: Any


@pytest.mark.parametrize(
    "rule, pattern_def",
    [
        ("any_class", "(*)"),
        ("top_level_cap", "(*) -> cap"),
        ("class_only", "(PTestParent)"),
        ("multi_class", "(PTestParent | PTestChild1)"),
        ("alternative", "(PTestParent) | (PTestChild1)"),
        ("alternative_with_bounds", "<(PTestParent) | (PTestChild1)>"),
        ("one_field_any_val", "(PTestParent @foo)"),
        ("one_field_any_val_cap", "(PTestParent @foo -> foo_val)"),
        ("one_field_val", '(PTestParent @foo="val")'),
        ("one_field_val_cap", '(PTestParent @foo="val" -> foo_val)'),
        ("two_fields_any_val_sec_any", "(PTestParent @foo @bar)"),
        ("two_fields_any_val_cap_sec_any", "(PTestParent @foo -> foo_val @bar)"),
        ("two_fields_val_sec_any", '(PTestParent @foo="val" @bar="barval")'),
        (
            "two_fields_val_cap_sec_val",
            '(PTestParent @foo="val" -> foo_val  @bar="barval")',
        ),
        (
            "two_fields_any_val_sec_val_cap",
            '(PTestParent @foo @bar="barval" -> bar_val)',
        ),
        (
            "two_fields_any_val_cap_sec_val_cap",
            '(PTestParent @foo=None -> foo_val  @bar="barval" -> bar_val)',
        ),
        ("one_field_val_none", "(PTestParent @child1=None)"),
        ("one_field_val_empty", "(PTestParent @child_tuple=[])"),
        ("one_child_val_any", "(PTestParent @child1=(*))"),
        ("one_child_val_any_cap", "(PTestParent @child1=(*) -> child_cap)"),
        ("one_child_val_alt", "(PTestParent @child1=(PTestParent) | (PTestChild1))"),
        (
            "one_child_val_alt_cap",
            "(PTestParent @child1=(PTestParent) | (PTestChild1) -> child_cap)",
        ),
        ("one_child_val_none_cap", "(PTestParent @child1=None -> child_cap)"),
        ("one_child_val_empty_cap", "(PTestParent @child_tuple=[] -> child_cap)"),
        ("var_any_cap", "(* @child1=(*) -> child_cap @child2 = $child_cap)"),
        ("one_child_val_arr", "(PTestParent @child_tuple=[(*), (*) *])"),
        (
            "one_child_val_arr_cap",
            "(PTestParent @child_tuple=[(*), (*), *] -> child_cap)",
        ),
        (
            "one_child_val_arr_cap_inner",
            "(PTestParent @child_tuple=[(*) -> child_cap, (*), *])",
        ),
        (
            "one_child_val_arr_alt",
            "(PTestParent @child_tuple=[(PTestParent) | (PTestChild1), (PTestParent) | (PTestChild1), *])",
        ),
        (
            "one_child_val_arr_alt_cap_inner",
            "(PTestParent @child_tuple=[(PTestParent) | (PTestChild1) -> child_cap, (*), *])",
        ),
        (
            "one_child_val_arr_alt_bound_cap_alt",
            "(PTestParent @child_tuple=[<(PTestParent) | (PTestChild1)> -> child_cap, (*), *])",
        ),
        (
            "one_child_val_arr_alt_bound_cap_parts",
            "(PTestParent @child_tuple=[<(PTestParent) -> child_cap | (PTestChild1) -> child_cap1>"
            " -> child_outer, (*), *])",
        ),
        (
            "one_child_val_arr_cap_all",
            "(PTestParent @child_tuple=[(*) -> child_cap, (*), * -> all_cap])",
        ),
        (
            "alternative_with_multi_cap",
            "(PTestParent @child_tuple=[(*) -> +child_cap, * -> all_cap]) | "
            "(PTestChild1 @foo -> +child_cap)",
        ),
        (
            "wildcards",
            "(PTestParent @child_tuple=[(PTestChild1)* -> cap]) | "
            "(PTestParent @child_tuple=[(PTestChild1)? -> cap]) | "
            "(PTestParent @child_tuple=[(PTestChild1)+ -> cap]) | "
            "(PTestParent @child_tuple=[(PTestChild1){5} -> cap]) | "
            "(PTestParent @child_tuple=[(PTestChild1){2,5} -> +capm])",
        ),
    ],
    ids=lambda p: f"test_pattern_{p}" if not p.startswith("(") else "",
)
def test_correct_pattern_grammar(rule: str, pattern_def: str) -> None:
    res, msg = validate_pattern(pattern_def)
    assert res, f"Error in rule {rule}: {msg}"


@pytest.mark.parametrize(
    "rule, pattern_def, expected_msg",
    [
        (
            "##empty",
            "",
            "Empty pattern definition. Make sure to include at least one tree pattern.",
        ),
        (
            "##empty_value",
            "(PTestChild1 @foo=)",
            "Expected one of: '[' (sequence start), '(' (pattern start), '<' (alternative boundary start)"
            ", '$' (variable indicator), a name (identifier), an escaped string (possibly a regex). "
            "Got: ')' (pattern end)",
        ),
        (
            "##seq_val_after_any",
            "(PTestParent @child_tuple=[* (PTestChild1)])",
            "Incorrect definition of sequence in a tree pattern.\n"
            "Expected: ']' (sequence end). Got: '(' (pattern start)",
        ),
        (
            "##any_and_multi_class",
            "(* | PTestChild1)",
            "Expected one of: ')' (pattern end), '@' (field indicator). Got: '|' (alternative separator)",
        ),
        (
            "##extra_rbracket",
            '(PTestParent @foo="val" -> same_val  @bar="barval" -> same_val])',
            "Expected one of: ')' (pattern end), '@' (field indicator). Got: ']' (sequence end)",
        ),
        (
            "##two_attr_var_before_cap",
            '(PTestParent @foo= $same_val  bar="barval" -> same_val])',
            "Pattern uses match variable <same_val> before it was captured",
        ),
        (
            "##mixed_cap_var_type",
            "(PTestParent @foo -> cap @bar -> +cap])",
            'Name <cap> is already used as a "single" type match variable.'
            " Using the same name to capture a single and multiple values is not allowed",
        ),
    ],
    # will effectively use the rule name as the test name
    ids=lambda p: f"test_pattern_{p[2:]}" if p.startswith("##") else "",
)
def test_incorrect_pattern_grammar(rule: str, pattern_def: str, expected_msg: str) -> None:
    res, msg = validate_pattern(pattern_def)
    assert not res
    assert expected_msg in msg


def test_props_match_and_capture(clean_ser_types) -> None:
    @dataclass
    class Node(ASTNode):
        foo: str
        bar: str | None
        baz: tuple[str | tuple[str, ...] | None, ...]

    node = Node("fooval", "barval", (), origin=NO_ORIGIN)
    test_content_id = node.content_id

    # Test match by content_id
    rule = f'(* @content_id="{test_content_id}")'
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values == {}, "Expected empty match dict"

    # allow any attributes
    rule = "(Node @foo -> foo_val)"
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values == {"foo_val": "fooval"}

    # allow any attributes, but mismatch attribute value
    rule = '(Node @foo = "other_val" -> foo_val)'
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert not ok, f"Matched, but wasn't supposed to: {rule}"
    assert values == {}, "Expected empty match dict"

    # test value alternatives
    rule = '(Node @foo = "other_val|f.o.*l" -> foo_val)'
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values == {"foo_val": "fooval"}

    # check non-existent attribute
    rule = "(Node @non_existent)"
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert not ok, f"Matched, but wasn't supposed to: {rule}"
    assert values == {}, "Expected empty match dict"

    # Test match by id
    rule = f'(* @id="{node.id}")'
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values == {}, "Expected empty match dict"

    # Test match None
    node = Node("fooval", None, (), origin=NO_ORIGIN)
    rule = "(* @bar=None -> bar_val)"
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values == {"bar_val": None}

    # Test match variable
    node = Node("fooval", "fooval", (), origin=NO_ORIGIN)
    rule = "(* @foo -> foo_val @bar=$foo_val)"
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values == {"foo_val": "fooval"}

    # Test sequence of scalars
    node = Node("fooval", "barval", ("bazval", "bazval2", None, "tail"), origin=NO_ORIGIN)
    rule = '(* @baz=["baz.*", ".*val2", None, * -> tail_val] -> baz_val)'
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values == {"tail_val": ("tail",), "baz_val": ("bazval", "bazval2", None, "tail")}

    # Test nested sequence of scalars
    node = Node("fooval", "barval", (("bazval", "bazval2"), ("start", "tail")), origin=NO_ORIGIN)
    rule = '(* @baz=[["baz.*", ".*val2"] -> nest1, ["start", * -> tail_val] -> nest2] -> baz_val)'
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values == {
        "tail_val": ("tail",),
        "baz_val": (("bazval", "bazval2"), ("start", "tail")),
        "nest1": ("bazval", "bazval2"),
        "nest2": ("start", "tail"),
    }


def test_child_field_match_and_capture() -> None:
    # Test none match
    node = PTestParent(origin=origin)

    matcher = BaseMatcher.from_pattern(
        "(PTestParent @child1=None @child_tuple=[] -> rule_all_none) | "
        "(PTestParent @child_tuple=[] -> rule_tuple_none) | "
        "(PTestParent @child1=None -> rule_first_none)"
    )

    ok, values = matcher.match(node)
    assert ok
    assert list(values.keys()) == ["rule_all_none"]

    node = node.replace(child1=PTestChild1("deep_1", origin=origin))
    ok, values = matcher.match(node)
    assert ok
    assert list(values.keys()) == ["rule_tuple_none"]

    node = node.replace(
        child1=None,
        child_tuple=(
            PTestChild1("deep_2", origin=origin),
            PTestChild1("deep_3", origin=origin),
        ),
    )
    ok, values = matcher.match(node)
    assert ok
    assert list(values.keys()) == ["rule_first_none"]

    sub1_ch1 = PTestChild1("deep_1", origin=origin)
    sub1 = PTestParent(
        child1=sub1_ch1,
        child_any=PTestChild2("deep_2", origin=origin),
        sub_parent=None,
        origin=origin,
    )

    sub2 = sub1.duplicate()

    node = PTestParent(
        child_tuple=(sub1, sub2),
        origin=origin,
    )

    rule = "(PTestParent @child_tuple -> test_capture)"
    matcher = BaseMatcher.from_pattern(rule)

    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values["test_capture"] == (sub1, sub2)

    node.detach()

    # Test array capture with follow rules
    f1 = PTestChild1("predecessor", origin=origin)
    f2 = f1.duplicate()
    only1 = PTestChild2("only1", origin=origin)

    node = PTestParent(
        child_tuple=(f1, f2, only1, sub1, sub2),
        origin=origin,
    )

    rule = "(PTestParent @child_tuple =[(PTestChild1), (*), (PTestChild2) -> only_capture, *])"
    matcher = BaseMatcher.from_pattern(rule)

    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values["only_capture"] == only1

    # Match any remaining
    rule = "(PTestParent @child_tuple =[(*) -> first, * -> remaining])"
    matcher = BaseMatcher.from_pattern(rule)

    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values["first"] == f1
    assert values["remaining"] == (f2, only1, sub1, sub2)


def test_var_match() -> None:
    f1 = PTestChild1("predecessor", origin=origin)
    f2 = f1.duplicate()  # not the same id, but same content
    only1 = PTestChild2("tail", origin=origin)

    node = PTestParent(
        child_tuple=(f1, f2, only1),
        origin=origin,
    )

    macher = BaseMatcher.from_pattern(
        "(PTestParent @child_tuple=[(* @foo -> cap_foo) -> cap, $cap, *])"
    )

    ok, match_dict = macher.match(node)
    assert ok
    assert match_dict["cap"] == f1
    assert match_dict["cap_foo"] == "predecessor"
    assert set(match_dict.keys()) == {"cap", "cap_foo"}

    # Now make sure capture variable context is correctly scoped
    node.detach_self()
    f3 = f1.duplicate()
    f4 = f1.duplicate()

    node = PTestParent(child1=f1, child_any=f2, child_tuple=(f3, f4), origin=origin)

    # First, multiple captures should be preserved
    rule = "(PTestParent @child1 -> ch1 @child_any -> ch2 @child_tuple=[$ch1, $ch2])"
    macher = BaseMatcher.from_pattern(rule)
    ok, match_dict = macher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert match_dict["ch1"] is f1
    assert match_dict["ch2"] is f2
    assert set(match_dict.keys()) == {"ch1", "ch2"}

    # Second, in alternatives, only the matching capture should be preserved
    rule = '(PTestParent @child1=<(PTestChild1 @foo="no match") | (PTestChild1 @foo="predecessor")> -> ch1 @child_any -> ch2 @child_tuple=[$ch1, $ch2] -> ch_tup)'
    macher = BaseMatcher.from_pattern(rule)
    ok, match_dict = macher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert match_dict["ch1"] is f1
    assert match_dict["ch2"] is f2
    assert match_dict["ch_tup"] == (f3, f4)
    assert set(match_dict.keys()) == {"ch1", "ch2", "ch_tup"}


@pytest.mark.parametrize(
    "rule, matched",
    [
        # Greedy match will consume all children, so this should not match
        ("(* @child_tuple=[(*)*, (*)])", False),
        ("(* @child_tuple=[(*)+, (*)])", False),
        ("(* @child_tuple=[(*){1,10}, (*)])", False),
        # Our special "tail" wildcard allows empty tail
        ("(* @child_tuple=[(*)*, *])", True),
        # Now more interesting cases
        ('(* @child_tuple=[(*){3}, (* @foo="ch4"), *])', True),
        ('(* @child_tuple=[(*){3}, (* @foo="ch5"), *])', False),
        ('(* @child_tuple=[(*){3}, (* @foo="ch5")?, *])', True),
        ('(* @child_tuple=[(* @foo="ch1")?, (*){9}, *])', True),
    ],
)
def test_wildcard_match(rule: str, matched: bool) -> None:
    children = [PTestChild1(f"ch{i+1}", origin=NO_ORIGIN) for i in range(10)]

    node = PTestParent(
        child_tuple=tuple(children),
        origin=NO_ORIGIN,
    )

    macher = BaseMatcher.from_pattern(rule)
    ok, match_dict = macher.match(node)
    assert ok == matched, f"Didn't match: {rule}"
    assert match_dict == {}, "Expected empty match dict"


def test_multi_capture() -> None:
    f1 = PTestChild1("predecessor", origin=origin)
    f2 = f1.duplicate()  # not the same id, but same content
    only1 = PTestChild2("tail", origin=origin)

    node = PTestParent(
        child_tuple=(f1, f2, only1),
        origin=origin,
    )

    macher = BaseMatcher.from_pattern("(PTestParent @child_tuple=[(*) -> +cap, (*), (*) -> +cap])")

    ok, match_dict = macher.match(node)
    assert ok
    assert match_dict == {"cap": (f1, only1)}

    # Test wildcards extending the captured sequence
    for rule in [
        "(* @child_tuple=[* -> +cap])",
        "(* @child_tuple=[(*) -> +cap, * -> +cap])",
        "(* @child_tuple=[(*) -> +cap, (*)* -> +cap])",
        "(* @child_tuple=[(*)* -> +cap])",
        "(* @child_tuple=[(*) -> +cap, (*)? -> +cap, (*)? -> +cap])",
        "(* @child_tuple=[(*) -> +cap, (*)+ -> +cap])",
        "(* @child_tuple=[(*) -> +cap, (*){2} -> +cap])",
        "(* @child_tuple=[(*) -> +cap, (*){1,3} -> +cap])",
    ]:
        macher = BaseMatcher.from_pattern(rule)
        ok, match_dict = macher.match(node)
        assert ok
        assert match_dict == {"cap": (f1, f2, only1)}

    # Now make sure capture variable context is correctly scoped
    # i.e. discarded alternatives do not pollute the match dict
    macher = BaseMatcher.from_pattern(
        # First alternative will partially match, but will be discarded
        "(PTestParent @child_tuple=[(*) -> +cap, (*) -> cap1]) |"
        "(PTestParent @child_tuple=[(*) -> +cap, (*), (*) -> +cap])"
    )

    ok, match_dict = macher.match(node)
    assert ok
    assert match_dict == {"cap": (f1, only1)}

    # Test with capturing at multiple levels
    macher = BaseMatcher.from_pattern(
        "(PTestParent @child_tuple=[(* @foo -> +cap) -> +cap, * -> +cap])"
    )

    ok, match_dict = macher.match(node)
    assert ok
    assert match_dict == {"cap": ("predecessor", f1, f2, only1)}


def test_empty_sequence_match() -> None:
    node = PTestParent(
        child_tuple=(),
        origin=origin,
    )

    macher = BaseMatcher.from_pattern("(PTestParent @child_tuple=[(PTestChild1) -> cap, *])")

    ok, match_dict = macher.match(node)
    assert not ok
    assert not match_dict

    macher = BaseMatcher.from_pattern("(PTestParent @child_tuple=[* -> cap_empty_seq])")

    ok, match_dict = macher.match(node)
    assert ok
    assert match_dict["cap_empty_seq"] == ()

    macher = BaseMatcher.from_pattern("(PTestParent @child_tuple=[] -> cap_empty_node)")

    ok, match_dict = macher.match(node)
    assert ok
    assert match_dict["cap_empty_node"] == ()


def test_same_name_capture() -> None:
    # If the same capture name used twice, the last one should be used
    ch1 = PTestChild1("deep_1", origin=origin)
    ch2 = PTestChild1("deep_2", origin=origin)
    node = PTestParent(
        child_tuple=(
            ch1,
            ch2,
        ),
        origin=origin,
    )

    rule = "(PTestParent @child_tuple=[(PTestChild1 @foo -> cap), (PTestChild1 @foo -> cap)])"
    matcher = BaseMatcher.from_pattern(rule)

    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values == {"cap": "deep_2"}

    # But in alternative, the one that matches should be used
    cnode = PTestChild1("deep_1", origin=origin)

    rule = '(PTestChild1 @foo=".*_1" -> cap) | (PTestChild1 @foo=".*_2" -> cap)'
    matcher = BaseMatcher.from_pattern(rule)

    ok, values = matcher.match(cnode)
    assert ok, f"Didn't match: {rule}"
    assert values == {"cap": "deep_1"}

    # the outer use of the capture name "wins"
    rule = "(PTestParent @child_tuple=[(PTestChild1 @foo -> cap), *] -> cap)"
    macher = BaseMatcher.from_pattern(rule)
    ok, match_dict = macher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert match_dict == {"cap": (ch1, ch2)}

    # Same but with top level capture
    rule = "(PTestChild1 @foo -> cap) | (*) -> cap"
    macher = BaseMatcher.from_pattern(rule)
    ok, match_dict = macher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert match_dict["cap"] is node
    assert set(match_dict.keys()) == {"cap"}
