from __future__ import annotations

from dataclasses import dataclass, field

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
class PTestChildMultiAttrs(ASTNode):
    foo: str
    bar: str
    baz: str
    zaz: str | None


@pytest.mark.parametrize(
    "rule, pattern_def",
    [
        ("any_class", "(*)"),
        ("class_only", "(PTestParent)"),
        ("multi_class", "(PTestParent | PTestChild1)"),
        ("alternative", "(PTestParent) | (PTestChild1)"),
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
            "(PTestParent @child_tuple=[(*) (*), *] -> child_cap)",
        ),
        (
            "one_child_val_arr_cap_inner",
            "(PTestParent @child_tuple=[(*) -> child_cap (*) *])",
        ),
        (
            "one_child_val_arr_alt",
            "(PTestParent @child_tuple=[(PTestParent) | (PTestChild1), (PTestParent) | (PTestChild1) *])",
        ),
        (
            "one_child_val_arr_alt_cap_inner",
            "(PTestParent @child_tuple=[(PTestParent) | (PTestChild1) -> child_cap (*) *])",
        ),
        (
            "one_child_val_arr_cap_all",
            "(PTestParent @child_tuple=[(*) -> child_cap (*) * -> all_cap])",
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
            "Expected: '(' (pattern start), got: end of text",
        ),
        (
            "##empty_value",
            "(PTestChild1 @foo=)",
            "Expected: '[' (sequence start), '(' (pattern start), '$' (variable indicator),"
            " a class name, an escaped string (possibly a regex), got: ')' (pattern end)",
        ),
        (
            "##any_and_multi_class",
            "(* | PTestChild1)",
            "Expected: ')' (pattern end), got: '|' (alternative separator)",
        ),
        (
            "##two_attr_cap_one_name",
            '(PTestParent @foo="val" -> same_val  @bar="barval" -> same_val])',
            "Capture name <same_val> used more than once",
        ),
        (
            "##two_attr_var_before_cap",
            '(PTestParent @foo= $same_val  bar="barval" -> same_val])',
            "Pattern uses match variable <same_val> before it was captured",
        ),
    ],
    # will effectively use the rule name as the test name
    ids=lambda p: f"test_pattern_{p[2:]}" if p.startswith("##") else "",
)
def test_incorrect_pattern_grammar(rule: str, pattern_def: str, expected_msg: str) -> None:
    res, msg = validate_pattern(pattern_def)
    assert not res
    assert expected_msg in msg


def test_props_match() -> None:
    node = PTestChildMultiAttrs("fooval", "barval", "bazval", "zazval", origin=origin)
    test_content_id = node.content_id

    # Test match by content_id
    rule = f'(* @content_id="{test_content_id}")'
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values == {}

    # allow any attributes
    rule = "(PTestChildMultiAttrs @foo -> foo_val)"
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values["foo_val"] == "fooval"

    # allow any attributes, but mismatch attribute value
    rule = '(PTestChildMultiAttrs @foo = "other_val" -> foo_val)'
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert not ok, f"Matched, but wasn't supposed to: {rule}"

    # test value alternatives
    rule = '(PTestChildMultiAttrs @foo = "other_val|f.o.*l" -> foo_val)'
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values["foo_val"] == "fooval"

    # check non-existent attribute
    rule = "(PTestChildMultiAttrs @non_existent)"
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert not ok, f"Matched, but wasn't supposed to: {rule}"

    # Test match by id
    rule = f'(* @id="{node.id}")'
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"

    # Test match None
    node = PTestChildMultiAttrs("fooval", "barval", "bazval", None, origin=origin)
    rule = "(* @zaz=None -> zaz_val)"
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values["zaz_val"] is None

    # Test match variable
    node = PTestChildMultiAttrs("fooval", "fooval", "bazval", None, origin=origin)
    rule = "(* @foo -> foo_val @bar=$foo_val)"
    matcher = BaseMatcher.from_pattern(rule)
    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values["foo_val"] == "fooval"


def test_attr_value_capture() -> None:
    matcher = BaseMatcher.from_pattern("(PTestParent @child1=(PTestChild1 @foo -> test_capture))")
    node = PTestParent(
        child1=PTestChild1("capture_value", origin=origin),
        child_any=PTestChild2("non_capture", origin=origin),
        origin=origin,
    )

    ok, values = matcher.match(node)
    assert ok
    assert values["test_capture"] == "capture_value"


def test_child_field_match() -> None:
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


def test_var_match() -> None:
    f1 = PTestChild1("predecessor", origin=origin)
    f2 = f1.duplicate()  # not the same id, but same content
    only1 = PTestChild2("tail", origin=origin)

    node = PTestParent(
        child_tuple=(f1, f2, only1),
        origin=origin,
    )

    macher = BaseMatcher.from_pattern(
        "(PTestParent @child_tuple=[(* @foo -> cap_foo) -> cap $cap *])"
    )

    ok, match_dict = macher.match(node)
    assert ok
    assert match_dict["cap"] == f1
    assert match_dict["cap_foo"] == "predecessor"

    # Now make sure capture variable context is correctly scoped
    node.detach_self()
    f3 = f1.duplicate()
    f4 = f1.duplicate()

    node = PTestParent(child1=f1, child_any=f2, child_tuple=(f3, f4), origin=origin)

    # First, multiple captures should be preserved
    rule = "(PTestParent @child1 -> ch1 @child_any -> ch2 @child_tuple=[$ch1 $ch2])"
    macher = BaseMatcher.from_pattern(rule)
    ok, match_dict = macher.match(node)
    assert ok, f"Didn't match: {rule}"

    # Second, in alternatives, only the matching capture should be preserved
    rule = '(PTestParent @child1=(PTestChild1 @foo="no match") | (PTestChild1 @foo="predecessor") -> ch1 @child_any -> ch2 @child_tuple=[$ch1 $ch2] -> ch_tup)'
    macher = BaseMatcher.from_pattern(rule)
    ok, match_dict = macher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert match_dict["ch1"] is f1
    assert match_dict["ch2"] is f2
    assert match_dict["ch_tup"] == (f3, f4)


def test_sequence_empty_match() -> None:
    node = PTestParent(
        child_tuple=(),
        origin=origin,
    )

    macher = BaseMatcher.from_pattern("(PTestParent @child_tuple=[(PTestChild1) -> cap *])")

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


def test_child_spec_capture() -> None:
    sub1 = PTestParent(
        child1=PTestChild1("deep_1", origin=origin),
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

    rule = "(PTestParent @child_tuple =[(PTestChild1) (*) (PTestChild2) -> only_capture *])"
    matcher = BaseMatcher.from_pattern(rule)

    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values["only_capture"] == only1

    # Match any remaining
    rule = "(PTestParent @child_tuple =[(*) -> first * -> remaining])"
    matcher = BaseMatcher.from_pattern(rule)

    ok, values = matcher.match(node)
    assert ok, f"Didn't match: {rule}"
    assert values["first"] == f1
    assert values["remaining"] == (f2, only1, sub1, sub2)
