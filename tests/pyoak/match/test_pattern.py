from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pyoak.match.pattern import MultiPatternMatcher, NodeMatcher, validate_pattern
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
    "rule, pattern_def",
    [
        ("any_and_multi_class", "(* | PTestChild1)"),
        (
            "two_attr_cap_one_name",
            '(PTestParent @foo="val" -> same_val  bar="barval" -> same_val])',
        ),
        (
            "two_attr_var_before_cap",
            '(PTestParent @foo= $same_val  bar="barval" -> same_val])',
        ),
    ],
    ids=lambda p: f"test_pattern_{p}" if not p.startswith("(") else "",
)
def test_incorrect_pattern_grammar(rule: str, pattern_def: str) -> None:
    res, _ = validate_pattern(pattern_def)
    assert not res


def test_props_match() -> None:
    node = PTestChildMultiAttrs("fooval", "barval", "bazval", "zazval", origin=origin)
    test_content_id = node.content_id

    # Test match by content_id
    matcher = MultiPatternMatcher([("rule", f'(* @content_id="{test_content_id}")')])
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values == {}

    # allow any attributes
    matcher = MultiPatternMatcher([("rule", "(PTestChildMultiAttrs @foo -> foo_val)")])
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["foo_val"] == "fooval"

    # allow any attributes, but mismatch attribute value
    matcher = MultiPatternMatcher(
        [("rule", '(PTestChildMultiAttrs @foo = "other_val" -> foo_val)')]
    )
    match = matcher.match(node)
    assert match is None

    # test value alternatives
    matcher = MultiPatternMatcher(
        [("rule", '(PTestChildMultiAttrs @foo = "other_val|f.o.*l" -> foo_val)')]
    )
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["foo_val"] == "fooval"

    # check non-existent attribute
    matcher = MultiPatternMatcher(
        [
            (
                "rule",
                "(PTestChildMultiAttrs @non_existent)",
            )
        ]
    )
    match = matcher.match(node)
    assert match is None

    # Test match by id
    matcher = MultiPatternMatcher([("rule", f'(* @id="{node.id}")')])
    match = matcher.match(node)
    assert match is not None

    # Test match None
    node = PTestChildMultiAttrs("fooval", "barval", "bazval", None, origin=origin)
    matcher = MultiPatternMatcher([("rule", "(* @zaz=None -> zaz_val)")])
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["zaz_val"] is None

    # Test match variable
    node = PTestChildMultiAttrs("fooval", "fooval", "bazval", None, origin=origin)
    matcher = MultiPatternMatcher([("rule", "(* @foo -> foo_val @bar=$foo_val)")])
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["foo_val"] == "fooval"


def test_attr_value_capture() -> None:
    rule = "(PTestParent @child1=(PTestChild1 @foo -> test_capture))"
    matcher = MultiPatternMatcher([("rule", rule)])
    node = PTestParent(
        child1=PTestChild1("capture_value", origin=origin),
        child_any=PTestChild2("non_capture", origin=origin),
        origin=origin,
    )

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["test_capture"] == "capture_value"


def test_child_field_match() -> None:
    # Test none match
    node = PTestParent(origin=origin)

    matcher = MultiPatternMatcher(
        [
            ("rule_all_none", "(PTestParent @child1=None @child_tuple=[])"),
            ("rule_first_none", "(PTestParent @child1=None)"),
            ("rule_second_none", "(PTestParent @child_tuple=[])"),
        ]
    )

    match = matcher.match(node)
    assert match is not None
    assert match[0] == "rule_all_none"

    node = node.replace(child1=PTestChild1("deep_1", origin=origin))
    match = matcher.match(node)
    assert match is not None
    assert match[0] == "rule_second_none"

    node = node.replace(
        child1=None,
        child_tuple=(
            PTestChild1("deep_2", origin=origin),
            PTestChild1("deep_3", origin=origin),
        ),
    )
    match = matcher.match(node)
    assert match is not None
    assert match[0] == "rule_first_none"


def test_sequence_var_match() -> None:
    f1 = PTestChild1("predecessor", origin=origin)
    f2 = f1.duplicate()  # not the same id, but same content
    only1 = PTestChild2("tail", origin=origin)

    node = PTestParent(
        child_tuple=(f1, f2, only1),
        origin=origin,
    )

    macher, msg = NodeMatcher.from_pattern(
        "(PTestParent @child_tuple=[(* @foo -> cap_foo) -> cap $cap *])"
    )
    assert macher is not None, msg

    ok, match_dict = macher.match(node)
    assert ok
    assert match_dict["cap"] == f1
    assert match_dict["cap_foo"] == "predecessor"


def test_sequence_empty_match() -> None:
    node = PTestParent(
        child_tuple=(),
        origin=origin,
    )

    macher, msg = NodeMatcher.from_pattern("(PTestParent @child_tuple=[(PTestChild1) -> cap *])")
    assert macher is not None, msg

    ok, match_dict = macher.match(node)
    assert not ok
    assert not match_dict

    macher, msg = NodeMatcher.from_pattern("(PTestParent @child_tuple=[* -> cap_empty_seq])")
    assert macher is not None, msg

    ok, match_dict = macher.match(node)
    assert ok
    assert match_dict["cap_empty_seq"] == ()

    macher, msg = NodeMatcher.from_pattern("(PTestParent @child_tuple=[] -> cap_empty_node)")
    assert macher is not None, msg

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
    matcher = MultiPatternMatcher([("rule", rule)])

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
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
    matcher = MultiPatternMatcher([("rule", rule)])

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["only_capture"] == only1

    # Match any remaining
    rule = "(PTestParent @child_tuple =[(*) -> first * -> remaining])"
    matcher = MultiPatternMatcher([("rule", rule)])

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["first"] == f1
    assert values["remaining"] == (f2, only1, sub1, sub2)
