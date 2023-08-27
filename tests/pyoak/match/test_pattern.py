from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest
from pyoak.match.pattern import PatternMatcher
from pyoak.node import ASTNode


@dataclass(frozen=True)
class PTestChild1(ASTNode):
    foo: str


@dataclass(frozen=True)
class PTestChild2(ASTNode):
    bar: str


@dataclass(frozen=True)
class PTestParent(ASTNode):
    foo: str = "foo_val"
    bar: str = "bar_val"
    child1: PTestChild1 | None = None
    child_any: PTestChild2 | PTestChild1 | None = None
    sub_parent: PTestParent | None = None
    child_tuple: tuple[ASTNode, ...] = field(default_factory=tuple)
    child_tuple2: tuple[ASTNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PTestChildMultiAttrs(ASTNode):
    foo: str
    bar: str
    baz: str
    zaz: str


DEFAULT_ATTRS = {
    "id",
    "content_id",
    "origin",
    "original_id",
    "id_collision_with",
}

DEFAULT_ATTRS_STR = " ".join(DEFAULT_ATTRS)


@pytest.mark.parametrize(
    "rule, pattern_def",
    [
        ("wildcard", "(*)"),
        ("class_only", "(PTestParent)"),
        ("one_attr_any_val", "(PTestParent #[foo])"),
        ("one_attr_any_val_cap", "(PTestParent #[foo -> foo_val])"),
        ("one_attr_val", '(PTestParent #[foo="val"])'),
        ("one_attr_val_cap", '(PTestParent #[foo="val" -> foo_val])'),
        ("one_attr_any_val_only", "(PTestParent #[foo !])"),
        ("one_attr_any_val_cap_only", "(PTestParent #[foo -> foo_val !])"),
        ("one_attr_val_only", '(PTestParent #[foo="val" !])'),
        ("one_attr_val_cap_only", '(PTestParent #[foo="val" -> foo_val !])'),
        ("one_attr_any_val_all_in_order", "(PTestParent #[foo !!])"),
        ("one_attr_any_val_cap_all_in_order", "(PTestParent #[foo -> foo_val !!])"),
        ("one_attr_val_all_in_order", '(PTestParent #[foo="val" !!])'),
        ("one_attr_val_cap_all_in_order", '(PTestParent #[foo="val" -> foo_val !!])'),
        ("two_attr_any_val_sec_any", "(PTestParent #[foo bar])"),
        ("two_attr_any_val_cap_sec_any", "(PTestParent #[foo -> foo_val bar])"),
        ("two_attr_val_sec_any", '(PTestParent #[foo="val" bar])'),
        ("two_attr_val_cap_sec_any", '(PTestParent #[foo="val" -> foo_val bar])'),
        ("two_attr_any_val_sec_val", '(PTestParent #[foo bar="barval"])'),
        (
            "two_attr_any_val_cap_sec_val",
            '(PTestParent #[foo -> foo_val  bar="barval"])',
        ),
        ("two_attr_val_sec_val", '(PTestParent #[foo="val"  bar="barval"])'),
        (
            "two_attr_val_cap_sec_val",
            '(PTestParent #[foo="val" -> foo_val  bar="barval"])',
        ),
        (
            "two_attr_any_val_sec_val_cap",
            '(PTestParent #[foo bar="barval" -> bar_val])',
        ),
        (
            "two_attr_any_val_cap_sec_val_cap",
            '(PTestParent #[foo -> foo_val  bar="barval" -> bar_val])',
        ),
        (
            "two_attr_val_sec_val_cap",
            '(PTestParent #[foo="val"  bar="barval" -> bar_val])',
        ),
        (
            "two_attr_val_cap_sec_val_cap",
            '(PTestParent #[foo="val" -> foo_val  bar="barval" -> bar_val])',
        ),
        (
            "two_attr_any_val_sec_val_cap_all",
            '(PTestParent #[foo bar="barval" -> bar_val !!])',
        ),
        (
            "two_attr_any_val_cap_sec_val_cap_all",
            '(PTestParent #[foo -> foo_val  bar="barval" -> bar_val !!])',
        ),
        (
            "two_attr_val_sec_val_cap_all",
            '(PTestParent #[foo="val"  bar="barval" -> bar_val !!])',
        ),
        (
            "two_attr_val_cap_sec_val_cap_all",
            '(PTestParent #[foo="val" -> foo_val  bar="barval" -> bar_val !!])',
        ),
        (
            "two_attr_cap_one_name",
            '(PTestParent #[foo="val" -> same_val  bar="barval" -> same_val])',
        ),
        ("one_child_any", "(PTestParent @[child1])"),
        ("one_child_any_cap", "(PTestParent @[child1 -> child_cap])"),
        ("one_child_val_any", "(PTestParent @[child1=(*)])"),
        ("one_child_val_any_cap", "(PTestParent @[child1=(*) -> child_cap])"),
        ("one_child_val_none", "(PTestParent @[child1=None])"),
        ("one_child_val_empty", "(PTestParent @[child_tuple=[]])"),
        ("one_child_val_none_cap", "(PTestParent @[child1=None -> child_cap])"),
        ("one_child_val_empty_cap", "(PTestParent @[child_tuple=[] -> child_cap])"),
        ("one_child_val_arr", "(PTestParent @[child_tuple=[(*), (*) !!, *]])"),
        (
            "one_child_val_arr_cap",
            "(PTestParent @[child_tuple=[(*), (*) !!, *] -> child_cap])",
        ),
        (
            "one_child_val_arr_cap_only",
            "(PTestParent @[child_tuple=[(*), (*) !!, *] -> child_cap !])",
        ),
        (
            "one_child_val_arr_cap_all",
            "(PTestParent @[child_tuple=[(*), (*) !!, *] -> child_cap !!])",
        ),
        ("two_child_any", "(PTestParent @[child1 child_any])"),
        ("two_child_any_cap", "(PTestParent @[child1 -> child_cap child_any])"),
        (
            "two_child_any_cap_both",
            "(PTestParent @[child1 -> child_cap child_any -> child_any_cap])",
        ),
        (
            "two_child_any_cap_same",
            "(PTestParent @[child1 -> child_cap child_any -> child_cap])",
        ),
        ("two_child_val_any", "(PTestParent @[child1=(*) child_any=(*)])"),
        (
            "two_child_val_any_cap",
            "(PTestParent @[child1=(*) -> child_cap child_any=(*)])",
        ),
        ("two_child_val_none", "(PTestParent @[child1=None child_any=None])"),
        ("two_child_val_empty", "(PTestParent @[child_tuple=[] child_tuple2=[]])"),
        (
            "two_child_val_none_cap",
            "(PTestParent @[child1=None -> child_cap child_any=None])",
        ),
        (
            "two_child_cap_arr_any",
            "(PTestParent @[child_tuple=[(*) -> first, * -> remaining] child_tuple2=[]])",
        ),
    ],
    ids=lambda p: f"test_pattern_{p}" if not p.startswith("(") else "",
)
def test_correct_pattern_grammar(rule: str, pattern_def: str) -> None:
    res, msg = PatternMatcher.validate_pattern(pattern_def)
    assert res, f"Error in rule {rule}: {msg}"


def test_attr_checks() -> None:
    node = PTestChildMultiAttrs("fooval", "barval", "bazval", "zazval")
    test_content_id = node.content_id

    # Test match by content_id
    matcher = PatternMatcher([("rule", f'(* #[content_id="{test_content_id}"])')])
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values == {}

    # allow any attributes
    matcher = PatternMatcher([("rule", "(PTestChildMultiAttrs #[foo -> foo_val])")])
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["foo_val"] == ["fooval"]

    # allow any attributes, but mismatch attribute value
    matcher = PatternMatcher([("rule", '(PTestChildMultiAttrs #[foo = "other_val" -> foo_val])')])
    match = matcher.match(node)
    assert match is None

    # test value alternatives
    matcher = PatternMatcher(
        [("rule", '(PTestChildMultiAttrs #[foo = ("other_val"|"f.o.*l") -> foo_val])')]
    )
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["foo_val"] == ["fooval"]

    # check ONLY mode (must fail because besides foo, there are other attributes including default ones)
    matcher = PatternMatcher([("rule", "(PTestChildMultiAttrs #[foo -> foo_val !])")])
    match = matcher.match(node)
    assert match is None

    # check only with extra non existant attribute. Must allow, because only doesn't check for extra attributes
    matcher = PatternMatcher(
        [
            (
                "rule",
                f"(PTestChildMultiAttrs #[{DEFAULT_ATTRS_STR} foo -> foo_val bar baz zaz extra !])",
            )
        ]
    )
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["foo_val"] == ["fooval"]

    # Check ALL, must fail because baz is not present
    matcher = PatternMatcher(
        [
            (
                "rule",
                f"(PTestChildMultiAttrs #[{DEFAULT_ATTRS_STR} foo -> foo_val bar baz zaz non_existent !!])",
            )
        ]
    )
    match = matcher.match(node)
    assert match is None

    # Test match by id
    matcher = PatternMatcher([("rule", f'(* #[id="{node.id}"])')])
    match = matcher.match(node)
    assert match is not None


def test_attr_value_capture() -> None:
    rule = "(PTestParent @[child1=(PTestChild1 #[foo -> test_capture])])"
    matcher = PatternMatcher([("rule", rule)])
    node = PTestParent(
        child1=PTestChild1("capture_value"),
        child_any=PTestChild2("non_capture"),
    )

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["test_capture"] == ["capture_value"]


def test_child_field_match() -> None:
    # Test none match
    node = PTestParent()

    matcher = PatternMatcher(
        [
            ("rule_all_none", "(PTestParent @[child1=None child_tuple=[]])"),
            ("rule_first_none", "(PTestParent @[child1=None])"),
            ("rule_second_none", "(PTestParent @[child_tuple=[]])"),
        ]
    )

    match = matcher.match(node)
    assert match is not None
    assert match[0] == "rule_all_none"

    node = replace(node, child1=PTestChild1("deep_1"))
    match = matcher.match(node)
    assert match is not None
    assert match[0] == "rule_second_none"

    node = replace(
        node,
        child1=None,
        child_tuple=(
            PTestChild1("deep_2"),
            PTestChild1("deep_3"),
        ),
    )
    match = matcher.match(node)
    assert match is not None
    assert match[0] == "rule_first_none"


def test_child_spec_capture() -> None:
    sub1 = PTestParent(
        child1=PTestChild1("deep_1"),
        child_any=PTestChild2("deep_2"),
        sub_parent=None,
    )

    sub2 = sub1.duplicate()

    node = PTestParent(
        child_tuple=(sub1, sub2),
    )

    rule = "(PTestParent @[child_tuple -> test_capture])"
    matcher = PatternMatcher([("rule", rule)])

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["test_capture"] == [sub1, sub2]

    # Test array capture with follow rules
    f1 = PTestChild1("predecessor")
    f2 = f1.duplicate()
    only1 = PTestChild2("only1")

    node = PTestParent(
        child_tuple=(f1, f2, only1, sub1, sub2),
    )

    rule = "(PTestParent @[child_tuple =[(PTestChild1), (PTestChild2)~1 -> only_capture, *]])"
    matcher = PatternMatcher([("rule", rule)])

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["only_capture"] == [only1]

    # Match any remaining
    rule = "(PTestParent @[child_tuple =[(*) !! -> first, * -> remaining]])"
    matcher = PatternMatcher([("rule", rule)])

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["first"] == [f1]
    assert values["remaining"] == [f2, only1, sub1, sub2]

    # Capture into same name
    rule = "(PTestParent @[child_tuple =[(PTestChild1)~2 -> multi_cap, (PTestChild2)~1 -> multi_cap, *]])"
    matcher = PatternMatcher([("rule", rule)])

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["multi_cap"] == [f1, f2, only1]
