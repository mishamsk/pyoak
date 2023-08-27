from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pyoak.legacy.match.pattern import PatternMatcher
from pyoak.legacy.node import AwareASTNode as ASTNode
from pyoak.origin import NoOrigin

origin = NoOrigin()


@dataclass
class LegacyPTestChild1(ASTNode):
    foo: str


@dataclass
class LegacyPTestChild2(ASTNode):
    bar: str


@dataclass
class LegacyPTestParent(ASTNode):
    foo: str = "foo_val"
    bar: str = "bar_val"
    child1: LegacyPTestChild1 | None = None
    child_any: LegacyPTestChild2 | LegacyPTestChild1 | None = None
    sub_parent: LegacyPTestParent | None = None
    child_tuple: tuple[ASTNode, ...] = field(default_factory=tuple)
    child_tuple2: tuple[ASTNode, ...] = field(default_factory=tuple)


@dataclass
class LegacyPTestChildMultiAttrs(ASTNode):
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
        ("class_only", "(LegacyPTestParent)"),
        ("one_attr_any_val", "(LegacyPTestParent #[foo])"),
        ("one_attr_any_val_cap", "(LegacyPTestParent #[foo -> foo_val])"),
        ("one_attr_val", '(LegacyPTestParent #[foo="val"])'),
        ("one_attr_val_cap", '(LegacyPTestParent #[foo="val" -> foo_val])'),
        ("one_attr_any_val_only", "(LegacyPTestParent #[foo !])"),
        ("one_attr_any_val_cap_only", "(LegacyPTestParent #[foo -> foo_val !])"),
        ("one_attr_val_only", '(LegacyPTestParent #[foo="val" !])'),
        ("one_attr_val_cap_only", '(LegacyPTestParent #[foo="val" -> foo_val !])'),
        ("one_attr_any_val_all_in_order", "(LegacyPTestParent #[foo !!])"),
        ("one_attr_any_val_cap_all_in_order", "(LegacyPTestParent #[foo -> foo_val !!])"),
        ("one_attr_val_all_in_order", '(LegacyPTestParent #[foo="val" !!])'),
        ("one_attr_val_cap_all_in_order", '(LegacyPTestParent #[foo="val" -> foo_val !!])'),
        ("two_attr_any_val_sec_any", "(LegacyPTestParent #[foo bar])"),
        ("two_attr_any_val_cap_sec_any", "(LegacyPTestParent #[foo -> foo_val bar])"),
        ("two_attr_val_sec_any", '(LegacyPTestParent #[foo="val" bar])'),
        ("two_attr_val_cap_sec_any", '(LegacyPTestParent #[foo="val" -> foo_val bar])'),
        ("two_attr_any_val_sec_val", '(LegacyPTestParent #[foo bar="barval"])'),
        (
            "two_attr_any_val_cap_sec_val",
            '(LegacyPTestParent #[foo -> foo_val  bar="barval"])',
        ),
        ("two_attr_val_sec_val", '(LegacyPTestParent #[foo="val"  bar="barval"])'),
        (
            "two_attr_val_cap_sec_val",
            '(LegacyPTestParent #[foo="val" -> foo_val  bar="barval"])',
        ),
        (
            "two_attr_any_val_sec_val_cap",
            '(LegacyPTestParent #[foo bar="barval" -> bar_val])',
        ),
        (
            "two_attr_any_val_cap_sec_val_cap",
            '(LegacyPTestParent #[foo -> foo_val  bar="barval" -> bar_val])',
        ),
        (
            "two_attr_val_sec_val_cap",
            '(LegacyPTestParent #[foo="val"  bar="barval" -> bar_val])',
        ),
        (
            "two_attr_val_cap_sec_val_cap",
            '(LegacyPTestParent #[foo="val" -> foo_val  bar="barval" -> bar_val])',
        ),
        (
            "two_attr_any_val_sec_val_cap_all",
            '(LegacyPTestParent #[foo bar="barval" -> bar_val !!])',
        ),
        (
            "two_attr_any_val_cap_sec_val_cap_all",
            '(LegacyPTestParent #[foo -> foo_val  bar="barval" -> bar_val !!])',
        ),
        (
            "two_attr_val_sec_val_cap_all",
            '(LegacyPTestParent #[foo="val"  bar="barval" -> bar_val !!])',
        ),
        (
            "two_attr_val_cap_sec_val_cap_all",
            '(LegacyPTestParent #[foo="val" -> foo_val  bar="barval" -> bar_val !!])',
        ),
        (
            "two_attr_cap_one_name",
            '(LegacyPTestParent #[foo="val" -> same_val  bar="barval" -> same_val])',
        ),
        ("one_child_any", "(LegacyPTestParent @[child1])"),
        ("one_child_any_cap", "(LegacyPTestParent @[child1 -> child_cap])"),
        ("one_child_val_any", "(LegacyPTestParent @[child1=(*)])"),
        ("one_child_val_any_cap", "(LegacyPTestParent @[child1=(*) -> child_cap])"),
        ("one_child_val_none", "(LegacyPTestParent @[child1=None])"),
        ("one_child_val_empty", "(LegacyPTestParent @[child_tuple=[]])"),
        ("one_child_val_none_cap", "(LegacyPTestParent @[child1=None -> child_cap])"),
        ("one_child_val_empty_cap", "(LegacyPTestParent @[child_tuple=[] -> child_cap])"),
        ("one_child_val_arr", "(LegacyPTestParent @[child_tuple=[(*), (*) !!, *]])"),
        (
            "one_child_val_arr_cap",
            "(LegacyPTestParent @[child_tuple=[(*), (*) !!, *] -> child_cap])",
        ),
        (
            "one_child_val_arr_cap_only",
            "(LegacyPTestParent @[child_tuple=[(*), (*) !!, *] -> child_cap !])",
        ),
        (
            "one_child_val_arr_cap_all",
            "(LegacyPTestParent @[child_tuple=[(*), (*) !!, *] -> child_cap !!])",
        ),
        ("two_child_any", "(LegacyPTestParent @[child1 child_any])"),
        ("two_child_any_cap", "(LegacyPTestParent @[child1 -> child_cap child_any])"),
        (
            "two_child_any_cap_both",
            "(LegacyPTestParent @[child1 -> child_cap child_any -> child_any_cap])",
        ),
        (
            "two_child_any_cap_same",
            "(LegacyPTestParent @[child1 -> child_cap child_any -> child_cap])",
        ),
        ("two_child_val_any", "(LegacyPTestParent @[child1=(*) child_any=(*)])"),
        (
            "two_child_val_any_cap",
            "(LegacyPTestParent @[child1=(*) -> child_cap child_any=(*)])",
        ),
        ("two_child_val_none", "(LegacyPTestParent @[child1=None child_any=None])"),
        ("two_child_val_empty", "(LegacyPTestParent @[child_tuple=[] child_tuple2=[]])"),
        (
            "two_child_val_none_cap",
            "(LegacyPTestParent @[child1=None -> child_cap child_any=None])",
        ),
        (
            "two_child_cap_arr_any",
            "(LegacyPTestParent @[child_tuple=[(*) -> first, * -> remaining] child_tuple2=[]])",
        ),
    ],
    ids=lambda p: f"test_pattern_{p}" if not p.startswith("(") else "",
)
def test_correct_pattern_grammar(rule: str, pattern_def: str) -> None:
    res, msg = PatternMatcher.validate_pattern(pattern_def)
    assert res, f"Error in rule {rule}: {msg}"


def test_attr_checks() -> None:
    node = LegacyPTestChildMultiAttrs("fooval", "barval", "bazval", "zazval", origin=NoOrigin())
    test_content_id = node.content_id

    # Test match by content_id
    matcher = PatternMatcher([("rule", f'(* #[content_id="{test_content_id}"])')])
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values == {}

    # allow any attributes
    matcher = PatternMatcher([("rule", "(LegacyPTestChildMultiAttrs #[foo -> foo_val])")])
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["foo_val"] == ["fooval"]

    # allow any attributes, but mismatch attribute value
    matcher = PatternMatcher(
        [("rule", '(LegacyPTestChildMultiAttrs #[foo = "other_val" -> foo_val])')]
    )
    match = matcher.match(node)
    assert match is None

    # test value alternatives
    matcher = PatternMatcher(
        [("rule", '(LegacyPTestChildMultiAttrs #[foo = ("other_val"|"f.o.*l") -> foo_val])')]
    )
    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["foo_val"] == ["fooval"]

    # check ONLY mode (must fail because besides foo, there are other attributes including default ones)
    matcher = PatternMatcher([("rule", "(LegacyPTestChildMultiAttrs #[foo -> foo_val !])")])
    match = matcher.match(node)
    assert match is None

    # check only with extra non existant attribute. Must allow, because only doesn't check for extra attributes
    matcher = PatternMatcher(
        [
            (
                "rule",
                f"(LegacyPTestChildMultiAttrs #[{DEFAULT_ATTRS_STR} foo -> foo_val bar baz zaz extra !])",
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
                f"(LegacyPTestChildMultiAttrs #[{DEFAULT_ATTRS_STR} foo -> foo_val bar baz zaz non_existent !!])",
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
    rule = "(LegacyPTestParent @[child1=(LegacyPTestChild1 #[foo -> test_capture])])"
    matcher = PatternMatcher([("rule", rule)])
    node = LegacyPTestParent(
        child1=LegacyPTestChild1("capture_value", origin=origin),
        child_any=LegacyPTestChild2("non_capture", origin=origin),
        origin=origin,
    )

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["test_capture"] == ["capture_value"]


def test_child_field_match() -> None:
    # Test none match
    node = LegacyPTestParent(
        origin=origin,
    )

    matcher = PatternMatcher(
        [
            ("rule_all_none", "(LegacyPTestParent @[child1=None child_tuple=[]])"),
            ("rule_first_none", "(LegacyPTestParent @[child1=None])"),
            ("rule_second_none", "(LegacyPTestParent @[child_tuple=[]])"),
        ]
    )

    match = matcher.match(node)
    assert match is not None
    assert match[0] == "rule_all_none"

    node = node.replace(child1=LegacyPTestChild1("deep_1", origin=origin))
    match = matcher.match(node)
    assert match is not None
    assert match[0] == "rule_second_none"

    node = node.replace(
        child1=None,
        child_tuple=(
            LegacyPTestChild1("deep_2", origin=origin),
            LegacyPTestChild1("deep_3", origin=origin),
        ),
    )
    match = matcher.match(node)
    assert match is not None
    assert match[0] == "rule_first_none"


def test_child_spec_capture() -> None:
    sub1 = LegacyPTestParent(
        child1=LegacyPTestChild1("deep_1", origin=origin),
        child_any=LegacyPTestChild2("deep_2", origin=origin),
        sub_parent=None,
        origin=origin,
    )

    sub2 = sub1.duplicate()

    node = LegacyPTestParent(
        child_tuple=(sub1, sub2),
        origin=origin,
    )

    rule = "(LegacyPTestParent @[child_tuple -> test_capture])"
    matcher = PatternMatcher([("rule", rule)])

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["test_capture"] == [sub1, sub2]

    # drop everything to avoid collisions
    node.detach()

    # Test array capture with follow rules
    f1 = LegacyPTestChild1("predecessor", origin=origin)
    f2 = f1.duplicate()
    only1 = LegacyPTestChild2("only1", origin=origin)

    node = LegacyPTestParent(
        child_tuple=(f1, f2, only1, sub1, sub2),
        origin=origin,
    )

    rule = "(LegacyPTestParent @[child_tuple =[(LegacyPTestChild1), (LegacyPTestChild2)~1 -> only_capture, *]])"
    matcher = PatternMatcher([("rule", rule)])

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["only_capture"] == [only1]

    # Match any remaining
    rule = "(LegacyPTestParent @[child_tuple =[(*) !! -> first, * -> remaining]])"
    matcher = PatternMatcher([("rule", rule)])

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["first"] == [f1]
    assert values["remaining"] == [f2, only1, sub1, sub2]

    # Capture into same name
    rule = "(LegacyPTestParent @[child_tuple =[(LegacyPTestChild1)~2 -> multi_cap, (LegacyPTestChild2)~1 -> multi_cap, *]])"
    matcher = PatternMatcher([("rule", rule)])

    match = matcher.match(node)
    assert match is not None
    rule, values = match
    assert rule == "rule"
    assert values["multi_cap"] == [f1, f2, only1]
