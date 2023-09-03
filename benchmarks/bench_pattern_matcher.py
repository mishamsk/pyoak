from __future__ import annotations

import time
from dataclasses import dataclass, field

from pyoak.legacy.match.pattern import PatternMatcher as OldPatternMatcher
from pyoak.legacy.node import AwareASTNode
from pyoak.match.pattern import MultiPatternMatcher
from pyoak.node import ASTNode
from pyoak.origin import NO_ORIGIN


@dataclass(frozen=True)
class PTestChild1(ASTNode):
    foo: str


@dataclass(frozen=True)
class PTestChild2(ASTNode):
    bar: str


@dataclass(frozen=True)
class PTestParent(ASTNode):
    child1: PTestChild1 | None = None
    child_any: PTestChild2 | PTestChild1 | None = None
    sub_parent: PTestParent | None = None
    child_tuple: tuple[ASTNode, ...] = field(default_factory=tuple)


f1 = PTestChild1("predecessor")
f2 = f1.duplicate()
only1 = PTestChild2("only1")
sub1 = PTestParent(
    child1=PTestChild1("deep_1"),
    child_any=PTestChild2("deep_2"),
    sub_parent=None,
)

sub2 = sub1.duplicate()

node = PTestParent(
    child_tuple=(f1, f2, only1, sub1, sub2),
)

rule = "(PTestParent @child_tuple =[(PTestChild1) (*) (PTestChild2) -> only_capture *])"

N = 10000

st = time.monotonic()

matcher = MultiPatternMatcher([("rule", rule)])
print(f"Time to build new matcher: {time.monotonic() - st}")

st = time.monotonic()
for _ in range(N):
    matcher.match(node)

print(f"Time to match {N} times with new matcher: {time.monotonic() - st}")


@dataclass
class LegacyPTestChild1(AwareASTNode):
    foo: str


@dataclass
class LegacyPTestChild2(AwareASTNode):
    bar: str


@dataclass
class LegacyPTestParent(AwareASTNode):
    child1: LegacyPTestChild1 | None = None
    child_any: LegacyPTestChild2 | LegacyPTestChild1 | None = None
    sub_parent: LegacyPTestParent | None = None
    child_tuple: tuple[AwareASTNode, ...] = field(default_factory=tuple)


f1 = LegacyPTestChild1("predecessor", origin=NO_ORIGIN)  # type: ignore
f2 = f1.duplicate()
only1 = LegacyPTestChild2("only1", origin=NO_ORIGIN)  # type: ignore
sub1 = LegacyPTestParent(  # type: ignore
    child1=LegacyPTestChild1("deep_1", origin=NO_ORIGIN),
    child_any=LegacyPTestChild2("deep_2", origin=NO_ORIGIN),
    sub_parent=None,
    origin=NO_ORIGIN,
)

sub2 = sub1.duplicate()

legacy_node = LegacyPTestParent(
    child_tuple=(f1, f2, only1, sub1, sub2),  # type: ignore
    origin=NO_ORIGIN,
)

old_rule = "(LegacyPTestParent @[child_tuple =[(LegacyPTestChild1), (LegacyPTestChild2)~1 -> only_capture, *]])"

st = time.monotonic()
old_matcher = OldPatternMatcher([("rule", old_rule)])
print(f"Time to build old matcher: {time.monotonic() - st}")

st = time.monotonic()
for _ in range(N):
    old_matcher.match(legacy_node)

print(f"Time to match {N} times with old matcher: {time.monotonic() - st}")
