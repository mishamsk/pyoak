from dataclasses import dataclass, field

from pyoak.node import ASTNode
from pyoak.origin import NO_ORIGIN, Origin


@dataclass
class Inner(ASTNode):
    attr1: str
    attr2: int
    child: ASTNode
    child_tuple: tuple[ASTNode, ...] = field(default_factory=tuple)

    origin: Origin = field(default=NO_ORIGIN, kw_only=True)


@dataclass
class Leaf(ASTNode):
    attr4: str
    attr5: int

    origin: Origin = field(default=NO_ORIGIN, kw_only=True)


COUNTER: int = 0


def gen_sample_tree(max_depth: int, max_nodes: int) -> ASTNode:
    global COUNTER
    if max_depth == 0:
        COUNTER += 1
        return Leaf(attr4=f"leaf_{COUNTER}", attr5=COUNTER)

    mandatory_child = gen_sample_tree(max_depth - 1, max_nodes)
    tuple_children = tuple(gen_sample_tree(max_depth - 1, max_nodes) for _ in range(max_nodes - 1))

    COUNTER += 1
    return Inner(
        attr1=f"attr1_{COUNTER}",
        attr2=COUNTER,
        child=mandatory_child,
        child_tuple=tuple_children,
    )
