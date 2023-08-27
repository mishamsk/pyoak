from __future__ import annotations

import timeit
from dataclasses import dataclass, field

from pyoak import config
from pyoak.node import ASTNode


@dataclass(frozen=True)
class Inner(ASTNode):
    attr1: str
    attr2: int
    child: ASTNode
    opt_child: ASTNode | None = None
    child_tuple: tuple[ASTNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Leaf(ASTNode):
    attr4: str
    attr5: int


COUNTER: int = 0


def gen_sample_tree(max_depth: int, max_nodes: int) -> ASTNode:
    global COUNTER
    if max_depth == 0:
        COUNTER += 1
        return Leaf(attr4=f"leaf_{COUNTER}", attr5=COUNTER)

    mandatory_child = gen_sample_tree(max_depth - 1, max_nodes)

    opt_child = gen_sample_tree(max_depth - 1, max_nodes)

    num_for_tuple = max_nodes - 2
    tuple_children = tuple(gen_sample_tree(max_depth - 1, max_nodes) for _ in range(num_for_tuple))

    COUNTER += 1
    return Inner(
        attr1=f"attr1_{COUNTER}",
        attr2=COUNTER,
        child=mandatory_child,
        opt_child=opt_child,
        child_tuple=tuple_children,
    )


def run_benchmark():
    times = timeit.repeat(lambda: gen_sample_tree(4, 10), repeat=10, number=1)
    no_runtime_avg_time = sum(times) / len(times)
    print(f"Average build time without runtime checks: {no_runtime_avg_time:.6f} seconds")

    config.RUNTIME_TYPE_CHECK = True
    times = timeit.repeat(lambda: gen_sample_tree(4, 10), repeat=10, number=1)
    runtime_avg_time = sum(times) / len(times)
    print(f"Average build time with runtime checks: {runtime_avg_time:.6f} seconds")

    print(f"An increase of {runtime_avg_time / no_runtime_avg_time:.2f}x")


if __name__ == "__main__":
    run_benchmark()
