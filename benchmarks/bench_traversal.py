from __future__ import annotations

import math
import timeit
from dataclasses import dataclass, field

from pyoak.node import ASTNode


@dataclass(frozen=True)
class Inner(ASTNode):
    attr1: str
    attr2: int
    child: ASTNode
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
    tuple_children = tuple(gen_sample_tree(max_depth - 1, max_nodes) for _ in range(max_nodes - 1))

    COUNTER += 1
    return Inner(
        attr1=f"attr1_{COUNTER}",
        attr2=COUNTER,
        child=mandatory_child,
        child_tuple=tuple_children,
    )


def run_benchmark():
    TREE_MAX_SIZE = 1000000
    DEPTH = 18
    max_nodes = math.floor(math.pow(TREE_MAX_SIZE, 1 / DEPTH))
    tree = gen_sample_tree(DEPTH, max_nodes)

    print(f"Tree size: {COUNTER}")

    n = 10
    time = timeit.timeit(lambda: list(tree.dfs()), number=n)
    print(f"Stack based DFS (tree size: {COUNTER}, invocations: {n}): {time:.6f} seconds")

    time = timeit.timeit(lambda: list(tree.bfs()), number=n)
    print(f"Queue based BFS (tree size: {COUNTER}, invocations: {n}): {time:.6f} seconds")


if __name__ == "__main__":
    run_benchmark()
