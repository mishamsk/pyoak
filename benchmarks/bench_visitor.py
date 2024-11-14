import math
import time
import timeit
from abc import ABC

from pyoak.node import ASTNode
from pyoak.visitor import ASTVisitor
from sample_tree import Inner, Leaf, gen_sample_tree


def run_benchmark():
    TREE_MAX_SIZE = 1000000
    DEPTH = 18
    max_nodes = math.floor(math.pow(TREE_MAX_SIZE, 1 / DEPTH))

    st = time.monotonic()
    tree = gen_sample_tree(DEPTH, max_nodes)

    print(
        f"Time to build tree with {len(list(tree.dfs()))} nodes: {time.monotonic() - st:.6f} seconds"
    )

    class Virtual(ABC):  # noqa: B024
        pass

    Virtual.register(Leaf)

    def build_visitor():
        class DummyVisitor(ASTVisitor[None], validate=True):
            def __init__(self) -> None:
                self.count_generic = 0
                self.count_virtual = 0
                self.count_inner = 0

                super().__init__()

            def generic_visit(self, node: ASTNode) -> None:
                self.count_generic += 1

            def visit_Virtual(self, node: Virtual) -> None:
                self.count_virtual += 1

            def visit_Inner(self, node: Inner) -> None:
                self.count_inner += 1

        return DummyVisitor()

    n = 100
    ttime = timeit.timeit(build_visitor, number=n)
    print(f"Time to build {n} visitors: {ttime:.6f} seconds")

    visitor = build_visitor()
    all_nodes = list(tree.dfs())

    # Check visitor works properly
    _ = [visitor.visit(node) for node in all_nodes]
    assert visitor.count_generic == len(
        [node for node in all_nodes if not isinstance(node, (Virtual, Inner))]
    )
    assert visitor.count_virtual == len([node for node in all_nodes if isinstance(node, Virtual)])
    assert visitor.count_inner == len([node for node in all_nodes if isinstance(node, Inner)])

    n = 10
    ttime = timeit.timeit(lambda: [visitor.visit(node) for node in all_nodes], number=n)
    print(
        f"Time to call (dispatch) visits on {len(all_nodes)} node (invocations: {n}): {ttime:.6f} seconds"
    )


if __name__ == "__main__":
    run_benchmark()
