from __future__ import annotations

import math
import time
import timeit

from pyoak.match.pattern import BaseMatcher
from pyoak.match.xpath import ASTXpath
from sample_tree import Leaf, gen_sample_tree


def run_benchmark():
    TREE_MAX_SIZE = 500000
    DEPTH = 18
    max_nodes = math.floor(math.pow(TREE_MAX_SIZE, 1 / DEPTH))

    tree = gen_sample_tree(DEPTH, max_nodes)
    total_nodes = len(list(tree.dfs()))

    xpath = '/(Inner @attr1="attr1_[0-9]+")//Leaf'

    print(f"Searching for {xpath} in tree with {total_nodes} nodes")

    N = 10

    st = time.monotonic()

    matcher = ASTXpath(xpath)
    print(f"Time to build new matcher: {time.monotonic() - st}")

    st = time.monotonic()
    leafs = [next(matcher.findall(tree)) for _ in range(N)]
    leafs = [leaf for leaf in leafs if leaf is not None]
    print(f"Time to find first {N} items (found {len(leafs)}): {time.monotonic() - st}")

    st = time.monotonic()
    for leaf in leafs:
        matcher.match(leaf)

    print(f"Time to match {len(leafs)} items: {time.monotonic() - st}")

    while not isinstance(left_most_leaf := next(tree.dfs(bottom_up=True)), Leaf):
        pass

    recursive_patterns = f"""
    with leaf as (Leaf @attr4="{left_most_leaf.attr4}"),
    inner as <(Inner @child=#inner) | #leaf>
    #inner
    """

    print(
        f"Searching for a node at {len(list(left_most_leaf.ancestors()))} depth using pattern:{recursive_patterns}"
    )

    st = time.monotonic()
    matcher = BaseMatcher.from_pattern(recursive_patterns)
    print(f"Time to build new matcher: {time.monotonic() - st}")

    # Time the execution of the function 10000 times
    mtime = timeit.timeit(lambda: matcher.match(tree), number=10000)

    print(f"Time to match 10000 times: {mtime:.6f} seconds")


if __name__ == "__main__":
    run_benchmark()
