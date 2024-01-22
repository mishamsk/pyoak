from __future__ import annotations

import math
import time
import timeit

from pyoak.match.pattern import BaseMatcher
from pyoak.match.xpath import ASTXpath
from sample_tree import gen_sample_tree


def run_benchmark():
    TREE_MAX_SIZE = 500000
    DEPTH = 18
    max_nodes = math.floor(math.pow(TREE_MAX_SIZE, 1 / DEPTH))

    tree = gen_sample_tree(DEPTH, max_nodes)

    st = time.monotonic()
    all_nodes = list(tree.bfs())
    full_tree_traversal_time = time.monotonic() - st

    total_nodes = len(all_nodes)

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

    right_bottom_most_leaf = all_nodes[-1]

    recursive_patterns = f"""
    with leaf as (Leaf @attr4="{right_bottom_most_leaf.attr4}") -> cap,
    inner as <(Inner @child=#inner @child_tuple=[#inner*]) | #leaf | (Leaf)>
    #inner
    """

    print(
        "Searching for a right most node at the maximum tree depth of "
        f"{len(list(right_bottom_most_leaf.ancestors()))} using pattern:{recursive_patterns}"
    )

    st = time.monotonic()
    matcher = BaseMatcher.from_pattern(recursive_patterns)
    print(f"Time to build new matcher: {time.monotonic() - st}")

    ok, match_dict = matcher.match(tree)
    assert ok
    assert match_dict["cap"] is right_bottom_most_leaf

    # Time the execution of the function 10 times
    mtime = timeit.timeit(lambda: matcher.match(tree), number=10) / 10

    print(f"Time to match (avg of 10 times): {mtime:.6f} seconds")
    print(f"Comparing to full tree traversal (1 times): {full_tree_traversal_time:.6f} seconds")
    print(f"Slowdown ratio: {mtime / full_tree_traversal_time:.2f}")


if __name__ == "__main__":
    run_benchmark()
