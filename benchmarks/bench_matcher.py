from __future__ import annotations

import math
import time

from pyoak.match.xpath import ASTXpath
from sample_tree import gen_sample_tree


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


if __name__ == "__main__":
    run_benchmark()
