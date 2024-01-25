import math
import time
import timeit

from sample_tree import gen_sample_tree


def run_benchmark():
    TREE_MAX_SIZE = 1000000
    DEPTH = 18
    max_nodes = math.floor(math.pow(TREE_MAX_SIZE, 1 / DEPTH))

    st = time.monotonic()
    tree = gen_sample_tree(DEPTH, max_nodes)

    print(
        f"Time to build tree with {len(list(tree.dfs()))} nodes: {time.monotonic() - st:.6f} seconds"
    )

    timer = timeit.Timer(lambda: list(tree.dfs()))
    n, _ = timer.autorange()
    ttime = timer.repeat(number=n)
    print(f"Queue based DFS. {n} loops, best of 5: {min(ttime):.6f} seconds")

    timer = timeit.Timer(lambda: list(tree.bfs()))
    n, _ = timer.autorange()
    ttime = timer.repeat(number=n)
    print(f"Queue based BFS. {n} loops, best of 5: {min(ttime):.6f} seconds")


if __name__ == "__main__":
    run_benchmark()
