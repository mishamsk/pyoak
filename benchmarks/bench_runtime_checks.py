from __future__ import annotations

import timeit

from pyoak import config
from sample_tree import gen_sample_tree


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
