"""Costa et al. Section 4.1.1: initial rows and VRPTW separation.

Unspecified choices are deterministic: ceil for mean route length, floor for
n/4 and 5% caps (at least one on small instances), most-violated seed, then
max-min symmetric difference with violation and customer-ID tie breaks.
"""
from __future__ import annotations

import math
import numpy as np
from .instance import Instance


def initial_groups(instance: Instance, kappa: int = 20) -> list[frozenset[int]]:
    if kappa < 1:
        raise ValueError("kappa must be positive")
    order = sorted(range(1, instance.n + 1), key=lambda i: (instance.distance[0, i], i))
    groups, start, size = [], 0, kappa
    while start < len(order):
        groups.append(frozenset(order[start:start + size]))
        start += size
        size *= 2
    return groups


def biased_neighborhoods(instance: Instance) -> dict[int, tuple[int, ...]]:
    size = min(instance.n - 1, max(1, instance.n // 4))
    radial = instance.distance[0]
    result = {}
    for i in range(1, instance.n + 1):
        def score(j):
            d = float(instance.distance[i, j])
            if d == 0:
                return -math.inf, j
            # Logarithm preserves the ordering of d**beta and avoids overflow.
            # A zero depot distance has no defined ratio in the paper: use 1e-12.
            beta = max(1.0, float(radial[i]) / max(float(radial[j]), 1e-12))
            return beta * math.log(d), j
        result[i] = tuple(sorted((j for j in range(1, instance.n + 1) if j != i), key=score)[:size])
    return result


def separate_rows(instance: Instance, coverage: np.ndarray, mean_route_size: float,
                  neighborhoods: dict[int, tuple[int, ...]], existing,
                  tolerance: float = 1e-7) -> list[frozenset[int]]:
    under = {i for i in range(1, instance.n + 1) if coverage[i] < 1 - tolerance}
    if not under:
        return []
    known = set(existing)
    delta = 1 - coverage
    count = max(1, math.ceil(mean_route_size))
    candidates = set()
    for i in sorted(under, key=lambda i: (-delta[i], i)):
        neighbors = sorted(under.intersection(neighborhoods[i]), key=lambda j: (-delta[j], j))
        group = frozenset([i] + neighbors[:count])
        if group not in known:
            candidates.add(group)
    # This fallback can only be needed because of tolerances/degenerate corner cases.
    if not candidates:
        candidates = {frozenset([i]) for i in under if frozenset([i]) not in known}
    if not candidates:
        raise RuntimeError("Undercoverage remains but no new violated row can be generated")
    def violation(group):
        return float(np.mean(delta[list(group)]))
    pool = sorted(candidates, key=lambda g: (-violation(g), tuple(sorted(g))))[:max(1, instance.n // 4)]
    selected = [pool.pop(0)]
    cap = max(1, instance.n // 20)
    while pool and len(selected) < cap:
        group = min(pool, key=lambda g: (-min(len(g ^ other) for other in selected),
                                         -violation(g), tuple(sorted(g))))
        pool.remove(group)
        selected.append(group)
    if any(float(np.mean(coverage[list(g)])) >= 1 - tolerance for g in selected):
        raise RuntimeError("Separation returned a nonviolated row")
    return selected
