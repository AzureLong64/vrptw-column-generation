"""Shared ESPPRC pricing: greedy search, reduced networks, and exact DSSR.

An exact call is allowed to stop after finding a batch of negative columns.
Only exhaustion of the full-network search certifies absence of negative
reduced-cost elementary routes. Time/label limits never provide a certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time
import numpy as np
from .instance import Instance, Route


@dataclass
class PricingResult:
    routes: list[Route]
    complete: bool
    reason: str
    min_reduced_cost: float
    labels: int
    seconds: float
    kind: str
    repeated_customers: tuple[int, ...] = ()
    refinements: int = 0


@dataclass(slots=True)
class Label:
    node: int
    departure: float
    load: float
    reduced: float
    mask: int
    parent: "Label | None"
    depth: int = 0
    active: bool = True


def _sequence(label: Label) -> tuple[int, ...]:
    route = []
    while label.parent is not None:
        route.append(label.node)
        label = label.parent
    return tuple(reversed(route))


class Pricer:
    def __init__(self, instance: Instance, tolerance=1e-7):
        self.instance = instance
        self.tolerance = tolerance
        self.reduced_networks = {k: self._reduced_network(k) for k in (5, 10)}

    def _reduced_network(self, k):
        instance = self.instance
        out = [set(seq[:k]) for seq in instance.successors]
        out[0] = set(instance.successors[0])
        incoming = [[] for _ in range(instance.n + 1)]
        for i in range(1, instance.n + 1):
            for j in instance.successors[i]:
                incoming[j].append(i)
        for j in range(1, instance.n + 1):
            for i in sorted(incoming[j], key=lambda i: (instance.distance[i, j], i))[:k]:
                out[i].add(j)
        return tuple(tuple(sorted(seq, key=lambda j: (instance.distance[i, j], j))) for i, seq in enumerate(out))

    def greedy(self, pi: np.ndarray, known: dict[int, float], max_columns: int, deadline: float) -> PricingResult:
        start = time.monotonic()
        inst = self.instance
        found: dict[int, Route] = {}
        minimum = 0.0
        customer = np.arange(1, inst.n + 1)
        seeds = sorted(customer.tolist(), key=lambda i: (-(pi[i] - inst.distance[0, i]), i))
        for seed in seeds:
            if time.monotonic() >= deadline:
                break
            seq = [seed]
            clock = max(inst.ready[0] + inst.distance[0, seed], inst.ready[seed]) + inst.service[seed]
            load = inst.demand[seed]
            used = np.zeros(inst.n + 1, dtype=bool)
            used[0] = used[seed] = True
            while True:
                route = inst.make_route(seq)
                if route is None:
                    break
                rc = float(route.cost - pi[list(route.customers)].sum())
                minimum = min(minimum, rc)
                if rc < -self.tolerance and route.cost < known.get(route.mask, math.inf) - 1e-9:
                    if route.mask not in found or route.cost < found[route.mask].cost:
                        found[route.mask] = route
                if len(found) >= max_columns:
                    return PricingResult(list(found.values()), False, "column_batch", minimum, 0,
                                         time.monotonic() - start, "greedy")
                last = seq[-1]
                begin = np.maximum(clock + inst.distance[last], inst.ready)
                feasible = (~used) & (load + inst.demand <= inst.capacity + 1e-9) & (begin <= inst.due + 1e-9)
                feasible &= begin + inst.service + inst.distance[:, 0] <= inst.due[0] + 1e-9
                delta = inst.distance[last] + inst.distance[:, 0] - inst.distance[last, 0] - pi
                feasible &= delta < -self.tolerance
                candidates = np.flatnonzero(feasible)
                if not len(candidates):
                    break
                nxt = int(candidates[np.argmin(delta[candidates])])
                clock = float(begin[nxt] + inst.service[nxt])
                load += inst.demand[nxt]
                used[nxt] = True
                seq.append(nxt)
        return PricingResult(list(found.values()), False, "heuristic", minimum, 0, time.monotonic() - start, "greedy")

    def labels(self, pi: np.ndarray, known: dict[int, float] | None = None,
               max_columns: int = 300, max_labels: int = 500_000,
               deadline: float = math.inf, network_size: int | None = None,
               beam: int | None = None, stop_on_batch: bool = True,
               critical_mask: int | None = None) -> PricingResult:
        start = time.monotonic()
        inst, eps = self.instance, self.tolerance
        known = known or {}
        full = network_size is None and beam is None
        relaxed = critical_mask is not None
        kind = "relaxed_labels" if relaxed else ("exact_labels" if full else f"network_{network_size}")
        graph = inst.successors if network_size is None else self.reduced_networks[network_size]
        d, q, ready, due, service = inst.distance, inst.demand, inst.ready, inst.due, inst.service
        positive = [i for i in range(1, inst.n + 1) if pi[i] > 0]
        reward_order = sorted(positive, key=lambda i: (-math.inf if q[i] == 0 else -pi[i] / q[i], i))
        found: dict[int, Route] = {}
        minimum, accepted, attempted, serial = 0.0, 1, 0, 0
        invalid_minimum, repeats = 0.0, ()
        first = Label(0, float(ready[0]), 0.0, 0.0, 0, None)
        queue = [(first.departure, serial, first)]
        buckets: list[list[Label]] = [[] for _ in range(inst.n + 1)]
        def finish(reason):
            routes = sorted(found.values(), key=lambda r: (r.cost - pi[list(r.customers)].sum(), r.customers))[:max_columns]
            return PricingResult(routes, full and reason == "exhausted", reason, minimum,
                                 accepted, time.monotonic() - start, kind, repeats)
        while queue:
            if time.monotonic() >= deadline:
                return finish("time_limit")
            _, _, label = heapq.heappop(queue)
            if not label.active:
                continue
            i = label.node
            if i:
                rc = float(label.reduced + d[i, 0])
                minimum = min(minimum, rc)
                if rc < -eps:
                    sequence = _sequence(label)
                    if len(set(sequence)) != len(sequence):
                        if not relaxed:
                            raise RuntimeError("Repeated customer in elementary pricing")
                        if rc < invalid_minimum:
                            invalid_minimum = rc
                            seen, repeated = set(), set()
                            for customer in sequence:
                                if customer in seen:
                                    repeated.add(customer)
                                seen.add(customer)
                            repeats = tuple(sorted(repeated))
                    else:
                        route = inst.make_route(sequence)
                        if route is None:
                            raise RuntimeError("Label algorithm produced an infeasible route")
                        if route.cost < known.get(route.mask, math.inf) - 1e-9:
                            previous = found.get(route.mask)
                            if previous is None or route.cost < previous.cost:
                                found[route.mask] = route
                    if stop_on_batch and len(found) >= max_columns:
                        return finish("column_batch")
            # An optimistic fractional-knapsack prize bound. Extra Euclidean
            # travel cannot reduce the cost of returning directly to the depot.
            if label.depth >= inst.n:
                continue
            capacity, upper_prize = inst.capacity - label.load, 0.0
            largest_prize, largest_density = 0.0, 0.0
            for j in reward_order:
                if label.mask & (1 << (j - 1)):
                    continue
                earliest = max(label.departure + d[i, j], ready[j])
                if earliest > due[j] + 1e-9 or earliest + service[j] + d[j, 0] > due[0] + 1e-9:
                    continue
                if relaxed:
                    largest_prize = max(largest_prize, pi[j])
                    largest_density = max(largest_density, math.inf if q[j] == 0 else pi[j]/q[j])
                    continue
                if q[j] == 0:
                    upper_prize += pi[j]
                    continue
                take = min(1.0, capacity / q[j])
                upper_prize += take * pi[j]
                capacity -= take * q[j]
                if capacity <= 1e-12:
                    break
            if relaxed:
                # Noncritical customers may repeat, so the elementary knapsack
                # bound is not valid. Use two optimistic repetition-aware bounds.
                capacity_bound = math.inf if math.isinf(largest_density) else capacity*largest_density
                upper_prize = min((inst.n-label.depth)*largest_prize, capacity_bound)
            if label.reduced + d[i, 0] - upper_prize >= -eps:
                continue
            for j in graph[i]:
                attempted += 1
                if attempted % 1024 == 0 and time.monotonic() >= deadline:
                    return finish("time_limit")
                bit = 1 << (j - 1)
                if label.mask & bit:
                    continue
                load = label.load + q[j]
                if load > inst.capacity + 1e-9:
                    continue
                begin = max(label.departure + d[i, j], ready[j])
                departure = begin + service[j]
                if begin > due[j] + 1e-9 or departure + d[j, 0] > due[0] + 1e-9:
                    continue
                memory_bit = bit if critical_mask is None or bit & critical_mask else 0
                mask, reduced = label.mask | memory_bit, float(label.reduced + d[i, j] - pi[j])
                depth = label.depth + 1
                bucket = buckets[j]
                # Subset dominance preserves every feasible continuation.
                if any(old.active and old.departure <= departure and old.load <= load
                       and old.reduced <= reduced and old.depth <= depth
                       and old.mask & mask == old.mask for old in bucket):
                    continue
                if accepted >= max_labels:
                    return finish("label_limit")
                new = Label(j, float(departure), float(load), reduced, mask, label, depth)
                live = []
                for old in bucket:
                    if old.active and departure <= old.departure and load <= old.load and reduced <= old.reduced and depth <= old.depth and mask & old.mask == mask:
                        old.active = False
                    if old.active:
                        live.append(old)
                live.append(new)
                if beam is not None and len(live) > beam:
                    worst = max(live, key=lambda x: (x.reduced + d[x.node, 0], x.departure, x.mask))
                    worst.active = False
                    live.remove(worst)
                buckets[j] = live
                accepted += 1
                if new.active:
                    serial += 1
                    heapq.heappush(queue, (departure, serial, new))
        return finish("exhausted")

    def exact(self, pi: np.ndarray, known: dict[int, float], max_columns=300,
              max_labels=300_000, deadline=math.inf) -> PricingResult:
        """DSSR pricing with exact absence certificates and elementary columns.

        Start with no remembered customers. If a negative relaxed route repeats a
        customer, require those repeated customers to be visited at most once,
        then resolve. All elementary routes remain in every relaxed search. At
        most n visits are allowed even before full elementarity, ensuring
        finiteness for zero resources.
        """
        start = time.monotonic()
        critical, labels, rounds = 0, 0, 0
        while True:
            if time.monotonic() >= deadline or labels >= max_labels:
                return PricingResult([],False,"time_limit" if time.monotonic() >= deadline else "label_limit",
                                     0.0,labels,time.monotonic()-start,"exact_dssr",refinements=rounds)
            priced = self.labels(pi,known,max_columns,max_labels-labels,deadline,
                                 critical_mask=None if critical == (1<<self.instance.n)-1 else critical)
            labels += priced.labels
            if priced.routes:
                # Only elementary columns leave this wrapper, even when the
                # most negative relaxed walk has repetitions.
                return PricingResult(priced.routes,False,"column_batch",
                                     min(r.cost-pi[list(r.customers)].sum() for r in priced.routes),
                                     labels,time.monotonic()-start,"exact_dssr",refinements=rounds)
            if priced.complete and priced.min_reduced_cost >= -self.tolerance:
                return PricingResult([],True,"exhausted",priced.min_reduced_cost,labels,
                                     time.monotonic()-start,"exact_dssr",refinements=rounds)
            if priced.repeated_customers:
                expanded = critical | sum(1<<(i-1) for i in priced.repeated_customers)
                if expanded == critical:
                    raise RuntimeError("DSSR cannot refine a previously forbidden repetition")
                critical = expanded
                rounds += 1
                continue
            return PricingResult([],False,priced.reason,priced.min_reduced_cost,labels,
                                 time.monotonic()-start,"exact_dssr",refinements=rounds)


def enumerate_routes(instance: Instance, max_routes: int = 2_000_000) -> list[Route]:
    """Independent exhaustive reference for tiny instances; not used by CG."""
    if instance.n > 10:
        raise ValueError("Exhaustive validation is limited to at most ten customers")
    best, count = {}, 0
    def visit(seq, available):
        nonlocal count
        for j in available:
            route = instance.make_route(seq + (j,))
            if route is None:
                continue
            count += 1
            if count > max_routes:
                raise RuntimeError("Exhaustive enumeration limit reached; no reference certificate")
            if route.mask not in best or route.cost < best[route.mask].cost:
                best[route.mask] = route
            visit(route.customers, tuple(i for i in available if i != j))
    visit((), tuple(range(1, instance.n + 1)))
    return list(best.values())
