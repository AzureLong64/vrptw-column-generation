"""Incremental HiGHS covering LP with true simplex-basis access.

Rows are normalized averages over customer subsets. Both algorithms use this
same class: standard CG supplies singleton rows, dyn-SAR supplies aggregate rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import highspy
import numpy as np
from .instance import Instance, Route


@dataclass
class MasterSolution:
    objective: float
    x: np.ndarray
    gamma: np.ndarray
    pi: np.ndarray
    coverage: np.ndarray
    basis_routes: tuple[int, ...]
    routes: tuple[Route, ...]
    groups: tuple[frozenset[int], ...]
    primal_dual_gap: float

    @property
    def mean_basis_route_size(self) -> float:
        sizes = [len(self.routes[j].customers) for j in self.basis_routes]
        if not sizes:
            raise RuntimeError("No basic route variables in an optimal covering LP")
        return float(np.mean(sizes))


def _check(status, action):
    if status == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS failed to {action}")


class Master:
    def __init__(self, instance: Instance, groups, feasibility_tolerance=1e-8):
        self.instance = instance
        self.h = highspy.Highs()
        for key, value in (("output_flag", False), ("solver", "simplex"), ("threads", 1),
                           ("random_seed", 0), ("primal_feasibility_tolerance", feasibility_tolerance),
                           ("dual_feasibility_tolerance", feasibility_tolerance)):
            _check(self.h.setOptionValue(key, value), f"set {key}")
        self.groups: list[frozenset[int]] = []
        self.group_masks: list[int] = []
        self.group_set: set[frozenset[int]] = set()
        self.routes: list[Route] = []
        self.mask_index: dict[int, int] = {}
        self.add_groups(groups)
        self.add_routes([instance.make_route((i,)) for i in range(1, instance.n + 1)])

    @property
    def known_costs(self):
        return {r.mask: r.cost for r in self.routes}

    def add_groups(self, groups) -> int:
        added = 0
        for raw in groups:
            group = frozenset(int(i) for i in raw)
            if not group or min(group) < 1 or max(group) > self.instance.n:
                raise ValueError("Invalid customer group")
            if group in self.group_set:
                continue
            mask = sum(1 << (i - 1) for i in group)
            indices, values = [], []
            for j, route in enumerate(self.routes):
                count = (mask & route.mask).bit_count()
                if count:
                    indices.append(j)
                    values.append(count / len(group))
            _check(self.h.addRow(1.0, highspy.kHighsInf, len(indices),
                                np.asarray(indices, dtype=np.int32), np.asarray(values, dtype=float)), "add row")
            self.groups.append(group)
            self.group_masks.append(mask)
            self.group_set.add(group)
            added += 1
        return added

    def add_routes(self, routes: list[Route]) -> tuple[int, int]:
        # For a given customer set, only its cheapest discovered visit order is useful.
        pending: dict[int, Route] = {}
        improved = 0
        for route in routes:
            if route is None:
                raise ValueError("Cannot add an infeasible route")
            index = self.mask_index.get(route.mask)
            if index is not None:
                if route.cost < self.routes[index].cost - 1e-9:
                    self.routes[index] = route
                    _check(self.h.changeColCost(index, route.cost), "improve route cost")
                    improved += 1
            elif route.mask not in pending or route.cost < pending[route.mask].cost:
                pending[route.mask] = route
        new = list(pending.values())
        if new:
            starts, indices, values = [0], [], []
            for route in new:
                for k, (mask, group) in enumerate(zip(self.group_masks, self.groups)):
                    count = (mask & route.mask).bit_count()
                    if count:
                        indices.append(k)
                        values.append(count / len(group))
                starts.append(len(indices))
            count = len(new)
            _check(self.h.addCols(count, np.asarray([r.cost for r in new]), np.zeros(count),
                                 np.full(count, highspy.kHighsInf), len(indices),
                                 np.asarray(starts, dtype=np.int32), np.asarray(indices, dtype=np.int32),
                                 np.asarray(values, dtype=float)), "add route columns")
            for route in new:
                self.mask_index[route.mask] = len(self.routes)
                self.routes.append(route)
        return len(new), improved

    def solve(self, seconds: float | None = None) -> MasterSolution:
        # HiGHS accumulates its run timer over incremental solves.
        if seconds is not None:
            _check(self.h.setOptionValue("time_limit", self.h.getRunTime() + max(0.001, seconds)), "set time limit")
        _check(self.h.run(), "solve master")
        status = self.h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            if status in (highspy.HighsModelStatus.kTimeLimit, highspy.HighsModelStatus.kIterationLimit):
                raise TimeoutError(self.h.modelStatusToString(status))
            raise RuntimeError(f"Master LP: {self.h.modelStatusToString(status)}")
        solution, basis = self.h.getSolution(), self.h.getBasis()
        if not basis.valid:
            raise RuntimeError("HiGHS did not return a valid simplex basis")
        x = np.asarray(solution.col_value, dtype=float)
        gamma = np.asarray(solution.row_dual, dtype=float)
        if gamma.min(initial=0) < -1e-7:
            raise RuntimeError("Unexpected negative dual for a covering row")
        gamma = np.maximum(gamma, 0)
        pi = np.zeros(self.instance.n + 1)
        for group, dual in zip(self.groups, gamma):
            pi[list(group)] += dual / len(group)
        coverage = np.zeros(self.instance.n + 1)
        for route, value in zip(self.routes, x):
            if value:
                coverage[list(route.customers)] += value
        basis_routes = tuple(j for j, status in enumerate(basis.col_status)
                             if status == highspy.HighsBasisStatus.kBasic)
        objective = float(self.h.getObjectiveValue())
        return MasterSolution(objective, x, gamma, pi, coverage, basis_routes,
                              tuple(self.routes), tuple(self.groups), abs(objective - gamma.sum()))
