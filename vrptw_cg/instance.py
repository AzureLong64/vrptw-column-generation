"""Solomon/Homberger input, elementary routes, and resource feasibility.

Internal customer indices are 1..n; index 0 represents the depot at both ends.
External IDs are retained when selecting a smaller subset of a benchmark.
Distances and travel times are unrounded Euclidean values by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import numpy as np


@dataclass(frozen=True, slots=True)
class Route:
    customers: tuple[int, ...]
    cost: float
    load: float
    finish: float
    mask: int


@dataclass
class Instance:
    name: str
    ids: np.ndarray
    xy: np.ndarray
    demand: np.ndarray
    ready: np.ndarray
    due: np.ndarray
    service: np.ndarray
    capacity: float
    fleet_metadata: int = 0
    source: str = ""
    distance_decimals: int | None = None
    distance: np.ndarray = field(init=False, repr=False)
    n: int = field(init=False)
    successors: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self):
        self.ids = np.asarray(self.ids, dtype=int)
        self.xy = np.asarray(self.xy, dtype=float)
        for name in ("demand", "ready", "due", "service"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.n = len(self.ids) - 1
        if self.n < 1 or self.ids[0] != 0 or len(set(self.ids.tolist())) != len(self.ids):
            raise ValueError("Expected depot ID 0 and at least one uniquely identified customer")
        if self.xy.shape != (self.n + 1, 2):
            raise ValueError("Coordinates must have shape (n+1, 2)")
        for name in ("demand", "ready", "due", "service"):
            a = getattr(self, name)
            if a.shape != (self.n + 1,) or not np.isfinite(a).all():
                raise ValueError(f"Invalid {name} array")
        if not np.isfinite(self.xy).all() or not np.isfinite(self.capacity) or self.capacity <= 0:
            raise ValueError("Coordinates must be finite and capacity positive")
        if (self.ready > self.due).any() or (self.demand < 0).any() or (self.service < 0).any():
            raise ValueError("Invalid time windows, demands, or service times")
        if self.demand[0] != 0 or self.service[0] != 0:
            raise ValueError("The depot must have zero demand and service time")
        self.distance = np.linalg.norm(self.xy[:, None] - self.xy[None, :], axis=2)
        if self.distance_decimals is not None:
            # Rounded distances can violate the triangle inequality used by pruning.
            raise ValueError("This implementation uses unrounded Euclidean distances only")
        successors = []
        for i in range(self.n + 1):
            reachable = [j for j in range(1, self.n + 1) if j != i
                         and self.ready[i] + self.service[i] + self.distance[i, j] <= self.due[j] + 1e-9
                         and max(self.ready[j], self.ready[i] + self.service[i] + self.distance[i, j])
                         + self.service[j] + self.distance[j, 0] <= self.due[0] + 1e-9]
            successors.append(tuple(sorted(reachable, key=lambda j: (self.distance[i, j], j))))
        self.successors = tuple(successors)
        # Singleton initialization requires all individual customers to be serviceable.
        for i in range(1, self.n + 1):
            if self.make_route((i,)) is None:
                raise ValueError(f"Customer {self.ids[i]} cannot be served by a singleton route")

    def make_route(self, customers: Iterable[int], tol: float = 1e-9) -> Route | None:
        seq = tuple(int(i) for i in customers)
        if not seq or len(set(seq)) != len(seq) or any(i < 1 or i > self.n for i in seq):
            return None
        load = float(self.demand[list(seq)].sum())
        if load > self.capacity + tol:
            return None
        clock, cost, last, mask = float(self.ready[0]), 0.0, 0, 0
        for i in seq:
            travel = float(self.distance[last, i])
            begin = max(clock + travel, float(self.ready[i]))
            if begin > self.due[i] + tol:
                return None
            clock = begin + float(self.service[i])
            cost += travel
            mask |= 1 << (i - 1)
            last = i
        cost += float(self.distance[last, 0])
        clock += float(self.distance[last, 0])
        if clock > self.due[0] + tol:
            return None
        return Route(seq, cost, load, clock, mask)

    def subset(self, ids: Iterable[int]) -> "Instance":
        wanted = list(ids)
        if not wanted or 0 in wanted or len(set(wanted)) != len(wanted):
            raise ValueError("Select distinct customer IDs without depot 0")
        lookup = {int(v): k for k, v in enumerate(self.ids)}
        try:
            take = [0] + [lookup[i] for i in wanted]
        except KeyError as e:
            raise ValueError(f"Unknown customer ID {e.args[0]}") from e
        return Instance(self.name + f"_n{len(wanted)}", self.ids[take], self.xy[take],
                        self.demand[take], self.ready[take], self.due[take], self.service[take],
                        self.capacity, self.fleet_metadata, self.source)


def read_instance(path: str | Path, limit_customers: int | None = None) -> Instance:
    path = Path(path).resolve()
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    nonempty = [s.strip() for s in lines if s.strip()]
    try:
        vehicle = next(i for i, s in enumerate(nonempty) if s.upper() == "VEHICLE")
        fleet, capacity = map(float, nonempty[vehicle + 2].split())
        customer = next(i for i, s in enumerate(nonempty) if s.upper() == "CUSTOMER")
        rows = [list(map(float, s.split())) for s in nonempty[customer + 2:]]
    except (StopIteration, ValueError, IndexError) as e:
        raise ValueError(f"Invalid Solomon/Homberger instance: {path}") from e
    if not rows or any(len(row) != 7 for row in rows):
        raise ValueError("Each customer record must have seven numeric fields")
    a = np.asarray(rows, dtype=float)
    if not np.equal(a[:, 0], np.floor(a[:, 0])).all():
        raise ValueError("Customer IDs must be integers")
    a = a[np.argsort(a[:, 0], kind="stable")]
    instance = Instance(nonempty[0], a[:, 0].astype(int), a[:, 1:3], a[:, 3],
                        a[:, 4], a[:, 5], a[:, 6], capacity, int(fleet), str(path))
    if limit_customers is not None:
        if not 1 <= limit_customers <= instance.n:
            raise ValueError(f"limit_customers must be in [1,{instance.n}]")
        if limit_customers < instance.n:
            instance = instance.subset(instance.ids[1:limit_customers + 1].tolist())
    return instance
