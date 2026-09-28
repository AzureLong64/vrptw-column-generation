"""Shared solver mechanics, certificates, and trace logging.

The two algorithm drivers live in standard.py and dyn_sar.py. This module owns
their common restricted-master and pricing loop so pricing comparisons are fair.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable
import highspy
import numpy as np
from .instance import Instance
from .master import Master, MasterSolution
from .pricing import Pricer


@dataclass
class Config:
    time_limit: float = 120.0
    pricing_time_limit: float = 20.0
    max_labels: int = 300_000
    max_iterations: int = 10_000
    max_columns: int = 300
    kappa: int = 20
    tolerance: float = 1e-7
    heuristics: bool = True
    heuristic_time_limit: float = 0.25
    heuristic_labels: int = 5_000
    heuristic_beam: int = 40

    def __post_init__(self):
        for name in ("time_limit", "pricing_time_limit", "max_labels", "max_iterations",
                     "max_columns", "kappa", "tolerance", "heuristic_time_limit", "heuristic_labels", "heuristic_beam"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class StopSolve(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__(status)


class Engine:
    def __init__(self, instance: Instance, algorithm: str, config: Config, groups, callback=None):
        self.instance, self.algorithm, self.config = instance, algorithm, config
        self.started = time.monotonic()
        self.deadline = self.started + config.time_limit
        self.master = Master(instance, groups, feasibility_tolerance=min(1e-8, config.tolerance / 10))
        self.pricer = Pricer(instance, config.tolerance)
        self.callback = callback
        self.history: list[dict] = []
        self.events: list[dict] = []
        self.last: MasterSolution | None = None
        self.best_feasible: MasterSolution | None = None
        self.lower_bound: float | None = None
        self.upper_bound = sum(r.cost for r in self.master.routes)
        self.round = 1
        self.iterations = 0
        self.lp_seconds = 0.0
        self.pricing_seconds = 0.0
        self.calls: dict[str, int] = {}
        self.generated_columns = instance.n
        self.improved_columns = 0
        self.generated_rows = len(self.master.groups)

    def emit(self, event):
        event = {"algorithm": self.algorithm, "elapsed": time.monotonic()-self.started, **event}
        self.events.append(event)
        if self.callback:
            self.callback(event)

    def check_limits(self):
        if time.monotonic() >= self.deadline:
            raise StopSolve("time_limit")
        if self.iterations >= self.config.max_iterations:
            raise StopSolve("iteration_limit")

    def converge_current_rows(self) -> MasterSolution:
        """Solve EMP(S) by column generation, or raise an explicit limit status."""
        cfg = self.config
        while True:
            self.check_limits()
            self.iterations += 1
            start = time.monotonic()
            try:
                solution = self.master.solve(self.deadline - start)
            except TimeoutError as e:
                raise StopSolve("lp_time_limit") from e
            lp_seconds = time.monotonic()-start
            self.lp_seconds += lp_seconds
            self.last = solution
            feasible = bool(np.all(solution.coverage[1:] >= 1-cfg.tolerance))
            if feasible and solution.objective <= self.upper_bound + cfg.tolerance:
                self.upper_bound = solution.objective
                self.best_feasible = solution
            known = self.master.known_costs
            calls = []
            def record(result):
                calls.append(result)
                self.calls[result.kind] = self.calls.get(result.kind, 0)+1
                self.pricing_seconds += result.seconds
                return result
            priced = None
            if cfg.heuristics:
                priced = record(self.pricer.greedy(solution.pi, known, cfg.max_columns,
                                                   min(self.deadline, time.monotonic()+cfg.heuristic_time_limit)))
                if not priced.routes:
                    for k in (5, 10):
                        if time.monotonic() >= self.deadline:
                            break
                        priced = record(self.pricer.labels(solution.pi, known, cfg.max_columns,
                                        cfg.heuristic_labels, min(self.deadline, time.monotonic()+cfg.heuristic_time_limit),
                                        network_size=k, beam=cfg.heuristic_beam))
                        if priced.routes:
                            break
            if priced is None or not priced.routes:
                priced = record(self.pricer.exact(solution.pi, known, cfg.max_columns, cfg.max_labels,
                                 min(self.deadline, time.monotonic()+cfg.pricing_time_limit)))
            event = dict(event="iteration", iteration=self.iterations, outer_round=self.round,
                         rows=len(solution.groups), columns=len(solution.routes), rmp_objective=solution.objective,
                         min_coverage=float(solution.coverage[1:].min()),
                         uncovered=int(np.sum(solution.coverage[1:] < 1-cfg.tolerance)),
                         dual_l2=float(np.linalg.norm(solution.pi[1:])),
                         min_reduced_cost_found=min(r.min_reduced_cost for r in calls),
                         pricing_kind=priced.kind, pricing_complete=priced.complete,
                         pricing_reason=priced.reason, labels=sum(r.labels for r in calls),
                         dssr_refinements=sum(r.refinements for r in calls),
                         lp_seconds=lp_seconds, pricing_seconds=sum(r.seconds for r in calls),
                         new_columns=0, improved_columns=0, emp_certified=False)
            if priced.routes:
                if time.monotonic() < self.deadline:
                    new, improved = self.master.add_routes(priced.routes)
                    event.update(new_columns=new, improved_columns=improved)
                    self.generated_columns += new
                    self.improved_columns += improved
                    if new+improved == 0:
                        self.history.append(event)
                        self.emit(event)
                        raise StopSolve("numerical_stall")
                self.history.append(event)
                self.emit(event)
                continue
            if not priced.complete:
                self.history.append(event)
                self.emit(event)
                status = "time_limit" if time.monotonic() >= self.deadline else "pricing_"+priced.reason
                raise StopSolve(status)
            if priced.min_reduced_cost < -cfg.tolerance:
                self.history.append(event)
                self.emit(event)
                raise StopSolve("numerical_stall")
            self.lower_bound = max(self.lower_bound if self.lower_bound is not None else -float("inf"), solution.objective)
            event["emp_certified"] = True
            self.history.append(event)
            self.emit(event)
            return solution

    def add_rows(self, groups):
        self.check_limits()
        new = self.master.add_groups(groups)
        if not new:
            raise RuntimeError("Row separation made no progress")
        self.generated_rows += new
        self.emit(dict(event="separation", outer_round=self.round, new_rows=new,
                       groups=[[int(self.instance.ids[i]) for i in sorted(g)] for g in groups]))
        self.round += 1

    def result(self, status):
        cfg, instance, sol = self.config, self.instance, self.last
        certified = status == "optimal_lp"
        def routes_json(snapshot):
            if snapshot is None:
                return [{"value":1.0, "customers":[int(instance.ids[i])],
                         "path":[0,int(instance.ids[i]),0], "cost":float(2*instance.distance[0,i])}
                        for i in range(1, instance.n+1)]
            return [{"column":j, "value":float(value),
                     "customers":[int(instance.ids[i]) for i in route.customers],
                     "path":[0]+[int(instance.ids[i]) for i in route.customers]+[0],
                     "cost":route.cost, "load":route.load, "return_time":route.finish}
                    for j,(route,value) in enumerate(zip(snapshot.routes,snapshot.x)) if value>1e-9]
        output = {
            "algorithm":self.algorithm, "instance":instance.name, "source":instance.source,
            "n_customers":instance.n, "customer_ids":instance.ids[1:].tolist(),
            "status":status, "certified_optimal":certified,
            "lp_optimum":sol.objective if certified and sol else None,
            "last_rmp_objective":sol.objective if sol else None,
            "last_solution_original_feasible":bool(sol is not None and np.all(sol.coverage[1:]>=1-cfg.tolerance)),
            "certified_emp_lower_bound":self.lower_bound, "feasible_lp_upper_bound":self.upper_bound,
            "elapsed_seconds":time.monotonic()-self.started, "lp_seconds":self.lp_seconds,
            "pricing_seconds":self.pricing_seconds, "iterations":self.iterations,
            "outer_rounds":self.round, "pricing_calls":self.calls,
            "generated_columns":self.generated_columns, "improved_column_orders":self.improved_columns,
            "generated_rows":self.generated_rows,
            "columns_in_last_solution":len(sol.routes) if sol else 0,
            "rows_in_last_solution":len(sol.groups) if sol else 0,
            "active_routes":routes_json(sol) if sol else [],
            "feasible_incumbent_routes":routes_json(self.best_feasible),
            "coverage":{str(instance.ids[i]):float(sol.coverage[i]) for i in range(1,instance.n+1)} if sol else {},
            "duals":{str(instance.ids[i]):float(sol.pi[i]) for i in range(1,instance.n+1)} if sol else {},
            "groups":[[int(instance.ids[i]) for i in sorted(g)] for g in sol.groups] if sol else [],
            "config":asdict(cfg), "highs_version":self.master.h.version(),
            "model":{"master":"set_covering_lp", "routes":"elementary", "distance":"unrounded_euclidean",
                     "travel_time":"distance", "fleet_constraint":False, "fleet_metadata":instance.fleet_metadata,
                     "integer_optimization":False},
            "note":"Only optimal_lp certifies the full elementary-route LP (within numerical tolerance). "
                   "An unfinished aggregated RMP objective is not generally an original-MP bound."
        }
        self.emit(dict(event="finish", status=status, objective=output["last_rmp_objective"], certified=certified))
        output["history"] = self.history
        return output


def solve(instance: Instance, algorithm="standard", config: Config | None=None,
          callback: Callable[[dict],None] | None=None) -> dict:
    config = config or Config()
    if algorithm == "standard":
        from .standard import run
    elif algorithm in ("dyn-sar", "dyn_sar"):
        algorithm = "dyn-sar"
        from .dyn_sar import run
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return run(instance, config, callback)
