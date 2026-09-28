"""Ordinary column generation: all original rows, dynamic route columns."""
from .solver import Engine, StopSolve


def run(instance, config, callback=None):
    engine = Engine(instance, "standard", config,
                    [frozenset([i]) for i in range(1,instance.n+1)], callback)
    try:
        solution = engine.converge_current_rows()
        if min(solution.coverage[1:]) < 1-config.tolerance:
            raise RuntimeError("Original coverage violated in standard CG")
        status = "optimal_lp"
    except StopSolve as e:
        status = e.status
    except KeyboardInterrupt:
        status = "interrupted"
    return engine.result(status)
