"""Algorithm 1: solve relaxed EMP, separate violated aggregate rows, repeat."""
import numpy as np
from .aggregation import initial_groups, biased_neighborhoods, separate_rows
from .solver import Engine, StopSolve


def run(instance, config, callback=None):
    engine = Engine(instance, "dyn-sar", config, initial_groups(instance, config.kappa), callback)
    neighborhoods = biased_neighborhoods(instance)
    try:
        while True:
            # Separation occurs only after complete pricing for the current row set.
            solution = engine.converge_current_rows()
            if np.all(solution.coverage[1:] >= 1-config.tolerance):
                status = "optimal_lp"
                break
            groups = separate_rows(instance, solution.coverage, solution.mean_basis_route_size,
                                   neighborhoods, engine.master.group_set, config.tolerance)
            engine.add_rows(groups)
    except StopSolve as e:
        status = e.status
    except KeyboardInterrupt:
        status = "interrupted"
    return engine.result(status)
