"""Independent enumeration/LP cross-checks and limit-state regression tests."""
from pathlib import Path
import itertools
import math
import time
import numpy as np
import pytest
from scipy.optimize import linprog
from vrptw_cg import Config, Instance, read_instance, solve
from vrptw_cg.aggregation import initial_groups, biased_neighborhoods, separate_rows
from vrptw_cg.master import Master
from vrptw_cg.pricing import Pricer, enumerate_routes

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent/'instances-ijoc-dyn-sar-main/VRPTW/homberger_200_customer_instances'


def synthetic(seed, n=6):
    rng = np.random.default_rng(seed)
    xy = np.vstack([[0,0],rng.uniform(0.5,12,size=(n,2))])
    demand = np.r_[0,rng.integers(1,4,size=n)]
    ready = np.r_[0,rng.uniform(0,15,size=n)]
    due = ready + np.r_[120,rng.uniform(20,50,size=n)]
    return Instance(f'random_{seed}',np.arange(n+1),xy,demand,ready,due,np.r_[0,np.ones(n)],7)


def reference_lp(instance):
    routes = enumerate_routes(instance)
    a = np.zeros((instance.n,len(routes)))
    for j,route in enumerate(routes):
        a[np.array(route.customers)-1,j]=1
    result = linprog([r.cost for r in routes],A_ub=-a,b_ub=-np.ones(instance.n),method='highs-ds')
    assert result.success
    return result.fun,routes


@pytest.mark.parametrize('seed',range(5))
def test_both_algorithms_match_all_route_lp(seed):
    instance = synthetic(seed)
    optimum,_ = reference_lp(instance)
    for algorithm in ('standard','dyn-sar'):
        result=solve(instance,algorithm,Config(time_limit=15))
        assert result['status']=='optimal_lp'
        assert result['lp_optimum']==pytest.approx(optimum,abs=1e-6)
        assert min(result['coverage'].values())>=1-1e-7
        assert sum(r['value']*r['cost'] for r in result['active_routes'])==pytest.approx(optimum)
        for r in result['active_routes']:
            assert instance.make_route(r['customers']) is not None
        exact_points=[h['rmp_objective'] for h in result['history'] if h['emp_certified']]
        assert all(z<=optimum+1e-6 for z in exact_points)
        assert all(a<=b+1e-6 for a,b in zip(exact_points,exact_points[1:]))


@pytest.mark.parametrize('seed',range(8))
def test_exact_pricer_against_exhaustive_minimum(seed):
    instance=synthetic(seed,n=7)
    routes=enumerate_routes(instance)
    pi=np.r_[0,np.random.default_rng(seed+500).uniform(0,30,size=instance.n)]
    expected=min([0.0]+[r.cost-pi[list(r.customers)].sum() for r in routes])
    priced=Pricer(instance).labels(pi,max_columns=100000,max_labels=200000,stop_on_batch=False)
    assert priced.complete
    assert priced.min_reduced_cost==pytest.approx(expected,abs=1e-8)
    assert all(r.cost-pi[list(r.customers)].sum() < -1e-7 for r in priced.routes)
    assert all(instance.make_route(r.customers) is not None for r in priced.routes)


def test_exact_certificate_at_reference_dual():
    instance=synthetic(42)
    _,routes=reference_lp(instance)
    master=Master(instance,[{i} for i in range(1,instance.n+1)])
    master.add_routes(routes)
    sol=master.solve()
    result=Pricer(instance).labels(sol.pi,master.known_costs,stop_on_batch=False)
    assert result.complete and not result.routes
    assert result.min_reduced_cost>=-1e-7
    relaxed=Pricer(instance).exact(sol.pi,master.known_costs)
    assert relaxed.complete and not relaxed.routes


@pytest.mark.parametrize('seed',range(6))
def test_dssr_never_misses_a_negative_elementary_route(seed):
    instance=synthetic(seed,n=7)
    all_routes=enumerate_routes(instance)
    pricer=Pricer(instance)
    for shift in (0,1,2):
        pi=np.r_[0,np.random.default_rng(seed+300*shift).uniform(0,12*(shift+1),size=instance.n)]
        minimum=min(r.cost-pi[list(r.customers)].sum() for r in all_routes)
        result=pricer.exact(pi,{},max_labels=200000)
        if minimum < -1e-7:
            assert result.routes
        else:
            assert result.complete and not result.routes
        assert all(instance.make_route(r.customers) is not None for r in result.routes)
        assert all(r.cost-pi[list(r.customers)].sum() < -1e-7 for r in result.routes)


def test_aggregate_dual_mapping_and_real_basis():
    instance=read_instance(PROJECT/'examples/tiny6.txt')
    master=Master(instance,[{1,2,3},{3,4,5,6}])
    sol=master.solve()
    assert sol.pi.sum()==pytest.approx(sol.gamma.sum())
    assert sol.pi[3]==pytest.approx(sol.gamma[0]/3+sol.gamma[1]/4)
    assert sol.primal_dual_gap<1e-7
    assert sol.basis_routes
    assert sol.mean_basis_route_size==1
    assert all(r.cost-sol.pi[list(r.customers)].sum() >= -1e-7 for r in sol.routes)


def test_dssr_refines_a_negative_cycle_before_certifying():
    # 0-1-2-1-0 has reduced cost -10 in the relaxed state space, whereas
    # every elementary route has reduced cost >=0. A relaxed walk must never
    # enter the master or cause an early, false pricing certificate.
    instance = Instance('cycle', np.arange(4),
                        np.array([[0, 0], [10, 0], [10, 0], [20, 0]]),
                        np.array([0, 1, 1, 1]), np.zeros(4),
                        np.full(4, 500), np.zeros(4), 10)
    pi = np.array([0, 10, 10, 0])
    relaxed = Pricer(instance).labels(pi, critical_mask=0, stop_on_batch=False)
    assert relaxed.min_reduced_cost == pytest.approx(-10)
    assert relaxed.repeated_customers
    result = Pricer(instance).exact(pi, {})
    assert result.complete and not result.routes
    assert result.refinements > 0
    assert result.min_reduced_cost >= -1e-7


def test_same_set_better_order_updates_existing_column():
    instance=synthetic(10,n=3)
    feasible=[instance.make_route(p) for p in itertools.permutations([1,2,3])]
    feasible=sorted((r for r in feasible if r is not None),key=lambda r:r.cost)
    if len(feasible)<2 or abs(feasible[0].cost-feasible[-1].cost)<1e-9:
        pytest.fail('Test fixture needs two feasible orders with different costs')
    master=Master(instance,[{1},{2},{3}])
    assert master.add_routes([feasible[-1]])==(1,0)
    assert master.add_routes([feasible[0]])==(0,1)
    assert len(master.routes)==4
    assert master.routes[-1].cost==feasible[0].cost


def test_limits_never_claim_optimality():
    instance=read_instance(PROJECT/'examples/tiny6.txt')
    for algorithm in ('standard','dyn-sar'):
        result=solve(instance,algorithm,Config(max_labels=1,heuristics=False))
        assert result['status']=='pricing_label_limit'
        assert not result['certified_optimal']
        assert result['lp_optimum'] is None
        assert result['certified_emp_lower_bound'] is None
    assert not result['last_solution_original_feasible']


def test_iteration_limit_keeps_coherent_last_solution():
    instance=read_instance(PROJECT/'examples/tiny6.txt')
    result=solve(instance,'standard',Config(max_iterations=1))
    assert result['status']=='iteration_limit'
    assert not result['certified_optimal']
    assert result['columns_in_last_solution']==6
    assert result['generated_columns']>6
    assert sum(r['cost']*r['value'] for r in result['active_routes'])==pytest.approx(result['last_rmp_objective'])


def test_heuristic_search_does_not_issue_full_certificate():
    instance=synthetic(4)
    result=Pricer(instance).labels(np.zeros(instance.n+1),network_size=5)
    assert result.reason=='exhausted'
    assert not result.complete
    expired=Pricer(instance).labels(np.ones(instance.n+1),deadline=time.monotonic()-1)
    assert expired.reason=='time_limit' and not expired.complete


def test_no_heuristic_cg_also_matches_reference():
    instance=read_instance(PROJECT/'examples/tiny6.txt')
    optimum,_=reference_lp(instance)
    for algo in ('standard','dyn-sar'):
        result=solve(instance,algo,Config(heuristics=False,max_columns=2))
        assert result['certified_optimal']
        assert result['lp_optimum']==pytest.approx(optimum,abs=1e-7)


def test_zero_demand_and_colocated_nodes():
    instance=Instance('zero',np.arange(4),np.array([[0,0],[1,0],[1,0],[2,0]]),
                      np.array([0,0,1,1]),np.zeros(4),np.full(4,20),np.zeros(4),1)
    routes=enumerate_routes(instance)
    pi=np.array([0,0.0001,10,3])
    expected=min([0.0]+[r.cost-pi[list(r.customers)].sum() for r in routes])
    result=Pricer(instance).labels(pi,stop_on_batch=False)
    assert result.complete
    assert result.min_reduced_cost==pytest.approx(expected)


@pytest.mark.skipif(not DATA.exists(),reason='External benchmark folder unavailable')
def test_real_instance_feasibility_and_initialization():
    instance=read_instance(DATA/'R2_2_5.TXT')
    assert instance.n==200 and instance.capacity==1000
    assert [len(g) for g in initial_groups(instance)]==[20,40,80,60]
    route=instance.make_route((4,3))
    assert route.cost==pytest.approx(87.4236550418)
    assert route.finish==pytest.approx(913.0594816896)
    assert instance.make_route((3,4)) is None
    assert instance.make_route((3,3)) is None
    neighborhoods=biased_neighborhoods(instance)
    assert all(len(v)==50 for v in neighborhoods.values())
    coverage=np.r_[0,np.ones(instance.n)]
    coverage[[3,4,5,6]]=0.3
    groups=separate_rows(instance,coverage,4,neighborhoods,[])
    assert groups and len(groups)<=10
    assert all(g <= {3,4,5,6} for g in groups)
    assert all(np.mean(coverage[list(g)])<1 for g in groups)


@pytest.mark.skipif(not DATA.exists(),reason='External benchmark folder unavailable')
def test_real_subset_matches_reference():
    instance=read_instance(DATA/'R2_2_5.TXT',8)
    optimum,_=reference_lp(instance)
    for algo in ('standard','dyn-sar'):
        result=solve(instance,algo,Config())
        assert result['certified_optimal']
        assert result['lp_optimum']==pytest.approx(optimum,abs=1e-6)
