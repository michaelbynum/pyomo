from pyomo.common.collections import ComponentMap, ComponentSet, OrderedSet
import pyomo.environ as pe
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.base.block import BlockData
from typing import Any, MutableSet, Tuple, MutableMapping, Optional
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.repn.linear import LinearRepnVisitor, LinearRepn
from collections import deque


def _build_adjacency_maps(
    m: BlockData
) -> Tuple[
    MutableMapping[VarData, MutableSet[ConstraintData]], 
    MutableMapping[ConstraintData, MutableSet[VarData]]
]:
    var_con_map = ComponentMap()
    con_var_map = {}
    for con in m.component_data_objects(pe.Constraint, active=True, descend_into=True):
        assert con not in con_var_map
        con_var_map[con] = ComponentSet()
        for v in identify_variables(con.expr, include_fixed=False):
            if v not in var_con_map:
                var_con_map[v] = OrderedSet()
            var_con_map[v].add(con)
            con_var_map[con].add(v)
    return var_con_map, con_var_map


def _build_model(
    m: BlockData, 
    seen: MutableSet[ConstraintData | VarData],
    var_con_map: MutableMapping[VarData, MutableSet[ConstraintData]],
    con_var_map: MutableMapping[ConstraintData, MutableSet[VarData]],
    num_hops: int,
    seed: VarData,
) -> None:
    if num_hops <= 0:
        return
    if seed in seen:
        return
    seen.add(seed)
    for c in var_con_map[seed]:
        if c in seen:
            continue
        seen.add(c)
        m.cons.add(c.expr)
        if num_hops > 1:
            for v in con_var_map[c]:
                _build_model(m, seen, var_con_map, con_var_map, num_hops-1, v)


def _tighten_var(
    v: VarData,
    var_con_map: MutableMapping[VarData, MutableSet[ConstraintData]],
    con_var_map: MutableMapping[ConstraintData, MutableSet[VarData]],
    num_hops: int,
    solver: SolverBase,
) -> Tuple[Optional[float], Optional[float]]:
    m = pe.ConcreteModel()
    m.obj = pe.Objective(expr=v, sense=pe.minimize)
    m.cons = pe.ConstraintList()
    seen = ComponentSet()
    _build_model(m, seen, var_con_map, con_var_map, num_hops, v)

    res = solver.solve(
        m, 
        raise_exception_on_nonoptimal_result=False, 
        load_solutions=False,
    )
    lb = res.objective_bound

    m.obj.sense = pe.maximize
    res = solver.solve(
        m, 
        raise_exception_on_nonoptimal_result=False, 
        load_solutions=False,
    )
    ub = res.objective_bound
    return lb, ub


def _get_nonlinear_vars(m: BlockData) -> MutableSet[VarData]:
    objs = list(m.component_data_objects(pe.Objective, active=True, descend_into=True))
    cons = list(m.component_data_objects(pe.Constraint, active=True, descend_into=True))
    visitor = LinearRepnVisitor(subexpression_cache={})
    nonlinear_vars = ComponentSet()

    exprs = [obj.expr for obj in objs]
    exprs.extend(con.body for con in cons)

    for expr in [obj.expr for obj in objs]:
        repn: LinearRepn = visitor.walk_expression(expr)
        if repn.nonlinear is None:
            continue
        nonlinear_vars.update(identify_variables(repn.nonlinear, include_fixed=False))

    return nonlinear_vars


class _NonlinearVarSorter:
    def __init__(
        self, 
        var_con_map: MutableMapping[VarData, MutableSet[ConstraintData]],
        con_var_map: MutableMapping[ConstraintData, MutableSet[VarData]],
    ) -> None:
        self.var_con_map = var_con_map
        self.con_var_map = con_var_map

    def __call__(self, v: VarData) -> Tuple[int, int, int, int]:
        """
        return a tuple containing:
        - negate(number of equality constraints)
        - number of unbounded variables in those equality constraints
        - negate(number of inequality constraints)
        - number of unbounded variables in the inequality constraints
        """
        # first, the number of unbounded variables in equality constraints
        n_eq = 0
        n_ineq = 0
        n_eq_unbounded = 0
        n_ineq_unbounded = 0
        for c in self.var_con_map[v]:
            if c.equality:
                n_eq += 1
                for ov in self.con_var_map[c]:
                    if ov.lb is None or ov.ub is None:
                        n_eq_unbounded += 1
            else:
                n_ineq += 1
                for ov in self.con_var_map[c]:
                    if ov.lb is None or ov.ub is None:
                        n_ineq_unbounded += 1
        return (-n_eq, n_eq_unbounded, -n_ineq, n_ineq_unbounded)


def run_obbt(
    m: BlockData,
    solver: SolverBase,
    num_hops: int = 1,
    improvement_tol: float = 0.1,
    max_iter: int = 1000,
) -> None:
    nonlinear_vars = list(_get_nonlinear_vars(m))
    var_con_map, con_var_map = _build_adjacency_maps(m)
    sorter = _NonlinearVarSorter(var_con_map, con_var_map)
    nonlinear_vars.sort(key=sorter)

    to_tighten = deque(nonlinear_vars)
    to_tighten_set = ComponentSet(nonlinear_vars)

    _iter = 0
    while to_tighten:
        if _iter >= max_iter:
            break
        v = to_tighten.popleft()
        if v not in to_tighten_set:
            continue
        to_tighten_set.discard(v)
        vl, vu = _tighten_var(
            v,
            var_con_map,
            con_var_map,
            num_hops,
            solver,
        )
        _iter += 1
        improved = False
        if vl is not None and (v.lb is None or vl > v.lb + improvement_tol + improvement_tol * abs(v.lb)):
            improved = True
            v.setlb(vl)
        elif vu is not None and (v.ub is None or vu < v.ub - improvement_tol - improvement_tol * abs(v.ub)):
            improved = True
            v.setub(vu)
        v.setlb(vl)
        v.setub(vu)
        if improved:
            for con in var_con_map[v]:
                for ov in con_var_map[con]:
                    if ov not in to_tighten_set:
                        to_tighten_set.add(ov)
                        to_tighten.append(ov)
