import pyomo.environ as pe
from pyomo.core.base.block import BlockData
from pyomo.core.base.objective import ObjectiveData, Objective
from pyomo.core.base.constraint import ConstraintData, Constraint
from pyomo.core.base.var import VarData
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.simplification import Simplifier
from weakref import WeakKeyDictionary
from typing import Optional, List
from pyomo.common.errors import PyomoException


def get_objective(m: BlockData) -> Optional[ObjectiveData]:
    """
    Assert that there is only one active objective in m and return it.

    Parameters
    ----------
    m: BlockData

    Returns
    -------
    obj: Optional[ObjectiveData]
    """
    obj = None
    for o in m.component_data_objects(
        Objective, descend_into=True, active=True, sort=True
    ):
        if obj is not None:
            raise PyomoException('Found multiple active objectives')
        obj = o
    return obj


_var_cache = WeakKeyDictionary()


def identify_variables_with_cache(
    con: ConstraintData, 
    include_fixed: bool,
) -> List[VarData]:
    e = con.expr
    if con in _var_cache and _var_cache[con][1] is e:
        vlist = _var_cache[con][0]
    else:
        vlist = list(identify_variables(e, include_fixed=True))
    _var_cache[con] = (vlist, e)
    if not include_fixed:
        vlist = [i for i in vlist if not i.fixed]
    return vlist


def active_vars(
    m: BlockData, 
    include_fixed: bool,
) -> List[VarData]:
    seen = {}
    for c in m.component_data_objects(Constraint, active=True, descend_into=True):
        for v in identify_variables_with_cache(c, include_fixed=include_fixed):
            seen[id(v)] = None
    obj = get_objective(m)
    if obj is not None:
        for v in identify_variables(obj.expr, include_fixed=include_fixed):
            seen[id(v)] = None
    return list(seen.keys())


def active_cons(m: BlockData) -> List[ConstraintData]:
    return [c for c in m.component_data_objects(Constraint, active=True, descend_into=True)]


simplifier = Simplifier()


def simplify_expr(expr):
    new_expr = simplifier.simplify(expr)
    if is_fixed(new_expr):
        new_expr = pe.value(new_expr)
    return new_expr
