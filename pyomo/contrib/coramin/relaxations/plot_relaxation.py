from pyomo.common.dependencies import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    pass
import pyomo.environ as pe
from pyomo.contrib.coramin.utils.pyomo_utils import get_objective
from pyomo.contrib.coramin.utils.coramin_enums import RelaxationSide
from pyomo.contrib.solver.common.base import SolverBase, SolverConfig
from pyomo.contrib.solver.common.results import SolutionStatus, Results
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.core.base.block import BlockData
from .relaxations_base import BaseRelaxationData
from pyomo.core.base.var import VarData
from typing import Sequence, List
from pyomo.contrib.solver.common.config import AutoUpdateConfig
from pyomo.core.base.objective import ObjectiveData

try:
    import tqdm
except ImportError:
    tqdm = None


def _solve(
    m: BlockData, 
    solver: SolverBase, 
    rhs_vars: Sequence[VarData], 
    aux_var: VarData, 
    obj: ObjectiveData,
) -> float:
    obj.activate()
    if solver.is_persistent():
        solver.update_variables(rhs_vars)
        solver.set_objective(obj)
    res: Results = solver.solve(m, load_solutions=False)
    if res.solution_status != SolutionStatus.optimal:
        raise RuntimeError(
            'Could not produce plot because solver did not terminate optimally'
        )
    obj.deactivate()
    return res.incumbent_objective


def _solve_loop(
    m: BlockData, 
    x: VarData, 
    w: VarData, 
    x_list: Sequence[float], 
    solver: SolverBase,
) -> List[float]:
    if solver.is_persistent():
        opt: SolverBase = SolverFactory(solver.name, treat_fixed_vars_as_params=False)
        opt.config = solver.config(preserve_implicit=True)
        opt.set_instance(m)
        config: AutoUpdateConfig = opt.config.auto_updates
        config.check_for_new_or_removed_constraints = False
        config.check_for_new_or_removed_vars = False
        config.check_for_new_or_removed_params = False
        config.check_for_new_objective = False
        config.update_constraints = False
        config.update_vars = True
        config.update_parameters = False
        config.update_named_expressions = False
        config.update_objective = False
    else:
        opt: SolverBase = SolverFactory(solver.name)
        opt.config = solver.config(preserve_implicit=True)
    w_list = list()
    for _xval in x_list:
        x.fix(_xval)
        res: Results = opt.solve(m, load_solutions=False)
        if res.solution_status != SolutionStatus.optimal:
            raise RuntimeError(
                'Could not produce plot because solver did not terminate optimally. SolutionStatus: '
                + str(res.solution_status)
            )
        w_list.append(res.incumbent_objective)
    return w_list


def _plot_2d(
    m: BlockData, 
    relaxation: BaseRelaxationData, 
    solver: SolverBase,
    show_plot: bool, 
    num_pts: int,
):
    rhs_vars = relaxation.get_rhs_vars()
    assert len(rhs_vars) == 1
    x = rhs_vars[0]
    w = relaxation.get_aux_var()

    xlb, xub = x.bounds
    if xlb is None or xub is None:
        raise ValueError('rhs var must have bounds')

    orig_xval = x.value
    orig_wval = w.value

    orig_obj = get_objective(m)
    if orig_obj is not None:
        orig_obj.deactivate()

    x_list = np.linspace(xlb, xub, num_pts)
    x_list = [float(i) for i in x_list]
    w_true = list()

    rhs_expr = relaxation.get_rhs_expr()
    for _x in x_list:
        x.value = _x
        w_true.append(pe.value(rhs_expr))
    plotly_data = [go.Scatter(x=x_list, y=w_true, name=str(rhs_expr))]

    m._plotting_objective = pe.Objective(expr=w)

    if relaxation.relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
        w_min = _solve_loop(m, x, w, x_list, solver)
        plotly_data.append(go.Scatter(x=x_list, y=w_min, name='underestimator'))

    del m._plotting_objective
    m._plotting_objective = pe.Objective(expr=w, sense=pe.maximize)

    if relaxation.relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
        w_max = _solve_loop(m, x, w, x_list, solver)
        plotly_data.append(go.Scatter(x=x_list, y=w_max, name='overestimator'))

    fig = go.Figure(data=plotly_data)
    if show_plot:
        fig.show()

    x.unfix()
    x.value = orig_xval
    w.value = orig_wval
    del m._plotting_objective
    if orig_obj is not None:
        orig_obj.activate()


def _plot_3d(
    m: BlockData, 
    relaxation: BaseRelaxationData, 
    solver: SolverBase, 
    show_plot: bool, 
    num_pts: int,
):
    if solver.is_persistent():
        opt: SolverBase = SolverFactory(solver.name, treat_fixed_vars_as_params=False)
        opt.config = solver.config(preserve_implicit=True)
        opt.set_instance(m)
        config: AutoUpdateConfig = opt.config.auto_updates
        config.check_for_new_or_removed_constraints = False
        config.check_for_new_or_removed_vars = False
        config.check_for_new_or_removed_params = False
        config.check_for_new_objective = False
        config.update_constraints = False
        config.update_vars = False
        config.update_parameters = False
        config.update_named_expressions = False
        config.update_objective = False
    else:
        opt: SolverBase = SolverFactory(solver.name)
        opt.config = solver.config(preserve_implicit=True)

    rhs_vars = relaxation.get_rhs_vars()
    x, y = rhs_vars
    w = relaxation.get_aux_var()


    if not x.has_lb() or not x.has_ub() or not y.has_lb() or not y.has_ub():
        raise ValueError('rhs vars must have bounds')

    orig_xval = x.value
    orig_yval = y.value
    orig_wval = w.value

    orig_obj = get_objective(m)
    if orig_obj is not None:
        orig_obj.deactivate()

    m._underestimator_obj = pe.Objective(expr=w)
    m._overestimator_obj = pe.Objective(expr=w, sense=pe.maximize)
    m._underestimator_obj.deactivate()
    m._overestimator_obj.deactivate()
    if opt.is_persistent():
        opt.set_instance(m)

    x_list = np.linspace(x.lb, x.ub, num_pts)
    y_list = np.linspace(y.lb, y.ub, num_pts)
    x_list = [float(i) for i in x_list]
    y_list = [float(i) for i in y_list]
    w_true = np.empty((num_pts, num_pts), dtype=float)
    w_min = np.empty((num_pts, num_pts), dtype=float)
    w_max = np.empty((num_pts, num_pts), dtype=float)

    rhs_expr = relaxation.get_rhs_expr()

    def sub_loop(x_ndx, _x):
        x.fix(_x)
        for y_ndx, _y in enumerate(y_list):
            y.fix(_y)
            w_true[x_ndx, y_ndx] = pe.value(rhs_expr)
            if relaxation.relaxation_side in {
                RelaxationSide.UNDER,
                RelaxationSide.BOTH,
            }:
                obj_val = _solve(
                    m=m,
                    solver=opt,
                    rhs_vars=rhs_vars,
                    aux_var=w,
                    obj=m._underestimator_obj,
                )
                w_min[x_ndx, y_ndx] = obj_val
            if relaxation.relaxation_side in {
                RelaxationSide.OVER, 
                RelaxationSide.BOTH,
            }:
                obj_val = _solve(
                    m=m,
                    solver=opt,
                    rhs_vars=rhs_vars,
                    aux_var=w,
                    obj=m._overestimator_obj,
                )
                w_max[x_ndx, y_ndx] = obj_val

    if tqdm is not None:
        for x_ndx, _x in tqdm.tqdm(list(enumerate(x_list))):
            sub_loop(x_ndx, _x)
    else:
        for x_ndx, _x in enumerate(x_list):
            sub_loop(x_ndx, _x)

    plotly_data = list()
    plotly_data.append(go.Surface(x=x_list, y=y_list, z=w_true, name=str(rhs_expr)))
    if relaxation.relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
        plotly_data.append(
            go.Surface(x=x_list, y=y_list, z=w_min, name='underestimator')
        )
    if relaxation.relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
        plotly_data.append(
            go.Surface(x=x_list, y=y_list, z=w_max, name='overestimator')
        )

    fig = go.Figure(data=plotly_data)
    if show_plot:
        fig.show()

    x.unfix()
    y.unfix()
    x.value = orig_xval
    y.value = orig_yval
    w.value = orig_wval
    del m._underestimator_obj
    del m._overestimator_obj
    if orig_obj is not None:
        orig_obj.activate()


def plot_relaxation(
    m: BlockData, 
    relaxation: BaseRelaxationData, 
    solver: SolverBase, 
    show_plot: bool = True, 
    num_pts: int = 100,
):
    rhs_vars = relaxation.get_rhs_vars()

    if len(rhs_vars) == 1:
        _plot_2d(m, relaxation, solver, show_plot, num_pts)
    elif len(rhs_vars) == 2:
        _plot_3d(m, relaxation, solver, show_plot, num_pts)
    else:
        raise NotImplementedError(
            'Cannot generate plot for relaxation with more than 2 RHS vars'
        )
