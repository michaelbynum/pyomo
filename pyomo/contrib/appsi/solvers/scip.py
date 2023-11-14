from pyomo.contrib.appsi.base import (
    Solver, 
    MIPSolverConfig, 
    SolutionLoader, 
    Results, 
    TerminationCondition
)
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _GeneralVarData, ScalarVar
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.base.expression import _GeneralExpressionData, ScalarExpression
from pyomo.core.base.set import Binary, Integers
from pyomo.core.expr import numeric_expr
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, polynomial_degree
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.contrib.appsi.utils.get_objective import get_objective
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.common.log import LogStream
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.config import ConfigValue, NonNegativeInt
import sys
import logging
try:
    import pyscipopt
    from pyscipopt import Model
    scip_available = True
except:
    scip_available = False


logger = logging.getLogger(__name__)


class PyomoToScipVisitor(StreamBasedExpressionVisitor):
    def __init__(
        self, 
        var_map: ComponentMap, 
        scip_model: Model,
        symbol_map: SymbolMap,
        labeler,
    ):
        super().__init__()
        self.var_map = var_map
        self.scip_model = scip_model
        self._symbol_map = symbol_map
        self._labeler = labeler

        self._handlers = h = dict()
        h[_GeneralVarData] = self._handle_var
        h[ScalarVar] = self._handle_var
        h[_ParamData] = self._handle_param
        h[ScalarParam] = self._handle_param
        h[_GeneralExpressionData] = self._handle_named_expression
        h[ScalarExpression] = self._handle_named_expression
        h[float] = self._handle_float
        h[int] = self._handle_float
        h[numeric_expr.NegationExpression] = self._handle_neg
        h[numeric_expr.NPV_NegationExpression] = self._handle_neg
        h[numeric_expr.PowExpression] = self._handle_pow
        h[numeric_expr.NPV_PowExpression] = self._handle_pow
        h[numeric_expr.ProductExpression] = self._handle_mul
        h[numeric_expr.NPV_ProductExpression] = self._handle_mul
        h[numeric_expr.MonomialTermExpression] = self._handle_mul
        h[numeric_expr.DivisionExpression] = self._handle_div
        h[numeric_expr.NPV_DivisionExpression] = self._handle_div
        h[numeric_expr.SumExpression] = self._handle_sum
        h[numeric_expr.LinearExpression] = self._handle_sum
        h[numeric_expr.NPV_SumExpression] = self._handle_sum
        h[numeric_expr.UnaryFunctionExpression] = self._handle_unary
        h[numeric_expr.NPV_UnaryFunctionExpression] = self._handle_unary
        h[numeric_expr.AbsExpression] = self._handle_abs
        h[numeric_expr.NPV_AbsExpression] = self._handle_abs

        self._unary_handlers = uh = dict()
        uh['exp'] = self._handle_exp
        uh['log'] = self._handle_log
        uh['sin'] = self._handle_sin
        uh['cos'] = self._handle_cos
        uh['sqrt'] = self._handle_sqrt
        uh['abs'] = self._handle_abs

    def _handle_var(self, node: _GeneralVarData, data):
        if node in self.var_map:
            return self.var_map[node]
        
        if node.is_fixed():
            return node.value
        
        name = self._symbol_map.getSymbol(node, self._labeler)
        if node.is_binary():
            vtype = 'B'
        elif node.is_integer():
            vtype = 'I'
        else:
            vtype = 'C'
        lb = node.lb
        ub = node.ub
        sv = self.scip_model.addVar(name, vtype, lb, ub)

        self.var_map[node] = sv
        
        return sv
    
    def _handle_param(self, node, data):
        return node.value
    
    def _handle_float(self, node, data):
        return node
    
    def _handle_mul(self, node, data):
        return data[0] * data[1]
    
    def _handle_sum(self, node, data):
        return sum(data)
    
    def _handle_neg(self, node, data):
        return -data[0]
    
    def _handle_pow(self, node, data):
        return data[0]**data[1]
    
    def _handle_div(self, node, data):
        return data[0] / data[1]
    
    def _handle_abs(self, node, data):
        return abs(data[0])
    
    def _handle_exp(self, node, data):
        return pyscipopt.exp(data[0])
    
    def _handle_log(self, node, data):
        return pyscipopt.log(data[0])
    
    def _handle_sin(self, node, data):
        return pyscipopt.sin(data[0])
    
    def _handle_cos(self, node, data):
        return pyscipopt.cos(data[0])
    
    def _handle_sqrt(self, node, data):
        return pyscipopt.sqrt(data[0])
    
    def _handle_named_expression(self, node, data):
        return data[0]
    
    def _handle_unary(self, node, data):
        unary_name = node.getname()
        if unary_name not in self._unary_handlers:
            raise NotImplementedError(f'Cannot convert expression of type {unary_name} to a SCIP expression')
        return self._unary_handlers[unary_name](node, data)

    def exitNode(self, node, data):
        node_type = type(node)
        if node_type not in self._handlers:
            raise NotImplementedError(f'Cannot convert expression of type {node_type} to a SCIP expression')
        return self._handlers[type(node)](node, data)


def create_scip_model(pyomo_model: _BlockData, symbol_map, labeler):
    pm = pyomo_model
    m = Model()

    var_map = ComponentMap()
    visitor = PyomoToScipVisitor(var_map, m, symbol_map=symbol_map, labeler=labeler)

    for con in pm.component_data_objects(Constraint, descend_into=True, active=True):
        scip_body = visitor.walk_expression(con.body)
        if con.lb is not None:
            m.addCons(con.lb <= scip_body)
        if con.ub is not None:
            m.addCons(scip_body <= con.ub)

    obj = get_objective(pm)
    if obj is not None:
        if obj.sense == minimize:
            obj_sense = 'minimize'
        else:
            obj_sense = 'maximize'
        scip_expr = visitor.walk_expression(obj.expr)
        if polynomial_degree(obj.expr) in {0, 1}:
            m.setObjective(scip_expr, obj_sense)
        else:
            obj_var = m.addVar('scip_obj_var', 'C', None, None)
            m.setObjective(obj_var, obj_sense)
            if obj.sense == minimize:
                m.addCons(obj_var >= scip_expr)
            else:
                m.addCons(obj_var <= scip_expr)

    return m, var_map


class ScipConfig(MIPSolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(ScipConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare('logfile', ConfigValue(domain=str))
        self.declare('solver_output_logger', ConfigValue())
        self.declare('log_level', ConfigValue(domain=NonNegativeInt))

        self.logfile = ''
        self.solver_output_logger = logger
        self.log_level = logging.INFO


class ScipResults(Results):
    def __init__(self):
        super().__init__()
        self.wallclock_time = None


class Scip(Solver):
    def __init__(self) -> None:
        super().__init__()
        self._config = ScipConfig()
        self._options = dict()
        self._symbol_map = None

    def available(self):
        if scip_available:
            return self.Availability.FullLicense
        else:
            return self.Availability.NotFound
        
    def version(self):
        return pyscipopt.scip.MAJOR, pyscipopt.scip.MINOR, pyscipopt.scip.PATCH

    @property
    def config(self):
        return self._config
    
    @property
    def options(self):
        return self._options
    
    @property
    def symbol_map(self):
        return self._symbol_map

    def _set_options(self, scip_model, var_map):
        if self.config.time_limit is not None:
            scip_model.setParam('limits/time', self.config.time_limit)

        if self.config.mip_gap is not None:
            scip_model.setParam('limits/gap', self.config.mip_gap)

        if self.config.relax_integrality:
            for v in var_map.values():
                scip_model.chgVarType(v, 'C')

        for key, val in self.options.items():
            scip_model.setParam(key, val)

    def _postsolve(self, scip_model, var_map):
        results = Results()

        status = scip_model.getStatus().lower()
        if status == 'optimal':
            results.termination_condition = TerminationCondition.optimal
        elif status == 'userinterrupt':
            results.termination_condition = TerminationCondition.interrupted
        elif status == 'nodelimit':
            results.termination_condition = TerminationCondition.maxIterations
        elif status == 'totalnodelimit':
            results.termination_condition = TerminationCondition.maxIterations
        elif status == 'stallnodelimit':
            results.termination_condition = TerminationCondition.maxIterations
        elif status == 'timelimit':
            results.termination_condition = TerminationCondition.maxTimeLimit
        elif status == 'terminate':
            results.termination_condition = TerminationCondition.interrupted
        elif status == 'inforunbd':
            results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif status == 'unbounded':
            results.termination_condition = TerminationCondition.unbounded
        elif status == 'infeasible':
            results.termination_condition = TerminationCondition.infeasible
        else:
            results.termination_condition = TerminationCondition.unknown

        if scip_model.getNSols() >= 1:
            results.best_feasible_objective = scip_model.getObjVal()
        else:
            results.best_feasible_objective = None
        results.best_objective_bound = scip_model.getDualbound()

        if scip_model.getNSols() >= 1:
            sol = scip_model.getBestSol()
            primals = dict()
            for pv, sv in var_map.items():
                val = sol[sv]
                primals[id(pv)] = (pv, val)
        else:
            primals = None

        results.solution_loader = SolutionLoader(primals=primals, duals=None, slacks=None, reduced_costs=None)

        if self.config.load_solution:
            if scip_model.getNSols() >= 1:
                if results.termination_condition != TerminationCondition.optimal:
                    logger.warning(
                        'Loading a feasible but suboptimal solution. '
                        'Please set load_solution=False and check '
                        'results.termination_condition and '
                        'results.found_feasible_solution() before loading a solution.'
                    )
                results.solution_loader.load_vars()
            else:
                raise RuntimeError(
                    'A feasible solution was not found, so no solution can be loaded.'
                    'Please set opt.config.load_solution=False and check '
                    'results.termination_condition and '
                    'results.best_feasible_objective before loading a solution.'
                )
            
        return results

    def solve(
        self, model: _BlockData, timer: HierarchicalTimer = None
    ) -> Results:
        if timer is None:
            timer = HierarchicalTimer()
        timer.start('solve')

        StaleFlagManager.mark_all_as_stale()
        self._symbol_map = SymbolMap()
        if self.config.symbolic_solver_labels:
            labeler = TextLabeler()
        else:
            labeler = NumericLabeler('x')

        timer.start('create SCIP model')
        scip_model, var_map = create_scip_model(model, symbol_map=self._symbol_map, labeler=labeler)
        timer.stop('create SCIP model')

        self._set_options(scip_model, var_map)

        ostreams = [
            LogStream(
                level=self.config.log_level, logger=self.config.solver_output_logger
            )
        ]
        if self.config.stream_solver:
            ostreams.append(sys.stdout)
        if self.config.logfile:
            f = open(self.config.logfile, 'w')
            ostreams.append(f)

        try:
            with TeeStream(*ostreams) as t:
                with capture_output(output=t.STDOUT, capture_fd=True):
                    timer.start('scip optimize')
                    scip_model.optimize()
                    timer.stop('scip optimize')
        finally:
            if self.config.logfile:
                f.close()

        res = self._postsolve(scip_model, var_map)

        timer.stop('solve')

        if self.config.report_timing:
            logger.info('\n' + str(timer))

        return res
