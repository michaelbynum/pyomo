#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import re
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.repn import generate_standard_repn
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.objective import Objective
from pyomo.common.config import ConfigValue, add_docstring_list
from pyomo.solver.base import MIPSolver, ResultsBase
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler


logger = logging.getLogger('pyomo.solvers')


class DegreeError(ValueError):
    pass


class GurobiPersistentResults(ResultsBase):
    def __init__(self, solver):
        super(GurobiPersistentResults, self).__init__()
        self._solver = solver

    def load_solution(self, model, solution_number=0):
        self._solver.setParam('SolutionNumber', solution_number)
        self._solver.load_vars()


def _is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


def _get_objective(block):
    obj = None
    for o in block.component_data_objects(Objective, descend_into=True, active=True, sort=True):
        if obj is not None:
            raise ValueError('Multiple active objectives found')
        obj = o
    return obj


class _MutableLinearCoefficient(object):
    def __init__(self):
        self.expr = None
        self.var = None
        self.con = None
        self.gurobi_model = None

    def update(self):
        self.gurobi_model.chgCoeff(self.con, self.var, value(self.expr))

    def __str__(self):
        s = str(self.var) + ': ' + str(self.expr)
        return s


class _MutableRangeConstant(object):
    def __init__(self):
        self.lhs_expr = None
        self.rhs_expr = None
        self.con = None
        self.slack = None

    def update(self):
        rhs_val = value(self.rhs_expr)
        lhs_val = value(self.lhs_expr)
        self.con.rhs = rhs_val
        self.slack.ub = rhs_val - lhs_val


class _MutableConstant(object):
    def __init__(self):
        self.expr = None
        self.con = None

    def update(self):
        self.con.rhs = value(self.expr)


class _MutableQuadraticConstraint(object):
    def __init__(self, gurobi_model, gurobi_con, constant, linear_coefs, quadratic_coefs):
        self.con = gurobi_con
        self.gurobi_model = gurobi_model
        self.constant = constant
        self.last_constant_value = value(self.constant)
        self.linear_coefs = linear_coefs
        self.last_linear_coef_values = [value(i.expr) for i in self.linear_coefs]
        self.quadratic_coefs = quadratic_coefs
        self.last_quadratic_coef_values = [value(i.expr) for i in self.quadratic_coefs]

    def get_updated_expression(self):
        gurobi_expr = self.gurobi_model.getQCRow(self.con)
        for ndx, coef in enumerate(self.linear_coefs):
            new_coef_value = value(coef.expr) - self.last_linear_coef_values[ndx]
            gurobi_expr += new_coef_value * coef.var
            self.last_linear_coef_values[ndx] = new_coef_value
        for ndx, coef in enumerate(self.quadratic_coefs):
            new_coef_value = value(coef.expr) - self.last_quadratic_coef_values[ndx]
            gurobi_expr += new_coef_value * coef.var1 * coef.var2
            self.last_quadratic_coef_values[ndx] = new_coef_value
        return gurobi_expr

    def get_updated_rhs(self):
        return value(self.constant.expr)


class _MutableObjective(object):
    def __init__(self, gurobi_model, constant, linear_coefs, quadratic_coefs):
        self.gurobi_model = gurobi_model
        self.constant = constant
        self.linear_coefs = linear_coefs
        self.quadratic_coefs = quadratic_coefs
        self.last_quadratic_coef_values = [value(i.expr) for i in self.quadratic_coefs]

    def get_updated_expression(self):
        for ndx, coef in enumerate(self.linear_coefs):
            coef.var.obj = value(coef.expr)
        self.gurobi_model.ObjCon = value(self.constant.expr)

        gurobi_expr = None
        for ndx, coef in enumerate(self.quadratic_coefs):
            if value(coef.expr) != self.last_quadratic_coef_values[ndx]:
                if gurobi_expr is None:
                    self.gurobi_model.update()
                    gurobi_expr = self.gurobi_model.getObjective()
                new_coef_value = value(coef.expr) - self.last_quadratic_coef_values[ndx]
                gurobi_expr += new_coef_value * coef.var1 * coef.var2
                self.last_quadratic_coef_values[ndx] = new_coef_value
        return gurobi_expr


class _MutableQuadraticCoefficient(object):
    def __init__(self):
        self.expr = None
        self.var1 = None
        self.var2 = None


@SolverFactory.register('gurobi_persistent_new', doc='Direct python interface to Gurobi with automatic updates')
class GurobiPersistentNew(MIPSolver):
    """
    Direct interface to Gurobi
    """
    CONFIG = MIPSolver.CONFIG()

    CONFIG.declare('symbolic_solver_labels', ConfigValue(default=False, domain=bool,
                                                         doc='If True, the gurobi variable and constraint names '
                                                             'will match those of the pyomo variables and constrains'))
    CONFIG.declare('stream_solver', ConfigValue(default=False, domain=bool,
                                                doc='If True, show the Gurobi output'))
    CONFIG.declare('load_solutions', ConfigValue(default=True, domain=bool,
                                                 doc='If True, load the solution back into the Pyomo model'))

    __doc__ = add_docstring_list(__doc__, CONFIG)

    def __init__(self):
        super(GurobiAuto, self).__init__()

        self._pyomo_model = None
        self._solver_model = None
        self._symbol_map = SymbolMap()
        self._labeler = None
        self._pyomo_var_to_solver_var_map = dict()
        self._solver_var_to_pyomo_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = dict()
        self._vars_referenced_by_con = dict()
        self._vars_referenced_by_obj = dict()
        self._objective = None
        self._objective_expr = None
        self._referenced_variables = dict()
        self._referenced_params = dict()
        self._range_constraints = set()
        self._tmp_config = None
        self._tmp_options = None
        self._mutable_helpers = list()
        self._mutable_quadratic_helpers = list()
        self._mutable_objective = None

        try:
            import gurobipy
            self._gurobipy = gurobipy
            self._python_api_exists = True
        except Exception as e:
            logger.warning("Import of gurobipy failed - gurobi message=" + str(e) + "\n")
            self._python_api_exists = False

    def available(self):
        return self._python_api_exists

    def license_status(self):
        try:
            tmp = self._gurobipy.Model()
            return True
        except self._gurobipy.GurobiError:
            return False

    def solve(self, model, options=None, **config_options):
        """
        solve a model
        """
        if options is None:
            options = dict()
        self._tmp_options = self.options(options, preserve_implicit=True)
        self._tmp_config = self.config(config_options)
        if model is self._pyomo_model:
            self._update()
        else:
            self._set_instance(model)
        self._apply_solver()
        return self._postsolve()

    def _apply_solver(self):
        if self._tmp_config.stream_solver:
            self._solver_model.setParam('OutputFlag', 1)
        else:
            self._solver_model.setParam('OutputFlag', 0)

        for key, option in self._tmp_options.items():
            self._solver_model.setParam(key, option)
        self._solver_model.optimize()

    def _add_var(self, var):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._gurobi_vtype_from_var(var)
        if var.has_lb():
            lb = value(var.lb)
        else:
            lb = -self._gurobipy.GRB.INFINITY
        if var.has_ub():
            ub = value(var.ub)
        else:
            ub = self._gurobipy.GRB.INFINITY
        if var.is_fixed():
            lb = value(var.value)
            ub = value(var.value)

        gurobipy_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype, name=varname)

        self._pyomo_var_to_solver_var_map[id(var)] = gurobipy_var
        self._solver_var_to_pyomo_var_map[id(gurobipy_var)] = var
        self._referenced_variables[id(var)] = 0

    def _set_instance(self, model):
        self._pyomo_model = model
        self._symbol_map = SymbolMap()
        self._labeler = None
        self._pyomo_var_to_solver_var_map = dict()
        self._solver_var_to_pyomo_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = dict()
        self._vars_referenced_by_con = dict()
        self._vars_referenced_by_obj = dict()
        self._objective = None
        self._referenced_variables = dict()
        self._referenced_params = dict()
        self._param_to_con_map = dict()
        self._range_constraints = set()
        self._mutable_helpers = list()
        self._mutable_quadratic_helpers = list()

        if self._tmp_config.symbolic_solver_labels:
            self._labeler = TextLabeler()
        else:
            self._labeler = NumericLabeler('x')

        if model.name is not None:
            self._solver_model = self._gurobipy.Model(model.name)
        else:
            self._solver_model = self._gurobipy.Model()

        for var in model.component_data_objects(Var, descend_into=True, sort=True):
            self._add_var(var)

        for con in model.component_data_objects(Constraint, descend_into=True, active=True, sort=True):
            self._add_constraint(con)

        for con in model.component_data_objects(SOSConstraint, descend_into=True, active=True, sort=True):
            self._add_sos_constraint(con)

        self._set_objective(_get_objective(model))

    def _get_expr_from_pyomo_expr(self, expr):
        mutable_linear_coefficients = list()
        mutable_quadratic_coefficients = list()
        repn = generate_standard_repn(expr, quadratic=True, compute_values=False)
        referenced_vars = ComponentSet()

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > 2):
            raise DegreeError('GurobiAuto does not support expressions of degree {0}.'.format(degree))

        if len(repn.linear_vars) > 0:
            referenced_vars.update(repn.linear_vars)
            linear_coef_vals = list()
            for ndx, coef in enumerate(repn.linear_coefs):
                if not is_constant(coef):
                    mutable_linear_coefficient = _MutableLinearCoefficient()
                    mutable_linear_coefficient.expr = coef
                    mutable_linear_coefficient.var = self._pyomo_var_to_solver_var_map[id(repn.linear_vars[ndx])]
                    mutable_linear_coefficients.append(mutable_linear_coefficient)
                linear_coef_vals.append(value(coef))
            new_expr = self._gurobipy.LinExpr(linear_coef_vals, [self._pyomo_var_to_solver_var_map[id(i)] for i in repn.linear_vars])
        else:
            new_expr = 0.0

        for ndx,v in enumerate(repn.quadratic_vars):
            x,y = v
            gurobi_x = self._pyomo_var_to_solver_var_map[id(x)]
            gurobi_y = self._pyomo_var_to_solver_var_map[id(y)]
            coef = repn.quadratic_coefs[ndx]
            if not is_constant(coef):
                mutable_quadratic_coefficient = _MutableQuadraticCoefficient()
                mutable_quadratic_coefficient.expr = coef
                mutable_quadratic_coefficient.var1 = gurobi_x
                mutable_quadratic_coefficient.var2 = gurobi_y
                mutable_quadratic_coefficients.append(mutable_quadratic_coefficient)
            coef_val = value(coef)
            new_expr += coef_val * gurobi_x * gurobi_y
            referenced_vars.add(x)
            referenced_vars.add(y)

        #if not is_constant(repn.constant):
        #    mutable_constant = _MutableConstant()
        #    mutable_constant.expr = repn.constant
        #constant_val = value(repn.constant)
        #new_expr += constant_val

        return new_expr, referenced_vars, repn.constant, mutable_linear_coefficients, mutable_quadratic_coefficients

    def _add_constraint(self, con):
        assert con.active

        conname = self._symbol_map.getSymbol(con, self._labeler)

        (gurobi_expr,
         referenced_vars,
         repn_constant,
         mutable_linear_coefficients,
         mutable_quadratic_coefficients) = self._get_expr_from_pyomo_expr(con.body)

        if gurobi_expr.__class__ is self._gurobipy.LinExpr:
            if con.equality:
                rhs_expr = con.lower - repn_constant
                rhs_val = value(rhs_expr)
                gurobipy_con = self._solver_model.addLConstr(gurobi_expr,
                                                             self._gurobipy.GRB.EQUAL,
                                                             rhs_val,
                                                             name=conname)
                if not is_constant(rhs_expr):
                    mutable_constant = _MutableConstant()
                    mutable_constant.expr = rhs_expr
                    mutable_constant.con = gurobipy_con
                    self._mutable_helpers.append(mutable_constant)
            elif con.has_lb() and con.has_ub():
                lhs_expr = con.lower - repn_constant
                rhs_expr = con.upper - repn_constant
                lhs_val = value(lhs_expr)
                rhs_val = value(rhs_expr)
                gurobipy_con = self._solver_model.addRange(gurobi_expr, lhs_val, rhs_val, name=conname)
                self._range_constraints.add(con)
                if not is_constant(lhs_expr) or not is_constant(rhs_expr):
                    mutable_range_constant = _MutableRangeConstant()
                    mutable_range_constant.lhs_expr = lhs_expr
                    mutable_range_constant.rhs_expr = rhs_expr
                    mutable_range_constant.con = gurobipy_con
                    mutable_range_constant.slack = self._solver_model.getVarByName('Rg'+conname)
                    self._mutable_helpers.append(mutable_range_constant)
            elif con.has_lb():
                rhs_expr = con.lower - repn_constant
                rhs_val = value(rhs_expr)
                gurobipy_con = self._solver_model.addLConstr(gurobi_expr, self._gurobipy.GRB.GREATER_EQUAL, rhs_val, name=conname)
                if not is_constant(rhs_expr):
                    mutable_constant = _MutableConstant()
                    mutable_constant.expr = rhs_expr
                    mutable_constant.con = gurobipy_con
                    self._mutable_helpers.append(mutable_constant)
            elif con.has_ub():
                rhs_expr = con.upper - repn_constant
                rhs_val = value(rhs_expr)
                gurobipy_con = self._solver_model.addLConstr(gurobi_expr, self._gurobipy.GRB.LESS_EQUAL, rhs_val, name=conname)
                if not is_constant(rhs_expr):
                    mutable_constant = _MutableConstant()
                    mutable_constant.expr = rhs_expr
                    mutable_constant.con = gurobipy_con
                    self._mutable_helpers.append(mutable_constant)
            else:
                raise ValueError("Constraint does not have a lower "
                                 "or an upper bound: {0} \n".format(con))
            for tmp in mutable_linear_coefficients:
                tmp.con = gurobipy_con
                tmp.gurobi_model = self._solver_model
            self._mutable_helpers.extend(mutable_linear_coefficients)
        elif gurobi_expr.__class__ is self._gurobipy.QuadExpr:
            if con.equality:
                raise NotImplementedError('Quadratic equality constraints are not supported')
            elif con.has_lb() and con.has_ub():
                raise NotImplementedError('Quadratic range constraints are not supported')
            elif con.has_lb():
                rhs_expr = con.lower - repn_constant
                rhs_val = value(rhs_expr)
                gurobipy_con = self._solver_model.addQConstr(gurobi_expr, self._gurobipy.GRB.GREATER_EQUAL, rhs_val, name=conname)
            elif con.has_ub():
                rhs_expr = con.upper - repn_constant
                rhs_val = value(rhs_expr)
                gurobipy_con = self._solver_model.addQConstr(gurobi_expr, self._gurobipy.GRB.LESS_EQUAL, rhs_val, name=conname)
            else:
                raise ValueError("Constraint does not have a lower "
                                 "or an upper bound: {0} \n".format(con))
            if len(mutable_linear_coefficients) > 0 or len(mutable_quadratic_coefficients) > 0 or not is_constant(repn_constant):
                mutable_constant = _MutableConstant()
                mutable_constant.expr = repn_constant
                mutable_quadratic_constraint = _MutableQuadraticConstraint(self._solver_model, gurobipy_con,
                                                                           mutable_constant,
                                                                           mutable_linear_coefficients,
                                                                           mutable_quadratic_coefficients)
                self._mutable_quadratic_helpers.append(mutable_quadratic_constraint)
            
        for var in referenced_vars:
            self._referenced_variables[id(var)] += 1
        self._vars_referenced_by_con[id(con)] = referenced_vars
        self._pyomo_con_to_solver_con_map[id(con)] = gurobipy_con
        self._solver_con_to_pyomo_con_map[id(gurobipy_con)] = con

    def _add_sos_constraint(self, con):
        assert con.active()

        conname = self._symbol_map.getSymbol(con, self._labeler)
        level = con.level
        if level == 1:
            sos_type = self._gurobipy.GRB.SOS_TYPE1
        elif level == 2:
            sos_type = self._gurobipy.GRB.SOS_TYPE2
        else:
            raise ValueError("Solver does not support SOS "
                             "level {0} constraints".format(level))

        gurobi_vars = []
        weights = []

        self._vars_referenced_by_con[id(con)] = ComponentSet()

        if hasattr(con, 'get_items'):
            # aml sos constraint
            sos_items = list(con.get_items())
        else:
            # kernel sos constraint
            sos_items = list(con.items())

        for v, w in sos_items:
            self._vars_referenced_by_con[id(con)].add(v)
            gurobi_vars.append(self._pyomo_var_to_solver_var_map[id(v)])
            self._referenced_variables[id(v)] += 1
            weights.append(w)

        gurobipy_con = self._solver_model.addSOS(sos_type, gurobi_vars, weights)
        self._pyomo_con_to_solver_con_map[id(con)] = gurobipy_con
        self._solver_con_to_pyomo_con_map[id(gurobipy_con)] = con

    def _remove_constraint(self, con):
        solver_con = self._pyomo_con_to_solver_con_map[id(con)]
        self._solver_model.remove(solver_con)
        self._symbol_map.removeSymbol(con)
        self._labeler.remove_obj(con)
        for var in self._vars_referenced_by_con[id(con)]:
            self._referenced_variables[id(var)] -= 1
        del self._vars_referenced_by_con[id(con)]
        del self._pyomo_con_to_solver_con_map[id(con)]
        del self._solver_con_to_pyomo_con_map[id(solver_con)]
        self._range_constraints.discard(con)

    def _remove_sos_constraint(self, con):
        solver_sos_con = self._pyomo_con_to_solver_con_map[id(con)]
        self._symbol_map.removeSymbol(con)
        self._labeler.remove_obj(con)
        for var in self._vars_referenced_by_con[id(con)]:
            self._referenced_variables[id(var)] -= 1
        del self._vars_referenced_by_con[id(con)]
        del self._pyomo_con_to_solver_con_map[id(con)]
        del self._solver_con_to_pyomo_con_map[id(solver_con)]

    def _remove_var(self, solver_var):
        if self._referenced_variables[id(var)] != 0:
            raise ValueError('Cannot remove Var {0} because it is still referenced by the objective or one or more constraints'.format(var))
        solver_var = self._pyomo_var_to_solver_var_map[id(var)]
        self._solver_model.remove(solver_var)
        self._symbol_map.removeSymbol(var)
        self._labeler.remove_obj(var)
        del self._referenced_variables[id(var)]
        del self._pyomo_var_to_solver_var_map[id(var)]
        del self._solver_var_to_pyomo_var_map[id(solver_var)]

    def _gurobi_vtype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate gurobi variable type
        :param var: pyomo.core.base.var.Var
        :return: gurobipy.GRB.CONTINUOUS or gurobipy.GRB.BINARY or gurobipy.GRB.INTEGER
        """
        if var.is_binary():
            vtype = self._gurobipy.GRB.BINARY
        elif var.is_integer():
            vtype = self._gurobipy.GRB.INTEGER
        elif var.is_continuous():
            vtype = self._gurobipy.GRB.CONTINUOUS
        else:
            raise ValueError('Variable domain type is not recognized for {0}'.format(var.domain))
        return vtype

    def _set_objective(self, obj):
        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[id(var)] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None
            self._objective_expr = None

        if obj.active is False:
            raise ValueError('Cannot add inactive objective to solver.')

        if obj.sense == minimize:
            sense = self._gurobipy.GRB.MINIMIZE
        elif obj.sense == maximize:
            sense = self._gurobipy.GRB.MAXIMIZE
        else:
            raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

        (gurobi_expr,
         referenced_vars,
         repn_constant,
         mutable_linear_coefficients,
         mutable_quadratic_coefficients) = self._get_expr_from_pyomo_expr(obj.expr)
        mutable_constant = _MutableConstant()
        mutable_constant.expr = repn_constant
        mutable_objective = _MutableObjective(self._solver_model,
                                              mutable_constant,
                                              mutable_linear_coefficients,
                                              mutable_quadratic_coefficients)
        self._mutable_objective = mutable_objective

        for var in referenced_vars:
            self._referenced_variables[id(var)] += 1

        self._solver_model.setObjective(gurobi_expr + value(repn_constant), sense=sense)
        self._objective = obj
        self._objective_expr = obj.expr
        self._vars_referenced_by_obj = referenced_vars

    def _postsolve(self):
        # the only suffixes that we extract from GUROBI are
        # constraint duals, constraint slacks, and variable
        # reduced-costs. scan through the solver suffix list
        # and throw an exception if the user has specified
        # any others.
        suffixes = list(self._pyomo_model.component_objects(Suffix, active=True, descend_into=False, sort=True))
        extract_duals = False
        extract_slacks = False
        extract_reduced_costs = False
        for suffix in suffixes:
            flag = False
            if re.match(suffix, "dual"):
                extract_duals = True
                flag = True
            if re.match(suffix, "slack"):
                extract_slacks = True
                flag = True
            if re.match(suffix, "rc"):
                extract_reduced_costs = True
                flag = True
            if not flag:
                raise RuntimeError("***The gurobi_direct solver plugin cannot extract solution suffix="+suffix)

        gprob = self._solver_model
        grb = self._gurobipy.GRB
        status = gprob.Status

        if gprob.getAttr(self._gurobipy.GRB.Attr.IsMIP):
            if extract_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
            if extract_duals:
                logger.warning("Cannot get duals for MIP.")
            extract_reduced_costs = False
            extract_duals = False

        results = SolverResults()
        results.solver.wallclock_time = gprob.Runtime

        if status == grb.LOADED:  # problem is loaded, but no solution
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = "Model is loaded, but no solution information is available."
            results.solver.termination_condition = TerminationCondition.error
        elif status == grb.OPTIMAL:  # optimal
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = "Model was solved to optimality (subject to tolerances), " \
                                                      "and an optimal solution is available."
            results.solver.termination_condition = TerminationCondition.optimal
        elif status == grb.INFEASIBLE:
            results.solver.status = SolverStatus.warning
            results.solver.termination_message = "Model was proven to be infeasible"
            results.solver.termination_condition = TerminationCondition.infeasible
        elif status == grb.INF_OR_UNBD:
            results.solver.status = SolverStatus.warning
            results.solver.termination_message = "Problem proven to be infeasible or unbounded."
            results.solver.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif status == grb.UNBOUNDED:
            results.solver.status = SolverStatus.warning
            results.solver.termination_message = "Model was proven to be unbounded."
            results.solver.termination_condition = TerminationCondition.unbounded
        elif status == grb.CUTOFF:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = "Optimal objective for model was proven to be worse than the " \
                                                      "value specified in the Cutoff parameter. No solution " \
                                                      "information is available."
            results.solver.termination_condition = TerminationCondition.minFunctionValue
        elif status == grb.ITERATION_LIMIT:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = "Optimization terminated because the total number of simplex " \
                                                      "iterations performed exceeded the value specified in the " \
                                                      "IterationLimit parameter."
            results.solver.termination_condition = TerminationCondition.maxIterations
        elif status == grb.NODE_LIMIT:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = "Optimization terminated because the total number of " \
                                                      "branch-and-cut nodes explored exceeded the value specified " \
                                                      "in the NodeLimit parameter"
            results.solver.termination_condition = TerminationCondition.maxEvaluations
        elif status == grb.TIME_LIMIT:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = "Optimization terminated because the time expended exceeded " \
                                                      "the value specified in the TimeLimit parameter."
            results.solver.termination_condition = TerminationCondition.maxTimeLimit
        elif status == grb.SOLUTION_LIMIT:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = "Optimization terminated because the number of solutions found " \
                                                      "reached the value specified in the SolutionLimit parameter."
            results.solver.termination_condition = TerminationCondition.unknown
        elif status == grb.INTERRUPTED:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = "Optimization was terminated by the user."
            results.solver.termination_condition = TerminationCondition.error
        elif status == grb.NUMERIC:
            results.solver.status = SolverStatus.error
            results.solver.termination_message = "Optimization was terminated due to unrecoverable numerical " \
                                                      "difficulties."
            results.solver.termination_condition = TerminationCondition.error
        elif status == grb.SUBOPTIMAL:
            results.solver.status = SolverStatus.warning
            results.solver.termination_message = "Unable to satisfy optimality tolerances; a sub-optimal " \
                                                      "solution is available."
            results.solver.termination_condition = TerminationCondition.other
        # note that USER_OBJ_LIMIT was added in Gurobi 7.0, so it may not be present
        elif (status is not None) and \
             (status == getattr(grb,'USER_OBJ_LIMIT',None)):
            results.solver.status = SolverStatus.aborted
            results.solver.termination_message = "User specified an objective limit " \
                                                      "(a bound on either the best objective " \
                                                      "or the best bound), and that limit has " \
                                                      "been reached. Solution is available."
            results.solver.termination_condition = TerminationCondition.other
        else:
            results.solver.status = SolverStatus.error
            results.solver.termination_message = \
                ("Unhandled Gurobi solve status "
                 "("+str(status)+")")
            results.solver.termination_condition = TerminationCondition.error

        results.problem.name = gprob.ModelName

        if gprob.ModelSense == 1:
            results.problem.sense = minimize
        elif gprob.ModelSense == -1:
            results.problem.sense = maximize
        else:
            raise RuntimeError('Unrecognized gurobi objective sense: {0}'.format(gprob.ModelSense))

        results.problem.upper_bound = None
        results.problem.lower_bound = None
        if (gprob.NumBinVars + gprob.NumIntVars) == 0:
            try:
                results.problem.upper_bound = gprob.ObjVal
                results.problem.lower_bound = gprob.ObjVal
            except (self._gurobipy.GurobiError, AttributeError):
                pass
        elif gprob.ModelSense == 1:  # minimizing
            try:
                results.problem.upper_bound = gprob.ObjVal
            except (self._gurobipy.GurobiError, AttributeError):
                pass
            try:
                results.problem.lower_bound = gprob.ObjBound
            except (self._gurobipy.GurobiError, AttributeError):
                pass
        elif gprob.ModelSense == -1:  # maximizing
            try:
                results.problem.upper_bound = gprob.ObjBound
            except (self._gurobipy.GurobiError, AttributeError):
                pass
            try:
                results.problem.lower_bound = gprob.ObjVal
            except (self._gurobipy.GurobiError, AttributeError):
                pass
        else:
            raise RuntimeError('Unrecognized gurobi objective sense: {0}'.format(gprob.ModelSense))

        results.problem.number_of_constraints = gprob.NumConstrs + gprob.NumQConstrs + gprob.NumSOS
        results.problem.number_of_nonzeros = gprob.NumNZs
        results.problem.number_of_variables = gprob.NumVars
        results.problem.number_of_binary_variables = gprob.NumBinVars
        results.problem.number_of_integer_variables = gprob.NumIntVars
        results.problem.number_of_continuous_variables = gprob.NumVars - gprob.NumIntVars - gprob.NumBinVars
        results.problem.number_of_objectives = 1
        results.problem.number_of_solutions = gprob.SolCount

        if self._tmp_config.load_solutions:
            if gprob.SolCount > 0:
                self.load_vars()

                if extract_reduced_costs:
                    self.load_rc()

                if extract_duals:
                    self.load_duals()

                if extract_slacks:
                    self.load_slacks()

        return results

    def load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = self._solver_var_to_pyomo_var_map.values()

        gurobi_vars_to_load = [var_map[id(pyomo_var)] for pyomo_var in vars_to_load]
        vals = self._solver_model.getAttr("Xn", gurobi_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[id(var)] > 0:
                var.value = val

    def load_rc(self, vars_to_load=None):
        if not hasattr(self._pyomo_model, 'rc'):
            self._pyomo_model.rc = Suffix(direction=Suffix.IMPORT)
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        rc = self._pyomo_model.rc
        if vars_to_load is None:
            vars_to_load = self._solver_var_to_pyomo_var_map.values()

        gurobi_vars_to_load = [var_map[id(pyomo_var)] for pyomo_var in vars_to_load]
        vals = self._solver_model.getAttr("Rc", gurobi_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[id(var)] > 0:
                rc[var] = val

    def load_duals(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'dual'):
            self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        dual = self._pyomo_model.dual

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.getConstrs()
            quadratic_cons_to_load = self._solver_model.getQConstrs()
        else:
            gurobi_cons_to_load = set([con_map[id(pyomo_con)] for pyomo_con in cons_to_load])
            linear_cons_to_load = list(gurobi_cons_to_load.intersection(set(self._solver_model.getConstrs())))
            quadratic_cons_to_load = list(gurobi_cons_to_load.intersection(set(self._solver_model.getQConstrs())))
        linear_vals = self._solver_model.getAttr("Pi", linear_cons_to_load)
        quadratic_vals = self._solver_model.getAttr("QCPi", quadratic_cons_to_load)

        for gurobi_con, val in zip(linear_cons_to_load, linear_vals):
            pyomo_con = reverse_con_map[id(gurobi_con)]
            dual[pyomo_con] = val
        for gurobi_con, val in zip(quadratic_cons_to_load, quadratic_vals):
            pyomo_con = reverse_con_map[id(gurobi_con)]
            dual[pyomo_con] = val

    def load_slacks(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'slack'):
            self._pyomo_model.slack = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        slack = self._pyomo_model.slack

        gurobi_range_con_vars = set(self._solver_model.getVars()) - set(self._pyomo_var_to_solver_var_map.values())

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.getConstrs()
            quadratic_cons_to_load = self._solver_model.getQConstrs()
        else:
            gurobi_cons_to_load = set([con_map[pyomo_con] for pyomo_con in cons_to_load])
            linear_cons_to_load = list(gurobi_cons_to_load.intersection(set(self._solver_model.getConstrs())))
            quadratic_cons_to_load = list(gurobi_cons_to_load.intersection(set(self._solver_model.getQConstrs())))
        linear_vals = self._solver_model.getAttr("Slack", linear_cons_to_load)
        quadratic_vals = self._solver_model.getAttr("QCSlack", quadratic_cons_to_load)

        for gurobi_con, val in zip(linear_cons_to_load, linear_vals):
            pyomo_con = reverse_con_map[id(gurobi_con)]
            if pyomo_con in self._range_constraints:
                lin_expr = self._solver_model.getRow(gurobi_con)
                for i in reversed(range(lin_expr.size())):
                    v = lin_expr.getVar(i)
                    if v in gurobi_range_con_vars:
                        Us_ = v.X
                        Ls_ = v.UB - v.X
                        if Us_ > Ls_:
                            slack[pyomo_con] = Us_
                        else:
                            slack[pyomo_con] = -Ls_
                        break
            else:
                slack[pyomo_con] = val
        for gurobi_con, val in zip(quadratic_cons_to_load, quadratic_vals):
            pyomo_con = reverse_con_map[id(gurobi_con)]
            slack[pyomo_con] = val

    def _update(self):
        if self.config.update_vars:
            last_solve_vars = ComponentSet(self._solver_var_to_pyomo_var_map.values())
            current_vars = ComponentSet(v for v in self._pyomo_model.component_data_objects(Var, descend_into=True, sort=True))
        if self.config.check_for_new_constraints or self.config.check_for_removed_constraints:
            last_solve_cons = ComponentSet(self._solver_con_to_pyomo_con_map.values())
            current_cons = ComponentSet(c for c in self._pyomo_model.component_data_objects(Constraint, active=True, descend_into=True, sort=True))
        new_cons = current_cons - last_solve_cons
        old_cons = last_solve_cons - current_cons
        for c in old_cons:
            self._remove_constraint(c)
        new_vars = current_vars - last_solve_vars
        old_vars = last_solve_vars - current_vars
        for v in old_vars:
            self._remove_var(v)
        for v in new_vars:
            self._add_var(v)
        for c in new_cons:
            self._add_constraint(c)
        for helper in self._mutable_helpers:
            helper.update()
        for helper in self._mutable_quadratic_helpers:
            gurobi_con = helper.con
            new_gurobi_expr = helper.get_updated_expression()
            new_rhs = helper.get_updated_rhs()
            new_sense = gurobi_con.sense
            pyomo_con = self._solver_con_to_pyomo_con_map[id(gurobi_con)]
            name = self._symbol_map.getSymbol(pyomo_con, self._labeler)
            self._solver_model.remove(gurobi_con)
            new_con = self._solver_model.addQConstr(new_gurobi_expr, new_sense, new_rhs, name=name)
            self._pyomo_con_to_solver_con_map[id(pyomo_con)] = new_con
            del self._solver_con_to_pyomo_con_map[id(gurobi_con)]
            self._solver_con_to_pyomo_con_map[id(new_con)] = pyomo_con
            helper.con = new_con
        pyomo_obj = _get_objective(self._pyomo_model)
        if pyomo_obj is self._objective and pyomo_obj.expr is self._objective_expr:
            helper = self._mutable_objective
            new_gurobi_expr = helper.get_updated_expression()
            if new_gurobi_expr is not None:
                if pyomo_obj.sense == minimize:
                    sense = self._gurobipy.GRB.MINIMIZE
                else:
                    sense = self._gurobipy.GRB.MAXIMIZE
                self._solver_model.setObjective(new_gurobi_expr, sense=sense)
        else:
            self._set_objective(pyomo_obj)


GurobiAuto.solve.__doc__ = add_docstring_list(GurobiAuto.solve.__doc__, GurobiAuto.CONFIG)
