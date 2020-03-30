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
from pyomo.core.expr.numvalue import value, is_constant, native_types, is_fixed
from pyomo.repn import generate_standard_repn
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.objective import Objective
from pyomo.common.config import ConfigValue, add_docstring_list, NonNegativeFloat
from pyutilib.misc.config import ImmutableConfigValue
from pyomo.common.errors import PyomoException
from pyomo.solver.base import MIPSolver, ResultsBase, SolutionLoaderBase, TerminationCondition
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_components, identify_variables
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.base.expression import SimpleExpression, _GeneralExpressionData
import collections
from pyomo.common.ordered_set import OrderedDict, OrderedSet


logger = logging.getLogger('pyomo.solvers')


class ConfigurationError(PyomoException):
    pass


class DegreeError(PyomoException):
    pass


class GurobiPersistentSolutionLoader(SolutionLoaderBase):
    def __init__(self, model, solver):
        self._solver = solver
        self._model = model
        self._valid = True

    def _assert_solution_still_valid(self):
        if not self._valid:
            raise RuntimeError('The results in the solver are no longer valid.')

    def load_suffix(self, suffix):
        if suffix == 'dual':
            self.load_duals()
        elif suffix == 'slack':
            self.load_slacks()
        elif suffix == 'rc':
            self.load_reduced_costs()
        else:
            raise ValueError('suffix not recognized')

    def load_solution(self):
        self._assert_solution_still_valid()

        self._solver.load_vars()

        if hasattr(self._model, 'dual'):
            self._solver.load_duals()

        if hasattr(self._model, 'slack'):
            self._solver.load_slacks()

        if hasattr(self._model, 'rc'):
            self._solver.load_rc()

    def load_vars(self, vars_to_load=None, solution_number=0):
        self._assert_solution_still_valid()
        self._solver.load_vars(vars_to_load=vars_to_load, solution_number=solution_number)

    def load_reduced_costs(self, vars_to_load=None):
        self._assert_solution_still_valid()
        self._solver.load_rc(vars_to_load=vars_to_load)

    def load_duals(self, cons_to_load=None):
        self._assert_solution_still_valid()
        self._solver.load_duals(cons_to_load=cons_to_load)

    def load_slacks(self, cons_to_load=None):
        self._assert_solution_still_valid()
        self._solver.load_slacks(cons_to_load=cons_to_load)


class GurobiPersistentResults(ResultsBase):
    def __init__(self, model, solver):
        super(GurobiPersistentResults, self).__init__()
        self.solution_loader = GurobiPersistentSolutionLoader(model=model, solver=solver)
        if solver.get_model_attr('SolCount') > 0:
            self._found_feasible_solution = True
        else:
            self._found_feasible_solution = False
        self.solver.declare('wallclock_time',
                            ConfigValue(default=None,
                                        domain=NonNegativeFloat,
                                        doc="The wallclock time reported by Gurobi"))

    def found_feasible_solution(self):
        return self._found_feasible_solution


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
        self.slack_name = None
        self.gurobi_model = None

    def update(self):
        rhs_val = value(self.rhs_expr)
        lhs_val = value(self.lhs_expr)
        self.con.rhs = rhs_val
        slack = self.gurobi_model.getVarByName(self.slack_name)
        slack.ub = rhs_val - lhs_val


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
        self.last_constant_value = value(self.constant.expr)
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


class _GurobiWalker(StreamBasedExpressionVisitor):
    def __init__(self, var_map):
        """
        Parameters
        ----------
        var_map: dict
            maps ids of pyomo vars to gurobi vars
        """
        import gurobipy
        self._gurobipy = gurobipy
        super(_GurobiWalker, self).__init__()
        self.var_map = var_map
        self.referenced_vars = ComponentSet()

    def initializeWalker(self, expr):
        self.referenced_vars = ComponentSet()
        walk, result = self.beforeChild(None, expr)
        if not walk:
            return False, result
        return True, None

    # before child - skip leaf nodes
    def beforeChild(self, node, child):
        child_type = child.__class__
        if child_type in native_types:
            return False, value(child)
        if child_type is numeric_expr.LinearExpression:
            self.referenced_vars.update(child.linear_vars)
            return (False, (self._gurobipy.LinExpr(child.linear_coefs,
                                                   [self.var_map[id(i)] for i in child.linear_vars]) +
                            child.constant))
        if child.is_expression_type():
            return True, None
        if child.is_fixed():  # we use is_fixed rather than is_constant in order to keep the same behavior as generate_standard_repn
            return False, value(child)
        if child.is_variable_type():
            self.referenced_vars.add(child)
            return False, self.var_map[id(child)]
        return True, None

    def exitNode(self, node, data):
        if node.__class__ is numeric_expr.PowExpression:
            arg1, arg2 = data
            if arg2 != 2:
                raise ValueError('Cannot handle exponent {0}'.format(str(arg2)))
            return arg1 * arg1
        return node._apply_operation(data)


@SolverFactory.register('gurobi_persistent_new', doc='Direct python interface to Gurobi with automatic updates')
class GurobiPersistentNew(MIPSolver):
    """
    Direct interface to Gurobi
    """
    CONFIG = MIPSolver.CONFIG()

    CONFIG.declare('symbolic_solver_labels', ImmutableConfigValue(default=False, domain=bool,
                                                                  doc='If True, the gurobi variable and constraint names '
                                                                      'will match those of the pyomo variables and constrains. '
                                                                      'Cannot be changed after set_instance is called.'))
    CONFIG.declare('stream_solver', ConfigValue(default=False, domain=bool,
                                                doc='If True, show the Gurobi output'))
    CONFIG.declare('check_for_updated_mutable_params_in_constraints',
                   ImmutableConfigValue(default=True, domain=bool,
                                        doc='If True, the solver interface will look for constraint coefficients that depend on '
                                            'mutable parameters, and automatically update the coefficients for each solve. '
                                            'Cannot be changed after set_instance is called.'))
    CONFIG.declare('check_for_updated_mutable_params_in_objective',
                   ImmutableConfigValue(default=True, domain=bool,
                                        doc='If True, the solver interface will look for objective coefficients that depend on '
                                            'mutable parameters, and automatically update the coefficients for each solve. '
                                            'Cannot be changed after set_instance is called.'))
    CONFIG.declare('check_for_new_or_removed_constraints',
                   ConfigValue(default=True,
                               domain=bool,
                               doc='If True, the solver interface will check '
                                   'for new or removed constraints when '
                                   'solve is called.'))
    CONFIG.declare('update_constraints',
                   ConfigValue(default=False,
                               domain=bool,
                               doc='If True, the solver interface will update '
                                   'constraint bounds each time solve is called.'))
    CONFIG.declare('check_for_new_or_removed_vars',
                   ConfigValue(default=True,
                               domain=bool,
                               doc='If True, the solver interface will check '
                                   'for new or removed vars each time solve is called.'))
    CONFIG.declare('update_vars',
                   ConfigValue(default=True,
                               domain=bool,
                               doc='If True, the solver interface will update '
                                   'variable bounds each time solve is called.'))
    CONFIG.declare('update_named_expressions',
                   ImmutableConfigValue(default=True,
                                        domain=bool,
                                        doc='If True, the solver interface will update '
                                            'Expressions each time solve is called. '
                                            'Cannot be changed after set_instance is called.'))
    CONFIG.declare('check_for_fixed_vars',
                   ImmutableConfigValue(default=True,
                                        domain=bool,
                                        doc='If True, the solver interface will check for fixed '
                                            'variables in each constraint that is added. If '
                                            'The variable is fixed when the constraint gets '
                                            'added and is later unfixed, the constraints will be '
                                            'updated accordingly. If check_for_fixed_vars is False '
                                            'and a variable is fixed when adding a constraint, the '
                                            'constraint will not be updated correctly when the '
                                            'variable is unfixed.'))

    __doc__ = add_docstring_list(__doc__, CONFIG)

    def __init__(self):
        super(GurobiPersistentNew, self).__init__()

        self._pyomo_model = None
        self._solver_model = None
        self._symbol_map = SymbolMap()
        self._labeler = None
        self._pyomo_var_to_solver_var_map = OrderedDict()
        self._solver_var_to_pyomo_var_map = OrderedDict()
        self._pyomo_con_to_solver_con_map = OrderedDict()
        self._solver_con_to_pyomo_con_map = OrderedDict()
        self._pyomo_sos_to_solver_sos_map = OrderedDict()
        self._solver_sos_to_pyomo_sos_map = OrderedDict()
        self._vars_referenced_by_con = OrderedDict()
        self._vars_referenced_by_obj = OrderedDict()
        self._objective = None
        self._objective_expr = None
        self._referenced_variables = OrderedDict()
        self._referenced_params = OrderedDict()
        self._range_constraints = OrderedSet()
        self._tmp_config = None
        self._tmp_options = None
        self._mutable_helpers = OrderedDict()
        self._mutable_quadratic_helpers = OrderedDict()
        self._mutable_objective = None
        self._last_results_object = None
        self._walker = _GurobiWalker(self._pyomo_var_to_solver_var_map)
        self._constraint_bodies = OrderedDict()
        self._constraint_lowers = OrderedDict()
        self._constraint_uppers = OrderedDict()
        self._named_expressions = OrderedDict()
        self._obj_named_expressions = list()
        self._needs_updated = True
        self._callback = None
        self._callback_func = None
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._fixed_vars_to_dependent_cons_map = OrderedDict()  # pyomo var to set of constraints that were added when the var was fixed
        self._cons_with_fixed_vars = OrderedDict()  # pyomo constraint to ComponentSet of fixed vars in the constraint when it was added

        try:
            import gurobipy
            self._gurobipy = gurobipy
            self._python_api_exists = True
        except Exception as e:
            logger.warning("Import of gurobipy failed - gurobi message=" + str(e) + "\n")
            self._python_api_exists = False

        if self._gurobipy.GRB.VERSION_MAJOR < 7:
            # The reason for this is that it is too difficult to manage the gurobi lazy updates both for
            # versions >= 7 and < 7.
            logger.warning('The persistent interface to Gurobi requires at least Gurobi version 7. ')
            self._python_api_exists = False

    def available(self):
        return self._python_api_exists

    def license_status(self):
        try:
            tmp = self._gurobipy.Model()
            return True
        except self._gurobipy.GurobiError:
            return False

    def version(self):
        return (self._gurobipy.GRB.VERSION_MAJOR,
                self._gurobipy.GRB.VERSION_MINOR,
                self._gurobipy.GRB.VERSION_TECHNICAL)

    def is_persistent(self):
        return True

    def solve(self, model, options=None, **config_options):
        """
        solve a model
        """
        if not self.available():
            raise RuntimeError('The persistent interface to Gurobi is not available either because gurobipy could '
                               'not be imported or because the version of Gurobi being used is less than 7.')
        if self._last_results_object is not None:
            self._last_results_object.solution_loader._valid = False

        if options is None:
            options = dict()
        self._tmp_options = self.options(options, preserve_implicit=True)
        self._tmp_config = self.config(config_options)

        if model is self._pyomo_model:
            self.update()
        else:
            self.set_instance(model)

        self._apply_solver()
        return self._postsolve()

    def _apply_solver(self):
        if self._tmp_config.stream_solver:
            self._solver_model.setParam('OutputFlag', 1)
        else:
            self._solver_model.setParam('OutputFlag', 0)

        for key, option in self._tmp_options.items():
            self._solver_model.setParam(key, option)
        self._solver_model.optimize(self._callback)
        self._needs_updated = False

    def add_var(self, var):
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
        self._vars_added_since_update.add(var)

        self._needs_updated = True

    def set_instance(self, model, **config_options):
        self._pyomo_model = model
        self._symbol_map = SymbolMap()
        self._labeler = None
        self._pyomo_var_to_solver_var_map = OrderedDict()
        self._solver_var_to_pyomo_var_map = OrderedDict()
        self._pyomo_con_to_solver_con_map = OrderedDict()
        self._solver_con_to_pyomo_con_map = OrderedDict()
        self._pyomo_sos_to_solver_sos_map = OrderedDict()
        self._solver_sos_to_pyomo_sos_map = OrderedDict()
        self._vars_referenced_by_con = OrderedDict()
        self._vars_referenced_by_obj = OrderedDict()
        self._objective = None
        self._referenced_variables = OrderedDict()
        self._range_constraints = OrderedSet()
        self._mutable_helpers = OrderedDict()
        self._mutable_quadratic_helpers = OrderedDict()
        self._last_results_object = None
        self._walker = _GurobiWalker(self._pyomo_var_to_solver_var_map)
        self._constraint_bodies = OrderedDict()
        self._constraint_lowers = OrderedDict()
        self._constraint_uppers = OrderedDict()
        self._named_expressions = OrderedDict()
        self._obj_named_expressions = list()
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._fixed_vars_to_dependent_cons_map = OrderedDict()  # pyomo var to set of constraints that were added when the var was fixed
        self._cons_with_fixed_vars = OrderedDict()  # pyomo constraint to ComponentSet of fixed vars in the constraint when it was added

        self.config.set_value(config_options)
        msg = ' can only be changed before set_instance is called or through the set_instance method'
        self.config.get('symbolic_solver_labels')._mutable = False
        self.config.get('symbolic_solver_labels')._immutable_error_message = 'symbolic_solver_labels' + msg
        self.config.get('check_for_updated_mutable_params_in_constraints')._mutable = False
        self.config.get('check_for_updated_mutable_params_in_constraints')._immutable_error_message = 'check_for_updated_mutable_params_in_constraints' + msg
        self.config.get('check_for_updated_mutable_params_in_objective')._mutable = False
        self.config.get('check_for_updated_mutable_params_in_objective')._immutable_error_message = 'check_for_updated_mutable_params_in_objective' + msg
        self.config.get('update_named_expressions')._mutable = False
        self.config.get('update_named_expressions')._immutable_error_message = 'update_named_expressions' + msg

        if self.config.symbolic_solver_labels:
            self._labeler = TextLabeler()
        else:
            self._labeler = NumericLabeler('x')

        if model.name is not None:
            self._solver_model = self._gurobipy.Model(model.name)
        else:
            self._solver_model = self._gurobipy.Model()

        self.add_block(model)

    def add_block(self, block):
        for var in block.component_data_objects(Var, descend_into=True, sort=True):
            self.add_var(var)

        for con in block.component_data_objects(Constraint, descend_into=True, active=True, sort=True):
            self.add_constraint(con)

        for con in block.component_data_objects(SOSConstraint, descend_into=True, active=True, sort=True):
            self.add_sos_constraint(con)

        self.set_objective(_get_objective(block))

    def remove_block(self, block):
        for con in block.component_data_objects(ctype=Constraint, descend_into=True, active=True, sort=True):
            self.remove_constraint(con)

        for con in block.component_data_objects(ctype=SOSConstraint, descend_into=True, active=True, sort=True):
            self.remove_sos_constraint(con)

        for var in block.component_data_objects(ctype=Var, descend_into=True, sort=True):
            self.remove_var(var)

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

        for ndx, v in enumerate(repn.quadratic_vars):
            x, y = v
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

        return new_expr, referenced_vars, repn.constant, mutable_linear_coefficients, mutable_quadratic_coefficients

    def add_constraint(self, con):
        assert con.active

        conname = self._symbol_map.getSymbol(con, self._labeler)

        if self.config.check_for_updated_mutable_params_in_constraints:
            (gurobi_expr,
             referenced_vars,
             repn_constant,
             mutable_linear_coefficients,
             mutable_quadratic_coefficients) = self._get_expr_from_pyomo_expr(con.body)
        else:
            gurobi_expr = self._walker.walk_expression(con.body)
            referenced_vars = self._walker.referenced_vars
            repn_constant = 0

        if gurobi_expr.__class__ in {self._gurobipy.LinExpr, self._gurobipy.Var}:
            if con.equality:
                rhs_expr = con.lower - repn_constant
                rhs_val = value(rhs_expr)
                gurobipy_con = self._solver_model.addLConstr(gurobi_expr,
                                                             self._gurobipy.GRB.EQUAL,
                                                             rhs_val,
                                                             name=conname)
                if self.config.check_for_updated_mutable_params_in_constraints:
                    if not is_constant(rhs_expr):
                        mutable_constant = _MutableConstant()
                        mutable_constant.expr = rhs_expr
                        mutable_constant.con = gurobipy_con
                        self._mutable_helpers[con] = [mutable_constant]
            elif con.has_lb() and con.has_ub():
                lhs_expr = con.lower - repn_constant
                rhs_expr = con.upper - repn_constant
                lhs_val = value(lhs_expr)
                rhs_val = value(rhs_expr)
                gurobipy_con = self._solver_model.addRange(gurobi_expr, lhs_val, rhs_val, name=conname)
                self._range_constraints.add(con)
                if self.config.check_for_updated_mutable_params_in_constraints:
                    if not is_constant(lhs_expr) or not is_constant(rhs_expr):
                        mutable_range_constant = _MutableRangeConstant()
                        mutable_range_constant.lhs_expr = lhs_expr
                        mutable_range_constant.rhs_expr = rhs_expr
                        mutable_range_constant.con = gurobipy_con
                        mutable_range_constant.slack_name = 'Rg' + conname
                        mutable_range_constant.gurobi_model = self._solver_model
                        self._mutable_helpers[con] = [mutable_range_constant]
            elif con.has_lb():
                rhs_expr = con.lower - repn_constant
                rhs_val = value(rhs_expr)
                gurobipy_con = self._solver_model.addLConstr(gurobi_expr, self._gurobipy.GRB.GREATER_EQUAL, rhs_val, name=conname)
                if self.config.check_for_updated_mutable_params_in_constraints:
                    if not is_constant(rhs_expr):
                        mutable_constant = _MutableConstant()
                        mutable_constant.expr = rhs_expr
                        mutable_constant.con = gurobipy_con
                        self._mutable_helpers[con] = [mutable_constant]
            elif con.has_ub():
                rhs_expr = con.upper - repn_constant
                rhs_val = value(rhs_expr)
                gurobipy_con = self._solver_model.addLConstr(gurobi_expr, self._gurobipy.GRB.LESS_EQUAL, rhs_val, name=conname)
                if self.config.check_for_updated_mutable_params_in_constraints:
                    if not is_constant(rhs_expr):
                        mutable_constant = _MutableConstant()
                        mutable_constant.expr = rhs_expr
                        mutable_constant.con = gurobipy_con
                        self._mutable_helpers[con] = [mutable_constant]
            else:
                raise ValueError("Constraint does not have a lower "
                                 "or an upper bound: {0} \n".format(con))
            if self.config.check_for_updated_mutable_params_in_constraints:
                for tmp in mutable_linear_coefficients:
                    tmp.con = gurobipy_con
                    tmp.gurobi_model = self._solver_model
                if len(mutable_linear_coefficients) > 0:
                    if con not in self._mutable_helpers:
                        self._mutable_helpers[con] = mutable_linear_coefficients
                    else:
                        self._mutable_helpers[con].extend(mutable_linear_coefficients)
        elif gurobi_expr.__class__ is self._gurobipy.QuadExpr:
            if con.equality:
                rhs_expr = con.lower - repn_constant
                rhs_val = value(rhs_expr)
                gurobipy_con = self._solver_model.addQConstr(gurobi_expr, self._gurobipy.GRB.EQUAL, rhs_val, name=conname)
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
            if self.config.check_for_updated_mutable_params_in_constraints:
                if len(mutable_linear_coefficients) > 0 or len(mutable_quadratic_coefficients) > 0 or not is_constant(repn_constant):
                    mutable_constant = _MutableConstant()
                    mutable_constant.expr = rhs_expr
                    mutable_quadratic_constraint = _MutableQuadraticConstraint(self._solver_model, gurobipy_con,
                                                                               mutable_constant,
                                                                               mutable_linear_coefficients,
                                                                               mutable_quadratic_coefficients)
                    self._mutable_quadratic_helpers[con] = mutable_quadratic_constraint
        else:
            raise ValueError('Unrecognized Gurobi expression type')
            
        if self.config.update_named_expressions:
            self._named_expressions[con] = list()
            for e in identify_components(con.body, {SimpleExpression, _GeneralExpressionData}):
                self._named_expressions[con].append((e, e.expr))
        for var in referenced_vars:
            self._referenced_variables[id(var)] += 1
        if self.config.check_for_fixed_vars:
            for v in identify_variables(con.body, include_fixed=True):
                if v.is_fixed():
                    v_id = id(v)
                    if v_id in self._fixed_vars_to_dependent_cons_map:
                        self._fixed_vars_to_dependent_cons_map[v_id].add(con)
                    else:
                        self._fixed_vars_to_dependent_cons_map[v_id] = OrderedSet([con])
                    if con not in self._cons_with_fixed_vars:
                        self._cons_with_fixed_vars[con] = ComponentSet()
                    self._cons_with_fixed_vars[con].add(v)
        self._vars_referenced_by_con[con] = referenced_vars
        self._pyomo_con_to_solver_con_map[con] = gurobipy_con
        self._solver_con_to_pyomo_con_map[id(gurobipy_con)] = con
        self._constraint_bodies[con] = con.body
        self._constraint_lowers[con] = value(con.lower)
        self._constraint_uppers[con] = value(con.upper)
        self._constraints_added_since_update.add(con)

        self._needs_updated = True

    def add_sos_constraint(self, con):
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

        self._vars_referenced_by_con[con] = ComponentSet()

        if hasattr(con, 'get_items'):
            # aml sos constraint
            sos_items = list(con.get_items())
        else:
            # kernel sos constraint
            sos_items = list(con.items())

        for v, w in sos_items:
            v_id = id(v)
            self._vars_referenced_by_con[con].add(v)
            gurobi_vars.append(self._pyomo_var_to_solver_var_map[v_id])
            self._referenced_variables[v_id] += 1
            weights.append(w)
            if self.config.check_for_fixed_vars:
                if v.is_fixed():
                    if v_id in self._fixed_vars_to_dependent_cons_map:
                        self._fixed_vars_to_dependent_cons_map[v_id].add(con)
                    else:
                        self._fixed_vars_to_dependent_cons_map[v_id] = OrderedSet([con])
                    if con not in self._cons_with_fixed_vars:
                        self._cons_with_fixed_vars[con] = ComponentSet()
                    self._cons_with_fixed_vars[con].add(v)

        gurobipy_con = self._solver_model.addSOS(sos_type, gurobi_vars, weights)
        self._pyomo_sos_to_solver_sos_map[con] = gurobipy_con
        self._solver_sos_to_pyomo_sos_map[id(gurobipy_con)] = con
        self._constraints_added_since_update.add(con)

        self._needs_updated = True

    def remove_constraint(self, con):
        if con in self._constraints_added_since_update:
            self._update_gurobi_model()
        con_id = id(con)
        solver_con = self._pyomo_con_to_solver_con_map[con]
        self._solver_model.remove(solver_con)
        self._symbol_map.removeSymbol(con)
        self._labeler.remove_obj(con)
        for var in self._vars_referenced_by_con[con]:
            self._referenced_variables[id(var)] -= 1
        del self._vars_referenced_by_con[con]
        del self._pyomo_con_to_solver_con_map[con]
        del self._solver_con_to_pyomo_con_map[id(solver_con)]
        self._range_constraints.discard(con)
        del self._constraint_bodies[con]
        del self._constraint_lowers[con]
        del self._constraint_uppers[con]
        self._mutable_helpers.pop(con, None)
        self._mutable_quadratic_helpers.pop(con, None)
        self._named_expressions.pop(con, None)
        if con in self._cons_with_fixed_vars:
            fixed_vars_in_con = self._cons_with_fixed_vars[con]
            for v in fixed_vars_in_con:
                v_id = id(v)
                self._fixed_vars_to_dependent_cons_map[v_id].remove(con)
                if len(self._fixed_vars_to_dependent_cons_map[v_id]) == 0:
                    del self._fixed_vars_to_dependent_cons_map[v_id]
            del self._cons_with_fixed_vars[con]
        self._needs_updated = True

    def remove_sos_constraint(self, con):
        if con in self._constraints_added_since_update:
            self._update_gurobi_model()
        solver_sos_con = self._pyomo_sos_to_solver_sos_map[con]
        self._solver_model.remove(solver_sos_con)
        self._symbol_map.removeSymbol(con)
        self._labeler.remove_obj(con)
        for var in self._vars_referenced_by_con[con]:
            self._referenced_variables[id(var)] -= 1
        del self._vars_referenced_by_con[con]
        del self._pyomo_sos_to_solver_sos_map[con]
        del self._solver_sos_to_pyomo_sos_map[id(solver_sos_con)]
        if con in self._cons_with_fixed_vars:
            fixed_vars_in_con = self._cons_with_fixed_vars[con]
            for v in fixed_vars_in_con:
                v_id = id(v)
                self._fixed_vars_to_dependent_cons_map[v_id].remove(con)
                if len(self._fixed_vars_to_dependent_cons_map[v_id]) == 0:
                    del self._fixed_vars_to_dependent_cons_map[v_id]
            del self._cons_with_fixed_vars[con]
        self._needs_updated = True

    def remove_var(self, var):
        if self._referenced_variables[id(var)] != 0:
            raise ValueError('Cannot remove Var {0} because it is still referenced by the objective or one or more constraints'.format(var))
        if var in self._vars_added_since_update:
            self._update_gurobi_model()
        solver_var = self._pyomo_var_to_solver_var_map[id(var)]
        self._solver_model.remove(solver_var)
        self._symbol_map.removeSymbol(var)
        self._labeler.remove_obj(var)
        del self._referenced_variables[id(var)]
        del self._pyomo_var_to_solver_var_map[id(var)]
        del self._solver_var_to_pyomo_var_map[id(solver_var)]

    def update_var(self, var):
        var_id = id(var)
        if var_id not in self._pyomo_var_to_solver_var_map:
            raise ValueError('The Var provided to update_var needs to be added first: {0}'.format(var))
        gurobipy_var = self._pyomo_var_to_solver_var_map[var_id]
        vtype = self._gurobi_vtype_from_var(var)
        if var.is_fixed():
            lb = var.value
            ub = var.value
        else:
            lb = -self._gurobipy.GRB.INFINITY
            ub = self._gurobipy.GRB.INFINITY
            if var.has_lb():
                lb = value(var.lb)
            if var.has_ub():
                ub = value(var.ub)
            if var_id in self._fixed_vars_to_dependent_cons_map:
                for con in self._fixed_vars_to_dependent_cons_map[var_id]:
                    self._cons_with_fixed_vars[con].remove(var)
                    if len(self._cons_with_fixed_vars[con]) == 0:
                        del self._cons_with_fixed_vars[con]
                    self.remove_constraint(con)
                    self.add_constraint(con)
                del self._fixed_vars_to_dependent_cons_map[var_id]
        gurobipy_var.setAttr('lb', lb)
        gurobipy_var.setAttr('ub', ub)
        gurobipy_var.setAttr('vtype', vtype)
        self._needs_updated = True

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

    def set_objective(self, obj):
        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[id(var)] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None
            self._objective_expr = None
            self._obj_named_expressions = list()

        if obj.active is False:
            raise ValueError('Cannot add inactive objective to solver.')

        if obj.sense == minimize:
            sense = self._gurobipy.GRB.MINIMIZE
        elif obj.sense == maximize:
            sense = self._gurobipy.GRB.MAXIMIZE
        else:
            raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

        if self.config.check_for_updated_mutable_params_in_objective:
            (gurobi_expr,
             referenced_vars,
             repn_constant,
             mutable_linear_coefficients,
             mutable_quadratic_coefficients) = self._get_expr_from_pyomo_expr(obj.expr)
        else:
            gurobi_expr = self._walker.walk_expression(obj.expr)
            referenced_vars = self._walker.referenced_vars
            repn_constant = 0
            mutable_linear_coefficients = list()
            mutable_quadratic_coefficients = list()

        if self.config.update_named_expressions:
            assert len(self._obj_named_expressions) == 0
            for e in identify_components(obj.expr, {SimpleExpression, _GeneralExpressionData}):
                self._obj_named_expressions.append((e, e.expr))

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

        self._needs_updated = True

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

        results = GurobiPersistentResults(self._pyomo_model, self)
        self._last_results_object = results
        results.solver.wallclock_time = gprob.Runtime

        if status == grb.LOADED:  # problem is loaded, but no solution
            results.solver.termination_condition = TerminationCondition.unknown
        elif status == grb.OPTIMAL:  # optimal
            results.solver.termination_condition = TerminationCondition.optimal
        elif status == grb.INFEASIBLE:
            results.solver.termination_condition = TerminationCondition.infeasible
        elif status == grb.INF_OR_UNBD:
            results.solver.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif status == grb.UNBOUNDED:
            results.solver.termination_condition = TerminationCondition.unbounded
        elif status == grb.CUTOFF:
            results.solver.termination_condition = TerminationCondition.objectiveLimit
        elif status == grb.ITERATION_LIMIT:
            results.solver.termination_condition = TerminationCondition.maxIterations
        elif status == grb.NODE_LIMIT:
            results.solver.termination_condition = TerminationCondition.maxIterations
        elif status == grb.TIME_LIMIT:
            results.solver.termination_condition = TerminationCondition.maxTimeLimit
        elif status == grb.SOLUTION_LIMIT:
            results.solver.termination_condition = TerminationCondition.unknown
        elif status == grb.INTERRUPTED:
            results.solver.termination_condition = TerminationCondition.interrupted
        elif status == grb.NUMERIC:
            results.solver.termination_condition = TerminationCondition.unknown
        elif status == grb.SUBOPTIMAL:
            results.solver.termination_condition = TerminationCondition.unknown
        elif status == grb.USER_OBJ_LIMIT:
            results.solver.termination_condition = TerminationCondition.objectiveLimit
        else:
            results.solver.termination_condition = TerminationCondition.unknown

        try:
            results.solver.best_feasible_objective = gprob.ObjVal
        except (self._gurobipy.GurobiError, AttributeError):
            pass
        try:
            results.solver.best_objective_bound = gprob.ObjBound
        except (self._gurobipy.GurobiError, AttributeError):
            pass

        if self._tmp_config.load_solution:
            if gprob.SolCount > 0:
                self.load_vars()

                if extract_reduced_costs:
                    self.load_rc()

                if extract_duals:
                    self.load_duals()

                if extract_slacks:
                    self.load_slacks()

        return results

    def _load_suboptimal_mip_solution(self, vars_to_load, solution_number):
        if self.get_model_attr('NumIntVars') == 0 and self.get_model_attr('NumBinVars') == 0:
            raise ValueError('Cannot obtain suboptimal solutions for a continuous model')
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        original_solution_number = self._solver.get_gurobi_param('SolutionNumber')
        self._solver.set_gurobi_param('SolutionNumber', solution_number)
        gurobi_vars_to_load = [var_map[id(pyomo_var)] for pyomo_var in vars_to_load]
        vals = self._solver_model.getAttr("Xn", gurobi_vars_to_load)
        for var, val in zip(vars_to_load, vals):
            if ref_vars[id(var)] > 0:
                var.value = val
        self._solver.set_gurobi_param('SolutionNumber', original_solution_number)

    def load_vars(self, vars_to_load=None, solution_number=0):
        if self._needs_updated:
            self._update_gurobi_model()  # this is needed to ensure that solutions cannot be loaded after the model has been changed
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = self._solver_var_to_pyomo_var_map.values()

        if solution_number != 0:
            self._load_suboptimal_mip_solution(vars_to_load=vars_to_load, solution_number=solution_number)
        else:
            gurobi_vars_to_load = [var_map[id(pyomo_var)] for pyomo_var in vars_to_load]
            vals = self._solver_model.getAttr("X", gurobi_vars_to_load)

            for var, val in zip(vars_to_load, vals):
                if ref_vars[id(var)] > 0:
                    var.value = val

    def load_rc(self, vars_to_load=None):
        if self._needs_updated:
            self._update_gurobi_model()
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
        if self._needs_updated:
            self._update_gurobi_model()
        if not hasattr(self._pyomo_model, 'dual'):
            self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        dual = self._pyomo_model.dual

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.getConstrs()
            quadratic_cons_to_load = self._solver_model.getQConstrs()
        else:
            gurobi_cons_to_load = OrderedSet([con_map[pyomo_con] for pyomo_con in cons_to_load])
            linear_cons_to_load = list(gurobi_cons_to_load.intersection(OrderedSet(self._solver_model.getConstrs())))
            quadratic_cons_to_load = list(gurobi_cons_to_load.intersection(OrderedSet(self._solver_model.getQConstrs())))
        linear_vals = self._solver_model.getAttr("Pi", linear_cons_to_load)
        quadratic_vals = self._solver_model.getAttr("QCPi", quadratic_cons_to_load)

        for gurobi_con, val in zip(linear_cons_to_load, linear_vals):
            pyomo_con = reverse_con_map[id(gurobi_con)]
            dual[pyomo_con] = val
        for gurobi_con, val in zip(quadratic_cons_to_load, quadratic_vals):
            pyomo_con = reverse_con_map[id(gurobi_con)]
            dual[pyomo_con] = val

    def load_slacks(self, cons_to_load=None):
        if self._needs_updated:
            self._update_gurobi_model()
        if not hasattr(self._pyomo_model, 'slack'):
            self._pyomo_model.slack = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        slack = self._pyomo_model.slack

        gurobi_range_con_vars = OrderedSet(self._solver_model.getVars()) - OrderedSet(self._pyomo_var_to_solver_var_map.values())

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.getConstrs()
            quadratic_cons_to_load = self._solver_model.getQConstrs()
        else:
            gurobi_cons_to_load = OrderedSet([con_map[pyomo_con] for pyomo_con in cons_to_load])
            linear_cons_to_load = list(gurobi_cons_to_load.intersection(OrderedSet(self._solver_model.getConstrs())))
            quadratic_cons_to_load = list(gurobi_cons_to_load.intersection(OrderedSet(self._solver_model.getQConstrs())))
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

    def update(self):
        if self._needs_updated:
            self._solver_model.update()
        if self.config.check_for_new_or_removed_vars or self.config.update_vars:
            last_solve_vars = ComponentSet(self._solver_var_to_pyomo_var_map.values())
            current_vars = ComponentSet(v for v in self._pyomo_model.component_data_objects(Var, descend_into=True, sort=True))
            new_vars = current_vars - last_solve_vars
            old_vars = last_solve_vars - current_vars
        if self.config.check_for_new_or_removed_constraints or self.config.update_constraints:
            last_solve_cons = ComponentSet(self._solver_con_to_pyomo_con_map.values())
            current_cons = ComponentSet(c for c in self._pyomo_model.component_data_objects(Constraint, active=True, descend_into=True, sort=True))
            new_cons = current_cons - last_solve_cons
            old_cons = last_solve_cons - current_cons
        if self.config.check_for_new_or_removed_constraints:
            for c in old_cons:
                self.remove_constraint(c)
            for c in self._pyomo_sos_to_solver_sos_map.keys():
                self.remove_sos_constraint(c)
        if self.config.check_for_new_or_removed_vars:
            for v in old_vars:
                self.remove_var(v)
            for v in new_vars:
                self.add_var(v)
        if self.config.check_for_new_or_removed_constraints:
            for c in new_cons:
                self.add_constraint(c)
            for c in self._pyomo_model.component_data_objects(SOSConstraint, descend_into=True, active=True, sort=True):
                self.add_sos_constraint(c)
        if self.config.update_constraints:
            cons_to_update = current_cons - new_cons
            for c in cons_to_update:
                if ((c.body is not self._constraint_bodies[id(c)]) or
                        (value(c.lower) != self._constraint_lowers[id(c)]) or
                        (value(c.upper) != self._constraint_uppers[id(c)])):
                    self.remove_constraint(c)
                    self.add_constraint(c)
        if self.config.update_vars:
            vars_to_update = current_vars - new_vars
            for v in vars_to_update:
                self.update_var(v)
        if self.config.update_named_expressions:
            for c, expr_list in self._named_expressions.items():
                for named_expr, old_expr in expr_list:
                    if not (named_expr.expr is old_expr):
                        self.remove_constraint(c)
                        self.add_constraint(c)
                        break
        if self.config.check_for_updated_mutable_params_in_constraints:
            for con, helpers in self._mutable_helpers.items():
                for helper in helpers:
                    helper.update()
            for con, helper in self._mutable_quadratic_helpers.items():
                if con in self._constraints_added_since_update:
                    self._update_gurobi_model()
                gurobi_con = helper.con
                new_gurobi_expr = helper.get_updated_expression()
                new_rhs = helper.get_updated_rhs()
                new_sense = gurobi_con.qcsense
                pyomo_con = self._solver_con_to_pyomo_con_map[id(gurobi_con)]
                name = self._symbol_map.getSymbol(pyomo_con, self._labeler)
                self._solver_model.remove(gurobi_con)
                new_con = self._solver_model.addQConstr(new_gurobi_expr, new_sense, new_rhs, name=name)
                self._pyomo_con_to_solver_con_map[id(pyomo_con)] = new_con
                del self._solver_con_to_pyomo_con_map[id(gurobi_con)]
                self._solver_con_to_pyomo_con_map[id(new_con)] = pyomo_con
                helper.con = new_con
                self._constraints_added_since_update.add(con)
        pyomo_obj = _get_objective(self._pyomo_model)
        already_called_set_objective = False
        if not (pyomo_obj is self._objective):
            self.set_objective(pyomo_obj)
            already_called_set_objective = True
        if (not already_called_set_objective) and (not (pyomo_obj.expr is self._objective_expr)):
            self._set_objective(pyomo_obj)
            already_called_set_objective = True
        if (not already_called_set_objective) and self.config.update_named_expressions:
            for named_expr, old_expr in self._obj_named_expressions:
                if not (named_expr.expr is old_expr):
                    self._set_objective(pyomo_obj)
                    already_called_set_objective = True
                    break
        if (not already_called_set_objective) and self.config.check_for_updated_mutable_params_in_objective:
            helper = self._mutable_objective
            new_gurobi_expr = helper.get_updated_expression()
            if new_gurobi_expr is not None:
                if pyomo_obj.sense == minimize:
                    sense = self._gurobipy.GRB.MINIMIZE
                else:
                    sense = self._gurobipy.GRB.MAXIMIZE
                self._solver_model.setObjective(new_gurobi_expr, sense=sense)
        self._needs_updated = True

    def _update_gurobi_model(self):
        self._solver_model.update()
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._needs_updated = False

    def get_model_attr(self, attr):
        """
        Get the value of an attribute on the Gurobi model.

        Parameters
        ----------
        attr: str
            The attribute to get. See Gurobi documentation for descriptions of the attributes.
        """
        if self._needs_updated:
            self._update_gurobi_model()
        return self._solver_model.getAttr(attr)

    def write(self, filename):
        """
        Write the model to a file (e.g., and lp file).

        Parameters
        ----------
        filename: str
            Name of the file to which the model should be written.
        """
        self._solver_model.write(filename)
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._needs_updated = False

    def set_linear_constraint_attr(self, con, attr, val):
        """
        Set the value of an attribute on a gurobi linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be modified.
        attr: str
            The attribute to be modified. Options are:
                CBasis
                DStart
                Lazy
        val: any
            See gurobi documentation for acceptable values.
        """
        if attr in {'Sense', 'RHS', 'ConstrName'}:
            raise ValueError('Linear constraint attr {0} cannot be set with' +
                             ' the set_linear_constraint_attr method. Please use' +
                             ' the remove_constraint and add_constraint methods.'.format(attr))
        self._pyomo_con_to_solver_con_map[con].setAttr(attr, val)
        self._needs_updated = True

    def set_var_attr(self, var, attr, val):
        """
        Set the value of an attribute on a gurobi variable.

        Parameters
        ----------
        var: pyomo.core.base.var._GeneralVarData
            The pyomo var for which the corresponding gurobi var attribute
            should be modified.
        attr: str
            The attribute to be modified. Options are:
                Start
                VarHintVal
                VarHintPri
                BranchPriority
                VBasis
                PStart
        val: any
            See gurobi documentation for acceptable values.
        """
        if attr in {'LB', 'UB', 'VType', 'VarName'}:
            raise ValueError('Var attr {0} cannot be set with' +
                             ' the set_var_attr method. Please use' +
                             ' the update_var method.'.format(attr))
        if attr == 'Obj':
            raise ValueError('Var attr Obj cannot be set with' +
                             ' the set_var_attr method. Please use' +
                             ' the set_objective method.')
        self._pyomo_var_to_solver_var_map[id(var)].setAttr(attr, val)
        self._needs_updated = True

    def get_var_attr(self, var, attr):
        """
        Get the value of an attribute on a gurobi var.

        Parameters
        ----------
        var: pyomo.core.base.var._GeneralVarData
            The pyomo var for which the corresponding gurobi var attribute
            should be retrieved.
        attr: str
            The attribute to get. See gurobi documentation
        """
        if self._needs_updated:
            self._update()
        return self._pyomo_var_to_solver_var_map[id(var)].getAttr(attr)

    def get_linear_constraint_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. See the Gurobi documentation
        """
        if self._needs_updated:
            self._update()
        return self._pyomo_con_to_solver_con_map[con].getAttr(attr)

    def get_sos_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi sos constraint.

        Parameters
        ----------
        con: pyomo.core.base.sos._SOSConstraintData
            The pyomo SOS constraint for which the corresponding gurobi SOS constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. See the Gurobi documentation
        """
        if self._needs_updated:
            self._update()
        return self._pyomo_sos_to_solver_sos_map[con].getAttr(attr)

    def get_quadratic_constraint_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi quadratic constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. See the Gurobi documentation
        """
        if self._needs_updated:
            self._update()
        return self._pyomo_con_to_solver_con_map[con].getAttr(attr)

    def set_gurobi_param(self, param, val):
        """
        Set a gurobi parameter.

        Parameters
        ----------
        param: str
            The gurobi parameter to set. Options include any gurobi parameter.
            Please see the Gurobi documentation for options.
        val: any
            The value to set the parameter to. See Gurobi documentation for possible values.
        """
        self._solver_model.setParam(param, val)

    def get_gurobi_param_info(self, param):
        """
        Get information about a gurobi parameter.

        Parameters
        ----------
        param: str
            The gurobi parameter to get info for. See Gurobi documenation for possible options.

        Returns
        -------
        six-tuple containing the parameter name, type, value, minimum value, maximum value, and default value.
        """
        return self._solver_model.getParamInfo(param)

    def _intermediate_callback(self):
        def f(gurobi_model, where):
            self._callback_func(self._pyomo_model, self, where)
        return f

    def set_callback(self, func=None):
        """
        Specify a callback for gurobi to use.

        Parameters
        ----------
        func: function
            The function to call. The function should have three arguments. The first will be the pyomo model being
            solved. The second will be the GurobiPersistent instance. The third will be an enum member of
            gurobipy.GRB.Callback. This will indicate where in the branch and bound algorithm gurobi is at. For
            example, suppose we want to solve

            min 2*x + y
            s.t.
                y >= (x-2)**2
                0 <= x <= 4
                y >= 0
                y integer

            as an MILP using exteneded cutting planes in callbacks.

            >>>
            >> from gurobipy import GRB
            >> import pyomo.environ as pe
            >> from pyomo.core.expr.taylor_series import taylor_series_expansion
            >>
            >> m = pe.ConcreteModel()
            >> m.x = pe.Var(bounds=(0, 4))
            >> m.y = pe.Var(within=pe.Integers, bounds=(0, None))
            >> m.obj = pe.Objective(expr=2*m.x + m.y)
            >> m.cons = pe.ConstraintList()  # for the cutting planes
            >>
            >> def _add_cut(xval):
            >>     # a function to generate the cut
            >>     m.x.value = xval
            >>     return m.cons.add(m.y >= taylor_series_expansion((m.x - 2)**2))
            >>
            >> _add_cut(0)  # start with 2 cuts at the bounds of x
            >> _add_cut(4)  # this is an arbitrary choice
            >>
            >> opt = pe.SolverFactory('gurobi_persistent')
            >> opt.set_instance(m)
            >> opt.set_gurobi_param('PreCrush', 1)
            >> opt.set_gurobi_param('LazyConstraints', 1)
            >>
            >> def my_callback(cb_m, cb_opt, cb_where):
            >>     if cb_where == GRB.Callback.MIPSOL:
            >>         cb_opt.cbGetSolution(vars=[m.x, m.y])
            >>         if m.y.value < (m.x.value - 2)**2 - 1e-6:
            >>             cb_opt.cbLazy(_add_cut(m.x.value))
            >>
            >> opt.set_callback(my_callback)
            >> opt.solve()
            >> assert abs(m.x.value - 1) <= 1e-6
            >> assert abs(m.y.value - 1) <= 1e-6

        """
        if func is not None:
            self._callback_func = func
            self._callback = self._intermediate_callback()
        else:
            self._callback = None
            self._callback_func = None

    def cbCut(self, con):
        """
        Add a cut within a callback.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The cut to add
        """
        if not con.active:
            raise ValueError('cbCut expected an active constraint.')

        if is_fixed(con.body):
            raise ValueError('cbCut expected a non-trival constraint')

        gurobi_expr = self._walker.walk_expression(con.body)

        if con.has_lb():
            if con.has_ub():
                raise ValueError('Range constraints are not supported in cbCut.')
            if not is_fixed(con.lower):
                raise ValueError('Lower bound of constraint {0} is not constant.'.format(con))
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError('Upper bound of constraint {0} is not constant.'.format(con))

        if con.equality:
            self._solver_model.cbCut(lhs=gurobi_expr, sense=self._gurobipy.GRB.EQUAL,
                                     rhs=value(con.lower))
        elif con.has_lb() and (value(con.lower) > -float('inf')):
            self._solver_model.cbCut(lhs=gurobi_expr, sense=self._gurobipy.GRB.GREATER_EQUAL,
                                     rhs=value(con.lower))
        elif con.has_ub() and (value(con.upper) < float('inf')):
            self._solver_model.cbCut(lhs=gurobi_expr, sense=self._gurobipy.GRB.LESS_EQUAL,
                                     rhs=value(con.upper))
        else:
            raise ValueError('Constraint does not have a lower or an upper bound {0} \n'.format(con))

    def cbGet(self, what):
        return self._solver_model.cbGet(what)

    def cbGetNodeRel(self, vars):
        """
        Parameters
        ----------
        vars: Var or iterable of Var
        """
        if not isinstance(vars, collections.Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[id(i)] for i in vars]
        var_values = self._solver_model.cbGetNodeRel(gurobi_vars)
        for i, v in enumerate(vars):
            v.value = var_values[i]

    def cbGetSolution(self, vars):
        """
        Parameters
        ----------
        vars: iterable of vars
        """
        if not isinstance(vars, collections.Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[id(i)] for i in vars]
        var_values = self._solver_model.cbGetSolution(gurobi_vars)
        for i, v in enumerate(vars):
            v.value = var_values[i]

    def cbLazy(self, con):
        """
        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The lazy constraint to add
        """
        if not con.active:
            raise ValueError('cbLazy expected an active constraint.')

        if is_fixed(con.body):
            raise ValueError('cbLazy expected a non-trival constraint')

        gurobi_expr = self._walker.walk_expression(con.body)

        if con.has_lb():
            if con.has_ub():
                raise ValueError('Range constraints are not supported in cbLazy.')
            if not is_fixed(con.lower):
                raise ValueError('Lower bound of constraint {0} is not constant.'.format(con))
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError('Upper bound of constraint {0} is not constant.'.format(con))

        if con.equality:
            self._solver_model.cbLazy(lhs=gurobi_expr, sense=self._gurobipy.GRB.EQUAL,
                                      rhs=value(con.lower))
        elif con.has_lb() and (value(con.lower) > -float('inf')):
            self._solver_model.cbLazy(lhs=gurobi_expr, sense=self._gurobipy.GRB.GREATER_EQUAL,
                                      rhs=value(con.lower))
        elif con.has_ub() and (value(con.upper) < float('inf')):
            self._solver_model.cbLazy(lhs=gurobi_expr, sense=self._gurobipy.GRB.LESS_EQUAL,
                                      rhs=value(con.upper))
        else:
            raise ValueError('Constraint does not have a lower or an upper bound {0} \n'.format(con))

    def cbSetSolution(self, vars, solution):
        if not isinstance(vars, collections.Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[id(i)] for i in vars]
        self._solver_model.cbSetSolution(gurobi_vars, solution)

    def cbUseSolution(self):
        return self._solver_model.cbUseSolution()

    def reset(self):
        self._solver_model.reset()


GurobiPersistentNew.solve.__doc__ = add_docstring_list(GurobiPersistentNew.solve.__doc__, GurobiPersistentNew.CONFIG)
