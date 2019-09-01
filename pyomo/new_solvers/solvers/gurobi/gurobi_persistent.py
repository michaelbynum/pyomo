#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import collections
import pyomo.core.expr
import pyomo.core.base
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.objective import minimize, maximize
import logging
from pyomo.new_solvers.results.results import Results
from pyomo.opt.results.solver import TerminationCondition
from pyomo.repn.standard_repn import generate_standard_repn
import time


logger = logging.getLogger(__name__)


class DegreeError(ValueError):
    pass


def _is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


class GurobiNew(object):
    def __init__(self):
        import gurobipy
        self._gurobipy = gurobipy
        self._pyomo_model = None
        self._solver_model = None
        self._symbol_map = pyomo.core.base.SymbolMap()
        self._labeler = None
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = dict()
        self._vars_referenced_by_con = ComponentMap()
        self._vars_referenced_by_obj = ComponentSet()
        self._objective = None
        self._symbolic_solver_labels = False
        self._referenced_variables = ComponentMap()
        self._keepfiles = False
        self._name = None
        self._range_constraints = set()
        self._max_obj_degree = 2
        self._max_constraint_degree = 2
        self._callback = None
        self._callback_func = None
        self.options = dict()

    def set_instance(self, model, symbolic_solver_labels=False):
        """
        This method is used to translate the Pyomo model provided to an instance of the solver's Python model. This
        discards any existing model and starts from scratch.

        Parameters
        ----------
        model: ConcreteModel
            The pyomo model to be used with the solver.
        symbolic_solver_labels: bool
            If True, the solver's components (e.g., variables, constraints) will be given names that correspond to
            the Pyomo component names.
        """
        if not isinstance(model, (pyomo.core.base.PyomoModel.Model, IBlock,
                                  pyomo.core.base.block.Block, pyomo.core.base.block._BlockData)):
            msg = "The problem instance supplied to the {0} plugin " \
                  "'_presolve' method must be a Model or a Block".format(type(self))
            raise ValueError(msg)
        self._range_constraints = set()
        self._pyomo_model = model
        self._symbolic_solver_labels = symbolic_solver_labels
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = ComponentMap()
        self._solver_con_to_pyomo_con_map = ComponentMap()
        self._vars_referenced_by_con = ComponentMap()
        self._vars_referenced_by_obj = ComponentSet()
        self._referenced_variables = ComponentMap()
        self._objective = None

        self._symbol_map = pyomo.core.base.SymbolMap()
        if self._symbolic_solver_labels:
            self._labeler = pyomo.core.base.TextLabeler()
        else:
            self._labeler = pyomo.core.base.NumericLabeler('x')

        if model.name is not None:
            self._solver_model = self._gurobipy.Model(model.name)
        else:
            self._solver_model = self._gurobipy.Model()

        self._add_block(model)

    def add_block(self, block):
        """Add a single Pyomo Block to the solver's model.

        This will keep any existing model components intact.

        Parameters
        ----------
        block: Block (scalar Block or single _BlockData)

        """
        if self._pyomo_model is None:
            raise RuntimeError('You must call set_instance before calling add_block.')
        self._add_block(block)

    def set_objective(self, obj):
        """
        Set the solver's objective. Note that, at least for now, any existing objective will be discarded. Other than
        that, any existing model components will remain intact.

        Parameters
        ----------
        obj: Objective
        """
        if self._pyomo_model is None:
            raise RuntimeError('You must call set_instance before calling set_objective.')
        return self._set_objective(obj)

    def add_constraint(self, con):
        """Add a single constraint to the solver's model.

        This will keep any existing model components intact.

        Parameters
        ----------
        con: Constraint (scalar Constraint or single _ConstraintData)

        """
        if self._pyomo_model is None:
            raise RuntimeError('You must call set_instance before calling add_constraint.')
        self._add_constraint(con)

    def add_var(self, var):
        """Add a single variable to the solver's model.

        This will keep any existing model components intact.

        Parameters
        ----------
        var: Var

        """
        if self._pyomo_model is None:
            raise RuntimeError('You must call set_instance before calling add_var.')
        self._add_var(var)

    def add_sos_constraint(self, con):
        """Add a single SOS constraint to the solver's model (if supported).

        This will keep any existing model components intact.

        Parameters
        ----------
        con: SOSConstraint

        """
        if self._pyomo_model is None:
            raise RuntimeError('You must call set_instance before calling add_sos_constraint.')
        self._add_sos_constraint(con)

    def remove_block(self, block):
        """Remove a single block from the solver's model.

        This will keep any other model components intact.

        WARNING: Users must call remove_block BEFORE modifying the block.

        Parameters
        ----------
        block: Block (scalar Block or a single _BlockData)

        """
        for sub_block in block.block_data_objects(descend_into=True, active=True):
            for con in sub_block.component_data_objects(ctype=pyomo.core.base.constraint.Constraint, descend_into=False, active=True):
                self.remove_constraint(con)

            for con in sub_block.component_data_objects(ctype=pyomo.core.base.sos.SOSConstraint, descend_into=False, active=True):
                self.remove_sos_constraint(con)

        for var in block.component_data_objects(ctype=pyomo.core.base.var.Var, descend_into=True, active=True):
            self.remove_var(var)

    def remove_constraint(self, con):
        """Remove a single constraint from the solver's model.

        This will keep any other model components intact.

        Parameters
        ----------
        con: Constraint (scalar Constraint or single _ConstraintData)

        """
        solver_con = self._pyomo_con_to_solver_con_map[con]
        self._remove_constraint(solver_con)
        self._symbol_map.removeSymbol(con)
        self._labeler.remove_obj(con)
        for var in self._vars_referenced_by_con[con]:
            self._referenced_variables[var] -= 1
        del self._vars_referenced_by_con[con]
        del self._pyomo_con_to_solver_con_map[con]
        del self._solver_con_to_pyomo_con_map[solver_con]

    def remove_sos_constraint(self, con):
        """Remove a single SOS constraint from the solver's model.

        This will keep any other model components intact.

        Parameters
        ----------
        con: SOSConstraint

        """
        solver_con = self._pyomo_con_to_solver_con_map[con]
        self._remove_sos_constraint(solver_con)
        self._symbol_map.removeSymbol(con)
        self._labeler.remove_obj(con)
        for var in self._vars_referenced_by_con[con]:
            self._referenced_variables[var] -= 1
        del self._vars_referenced_by_con[con]
        del self._pyomo_con_to_solver_con_map[con]
        del self._solver_con_to_pyomo_con_map[solver_con]

    def remove_var(self, var):
        """Remove a single variable from the solver's model.

        This will keep any other model components intact.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)

        """
        if self._referenced_variables[var] != 0:
            raise ValueError('Cannot remove Var {0} because it is still referenced by the '.format(var) +
                             'objective or one or more constraints')
        solver_var = self._pyomo_var_to_solver_var_map[var]
        self._remove_var(solver_var)
        self._symbol_map.removeSymbol(var)
        self._labeler.remove_obj(var)
        del self._referenced_variables[var]
        del self._pyomo_var_to_solver_var_map[var]
        del self._solver_var_to_pyomo_var_map[solver_var]

    def update_var(self, var):
        """Update a single variable in the solver's model.

        This will update bounds, fix/unfix the variable as needed, and
        update the variable type.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)

        """
        if var not in self._pyomo_var_to_solver_var_map:
            raise ValueError('The Var provided to update_var needs to be added first: {0}'.format(var))
        gurobipy_var = self._pyomo_var_to_solver_var_map[var]
        vtype = self._gurobi_vtype_from_var(var)
        if var.is_fixed():
            lb = var.value
            ub = var.value
        else:
            lb = -self._gurobipy.GRB.INFINITY
            ub = self._gurobipy.GRB.INFINITY
            if var.has_lb():
                lb = pyomo.core.expr.numvalue.value(var.lb)
            if var.has_ub():
                ub = pyomo.core.expr.numvalue.value(var.ub)
        gurobipy_var.setAttr('lb', lb)
        gurobipy_var.setAttr('ub', ub)
        gurobipy_var.setAttr('vtype', vtype)

    def _add_block(self, block):
        for var in block.component_data_objects(
                ctype=pyomo.core.base.var.Var,
                descend_into=True,
                active=True,
                sort=True):
            self._add_var(var)

        for sub_block in block.block_data_objects(descend_into=True,
                                                  active=True):
            for con in sub_block.component_data_objects(
                    ctype=pyomo.core.base.constraint.Constraint,
                    descend_into=False,
                    active=True,
                    sort=True):
                if (not con.has_lb()) and \
                   (not con.has_ub()):
                    assert not con.equality
                    continue  # non-binding, so skip
                self._add_constraint(con)

            for con in sub_block.component_data_objects(
                    ctype=pyomo.core.base.sos.SOSConstraint,
                    descend_into=False,
                    active=True,
                    sort=True):
                self._add_sos_constraint(con)

            obj_counter = 0
            for obj in sub_block.component_data_objects(
                    ctype=pyomo.core.base.objective.Objective,
                    descend_into=False,
                    active=True):
                obj_counter += 1
                if obj_counter > 1:
                    raise ValueError("Solver interface does not "
                                     "support multiple objectives.")
                self._set_objective(obj)

    def _add_var(self, var):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._gurobi_vtype_from_var(var)
        if var.has_lb():
            lb = pyomo.core.expr.numvalue.value(var.lb)
        else:
            lb = -self._gurobipy.GRB.INFINITY
        if var.has_ub():
            ub = pyomo.core.expr.numvalue.value(var.ub)
        else:
            ub = self._gurobipy.GRB.INFINITY

        if var.is_fixed():
            lb = var.value
            ub = var.value

        gurobipy_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype, name=varname)

        self._pyomo_var_to_solver_var_map[var] = gurobipy_var
        self._solver_var_to_pyomo_var_map[gurobipy_var] = var
        self._referenced_variables[var] = 0

    def _add_constraint(self, con):
        if not con.active:
            return None

        if pyomo.core.expr.numvalue.is_fixed(con.body):
            return None

        conname = self._symbol_map.getSymbol(con, self._labeler)

        gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(con.body, self._max_constraint_degree)

        if con.has_lb():
            if not pyomo.core.expr.numvalue.is_fixed(con.lower):
                raise ValueError("Lower bound of constraint {0} "
                                 "is not constant.".format(con))
        if con.has_ub():
            if not pyomo.core.expr.numvalue.is_fixed(con.upper):
                raise ValueError("Upper bound of constraint {0} "
                                 "is not constant.".format(con))

        if con.equality:
            gurobipy_con = self._solver_model.addConstr(lhs=gurobi_expr,
                                                        sense=self._gurobipy.GRB.EQUAL,
                                                        rhs=pyomo.core.expr.numvalue.value(con.lower),
                                                        name=conname)
        elif con.has_lb() and con.has_ub():
            gurobipy_con = self._solver_model.addRange(gurobi_expr,
                                                       pyomo.core.expr.numvalue.value(con.lower),
                                                       pyomo.core.expr.numvalue.value(con.upper),
                                                       name=conname)
            self._range_constraints.add(con)
        elif con.has_lb():
            gurobipy_con = self._solver_model.addConstr(lhs=gurobi_expr,
                                                        sense=self._gurobipy.GRB.GREATER_EQUAL,
                                                        rhs=pyomo.core.expr.numvalue.value(con.lower),
                                                        name=conname)
        elif con.has_ub():
            gurobipy_con = self._solver_model.addConstr(lhs=gurobi_expr,
                                                        sense=self._gurobipy.GRB.LESS_EQUAL,
                                                        rhs=pyomo.core.expr.numvalue.value(con.upper),
                                                        name=conname)
        else:
            raise ValueError("Constraint does not have a lower "
                             "or an upper bound: {0} \n".format(con))

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars
        self._pyomo_con_to_solver_con_map[con] = gurobipy_con
        self._solver_con_to_pyomo_con_map[gurobipy_con] = con

    def _add_sos_constraint(self, con):
        if not con.active:
            return None

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
            self._vars_referenced_by_con[con].add(v)
            gurobi_vars.append(self._pyomo_var_to_solver_var_map[v])
            self._referenced_variables[v] += 1
            weights.append(w)

        gurobipy_con = self._solver_model.addSOS(sos_type, gurobi_vars, weights)
        self._pyomo_con_to_solver_con_map[con] = gurobipy_con
        self._solver_con_to_pyomo_con_map[gurobipy_con] = con

    def _set_objective(self, obj):
        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[var] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None

        if obj.active is False:
            raise ValueError('Cannot add inactive objective to solver.')

        if obj.sense == minimize:
            sense = self._gurobipy.GRB.MINIMIZE
        elif obj.sense == maximize:
            sense = self._gurobipy.GRB.MAXIMIZE
        else:
            raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

        gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(obj.expr, self._max_obj_degree)

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        self._solver_model.setObjective(gurobi_expr, sense=sense)
        self._objective = obj
        self._vars_referenced_by_obj = referenced_vars

    def _remove_constraint(self, solver_con):
        try:
            self._solver_model.remove(solver_con)
        except (self._gurobipy.GurobiError, AttributeError):
            self._solver_model.update()
            self._solver_model.remove(solver_con)

    def _remove_sos_constraint(self, solver_sos_con):
        try:
            self._solver_model.remove(solver_sos_con)
        except (self._gurobipy.GurobiError, AttributeError):
            self._solver_model.update()
            self._solver_model.remove(solver_sos_con)

    def _remove_var(self, solver_var):
        try:
            self._solver_model.remove(solver_var)
        except (self._gurobipy.GurobiError, AttributeError):
            self._solver_model.update()
            self._solver_model.remove(solver_var)

    def _apply_solver(self, tee=False, load_duals=False):
        for block in self._pyomo_model.block_data_objects(descend_into=True,
                                                          active=True):
            for var in block.component_data_objects(ctype=pyomo.core.base.var.Var,
                                                    descend_into=False,
                                                    active=True,
                                                    sort=False):
                var.stale = True
        if tee:
            self._solver_model.setParam('OutputFlag', 1)
        else:
            self._solver_model.setParam('OutputFlag', 0)

        # Options accepted by gurobi (case insensitive):
        # ['Cutoff', 'IterationLimit', 'NodeLimit', 'SolutionLimit', 'TimeLimit',
        #  'FeasibilityTol', 'IntFeasTol', 'MarkowitzTol', 'MIPGap', 'MIPGapAbs',
        #  'OptimalityTol', 'PSDTol', 'Method', 'PerturbValue', 'ObjScale', 'ScaleFlag',
        #  'SimplexPricing', 'Quad', 'NormAdjust', 'BarIterLimit', 'BarConvTol',
        #  'BarCorrectors', 'BarOrder', 'Crossover', 'CrossoverBasis', 'BranchDir',
        #  'Heuristics', 'MinRelNodes', 'MIPFocus', 'NodefileStart', 'NodefileDir',
        #  'NodeMethod', 'PumpPasses', 'RINS', 'SolutionNumber', 'SubMIPNodes', 'Symmetry',
        #  'VarBranch', 'Cuts', 'CutPasses', 'CliqueCuts', 'CoverCuts', 'CutAggPasses',
        #  'FlowCoverCuts', 'FlowPathCuts', 'GomoryPasses', 'GUBCoverCuts', 'ImpliedCuts',
        #  'MIPSepCuts', 'MIRCuts', 'NetworkCuts', 'SubMIPCuts', 'ZeroHalfCuts', 'ModKCuts',
        #  'Aggregate', 'AggFill', 'PreDual', 'DisplayInterval', 'IISMethod', 'InfUnbdInfo',
        #  'LogFile', 'PreCrush', 'PreDepRow', 'PreMIQPMethod', 'PrePasses', 'Presolve',
        #  'ResultFile', 'ImproveStartTime', 'ImproveStartGap', 'Threads', 'Dummy', 'OutputFlag']
        for key, option in self.options.items():
            # When options come from the pyomo command, all
            # values are string types, so we try to cast
            # them to a numeric value in the event that
            # setting the parameter fails.
            try:
                self._solver_model.setParam(key, option)
            except TypeError:
                # we place the exception handling for
                # checking the cast of option to a float in
                # another function so that we can simply
                # call raise here instead of except
                # TypeError as e / raise e, because the
                # latter does not preserve the Gurobi stack
                # trace
                if not _is_numeric(option):
                    raise
                self._solver_model.setParam(key, float(option))

        if load_duals:
            self._solver_model.setParam('QCPDual', 1)

        binary_vars = ComponentSet()
        integer_vars = ComponentSet()
        if 'relax_integrality' in self.options:
            if self.options['relax_integrality']:
                self._solver_model.update()
                for v in self._solver_model.getVars():
                    if v.vtype == self._gurobipy.GRB.BINARY:
                        binary_vars.add(v)
                        v.vtype = self._gurobipy.GRB.CONTINUOUS
                    elif v.vtype == self._gurobipy.GRB.INTEGER:
                        integer_vars.add(v)
                        v.vtype = self._gurobipy.GRB.CONTINUOUS

        self._solver_model.optimize(self._callback)

        for v in binary_vars:
            v.vtype = self._gurobipy.GRB.BINARY
        for v in integer_vars:
            v.vtype = self._gurobipy.GRB.INTEGER

    def _postsolve(self, load_solutions=True, get_duals=False, get_slacks=False, get_reduced_costs=False):
        gprob = self._solver_model
        grb = self._gurobipy.GRB
        status = gprob.Status

        if gprob.getAttr(self._gurobipy.GRB.Attr.IsMIP):
            if get_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
            if get_duals:
                logger.warning("Cannot get duals for MIP.")
            get_reduced_costs = False
            get_duals = False

        results = Results()
        results.wallclock_time = gprob.Runtime

        if status == grb.LOADED:  # problem is loaded, but no solution
            results.termination_condition = TerminationCondition.error
        elif status == grb.OPTIMAL:  # optimal
            results.termination_condition = TerminationCondition.optimal
        elif status == grb.INFEASIBLE:
            results.termination_condition = TerminationCondition.infeasible
        elif status == grb.INF_OR_UNBD:
            results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif status == grb.UNBOUNDED:
            results.termination_condition = TerminationCondition.unbounded
        elif status == grb.CUTOFF:
            results.termination_condition = TerminationCondition.minFunctionValue
        elif status == grb.ITERATION_LIMIT:
            results.termination_condition = TerminationCondition.maxIterations
        elif status == grb.NODE_LIMIT:
            results.termination_condition = TerminationCondition.maxEvaluations
        elif status == grb.TIME_LIMIT:
            results.termination_condition = TerminationCondition.maxTimeLimit
        elif status == grb.SOLUTION_LIMIT:
            results.termination_condition = TerminationCondition.unknown
        elif status == grb.INTERRUPTED:
            results.termination_condition = TerminationCondition.error
        elif status == grb.NUMERIC:
            results.termination_condition = TerminationCondition.error
        elif status == grb.SUBOPTIMAL:
            results.termination_condition = TerminationCondition.other
        # note that USER_OBJ_LIMIT was added in Gurobi 7.0, so it may not be present
        elif (status is not None) and (status == getattr(grb,'USER_OBJ_LIMIT', None)):
            results.termination_condition = TerminationCondition.other
        else:
            results.termination_condition = TerminationCondition.error

        if (gprob.NumBinVars + gprob.NumIntVars) == 0:
            try:
                results.upper_bound = gprob.ObjVal
                results.lower_bound = gprob.ObjVal
                results.objective_value = gprob.ObjVal
            except (self._gurobipy.GurobiError, AttributeError):
                pass
        elif gprob.ModelSense == 1:  # minimizing
            try:
                results.upper_bound = gprob.ObjVal
            except (self._gurobipy.GurobiError, AttributeError):
                pass
            try:
                results.lower_bound = gprob.ObjBound
            except (self._gurobipy.GurobiError, AttributeError):
                pass
        elif gprob.ModelSense == -1:  # maximizing
            try:
                results.upper_bound = gprob.ObjBound
            except (self._gurobipy.GurobiError, AttributeError):
                pass
            try:
                results.lower_bound = gprob.ObjVal
            except (self._gurobipy.GurobiError, AttributeError):
                pass
        else:
            raise RuntimeError('Unrecognized gurobi objective sense: {0}'.format(gprob.ModelSense))

        if gprob.SolCount > 0:
            if load_solutions:
                self.load_vars()
            if get_duals:
                results.duals = self.get_duals()
            if get_reduced_costs:
                results.reduced_costs = self.get_reduced_costs()
            if get_slacks:
                results.slacks = self.get_slacks()

        return results

    def solve(self, tee=False, load_solutions=True, get_duals=False, get_reduced_costs=False, get_slacks=False):
        """
        Solve the model.

        Keyword Arguments
        -----------------
        tee: bool
            If True, then the solver log will be printed.
        load_solutions: bool
            If True and a solution exists, the solution will be loaded into the Pyomo model.
        """
        t0 = time.time()

        if self._pyomo_model is None:
            msg = 'Please use set_instance to set the instance before calling solve with the persistent'
            msg += ' solver interface.'
            raise RuntimeError(msg)

        self._apply_solver(tee=tee, load_duals=get_duals)
        results = self._postsolve(load_solutions=load_solutions, get_duals=get_duals,
                                  get_reduced_costs=get_reduced_costs, get_slacks=get_slacks)

        results.total_wallclock_time = time.time() - t0

        return results

    def _get_expr_from_pyomo_repn(self, repn, max_degree=2):
        referenced_vars = ComponentSet()

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > max_degree):
            raise DegreeError('GurobiDirect does not support expressions of degree {0}.'.format(degree))

        if len(repn.linear_vars) > 0:
            referenced_vars.update(repn.linear_vars)
            new_expr = self._gurobipy.LinExpr(repn.linear_coefs, [self._pyomo_var_to_solver_var_map[i] for i in repn.linear_vars])
        else:
            new_expr = 0.0

        for i,v in enumerate(repn.quadratic_vars):
            x,y = v
            new_expr += repn.quadratic_coefs[i] * self._pyomo_var_to_solver_var_map[x] * self._pyomo_var_to_solver_var_map[y]
            referenced_vars.add(x)
            referenced_vars.add(y)

        new_expr += repn.constant

        return new_expr, referenced_vars

    def _get_expr_from_pyomo_expr(self, expr, max_degree=2):
        if max_degree == 2:
            repn = generate_standard_repn(expr, quadratic=True)
        else:
            repn = generate_standard_repn(expr, quadratic=False)

        try:
            gurobi_expr, referenced_vars = self._get_expr_from_pyomo_repn(repn, max_degree)
        except DegreeError as e:
            msg = e.args[0]
            msg += '\nexpr: {0}'.format(expr)
            raise DegreeError(msg)

        return gurobi_expr, referenced_vars

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

    def warm_start(self):
        self._solver_model.update()
        for pyomo_var, gurobipy_var in self._pyomo_var_to_solver_var_map.items():
            if pyomo_var.is_binary():
                if pyomo_var.value is not None:
                    gurobipy_var.setAttr(self._gurobipy.GRB.Attr.Start, pyomo.core.expr.numvalue.value(pyomo_var))

    def load_vars(self, vars_to_load=None):
        """
        Load the values from the solver's variables into the corresponding pyomo variables.

        Parameters
        ----------
        vars_to_load: list of Var
        """
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        gurobi_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        if self._solver_model.getAttr(self._gurobipy.GRB.Attr.IsMIP):
            vals = self._solver_model.getAttr("Xn", gurobi_vars_to_load)
        else:
            vals = self._solver_model.getAttr("X", gurobi_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                var.stale = False
                var.value = val

    def get_reduced_costs(self, vars_to_load=None):
        """
        Load the reduced costs into a component map

        Parameters
        ----------
        vars_to_load: list of Var
        """
        if self._solver_model.getAttr(self._gurobipy.GRB.Attr.IsMIP):
            raise ValueError("Cannot get reduced costs for MIP.")
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        gurobi_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getAttr("Rc", gurobi_vars_to_load)
        rc = ComponentMap()
        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                rc[var] = val
        return rc

    def get_duals(self, cons_to_load=None):
        """
        Load the duals into a ComponentMap

        Parameters
        ----------
        cons_to_load: list of Constraint
        """
        if self._solver_model.getAttr(self._gurobipy.GRB.Attr.IsMIP):
            raise ValueError("Cannot get duals for MIP.")
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.getConstrs()
            quadratic_cons_to_load = self._solver_model.getQConstrs()
        else:
            gurobi_cons_to_load = set([con_map[pyomo_con] for pyomo_con in cons_to_load])
            linear_cons_to_load = gurobi_cons_to_load.intersection(set(self._solver_model.getConstrs()))
            quadratic_cons_to_load = gurobi_cons_to_load.intersection(set(self._solver_model.getQConstrs()))
        linear_vals = self._solver_model.getAttr("Pi", linear_cons_to_load)
        quadratic_vals = self._solver_model.getAttr("QCPi", quadratic_cons_to_load)

        dual = ComponentMap()
        for gurobi_con, val in zip(linear_cons_to_load, linear_vals):
            pyomo_con = reverse_con_map[gurobi_con]
            dual[pyomo_con] = val
        for gurobi_con, val in zip(quadratic_cons_to_load, quadratic_vals):
            pyomo_con = reverse_con_map[gurobi_con]
            dual[pyomo_con] = val
        return dual

    def get_slacks(self, cons_to_load=None):
        """
        Load the values of the slack variables into a ComponentMap

        Parameters
        ----------
        cons_to_load: list of Constraint
        """
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        slack = ComponentMap()

        gurobi_range_con_vars = set(self._solver_model.getVars()) - set(self._pyomo_var_to_solver_var_map.values())

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.getConstrs()
            quadratic_cons_to_load = self._solver_model.getQConstrs()
        else:
            gurobi_cons_to_load = set([con_map[pyomo_con] for pyomo_con in cons_to_load])
            linear_cons_to_load = gurobi_cons_to_load.intersection(set(self._solver_model.getConstrs()))
            quadratic_cons_to_load = gurobi_cons_to_load.intersection(set(self._solver_model.getQConstrs()))
        linear_vals = self._solver_model.getAttr("Slack", linear_cons_to_load)
        quadratic_vals = self._solver_model.getAttr("QCSlack", quadratic_cons_to_load)

        for gurobi_con, val in zip(linear_cons_to_load, linear_vals):
            pyomo_con = reverse_con_map[gurobi_con]
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
            pyomo_con = reverse_con_map[gurobi_con]
            slack[pyomo_con] = val

        return slack

    def write(self, filename):
        """
        Write the model to a file (e.g., and lp file).

        Parameters
        ----------
        filename: str
            Name of the file to which the model should be written.
        """
        self._solver_model.write(filename)

    def set_linear_constraint_attr(self, con, attr, val):
        """
        Set the value of an attribute on a gurobi linear constraint.

        Paramaters
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
        try:
            self._pyomo_con_to_solver_con_map[con].setAttr(attr, val)
        except (self._gurobipy.GurobiError, AttributeError):
            self._solver_model.update()
            self._pyomo_con_to_solver_con_map[con].setAttr(attr, val)

    def set_var_attr(self, var, attr, val):
        """
        Set the value of an attribute on a gurobi variable.

        Paramaters
        ----------
        con: pyomo.core.base.var._GeneralVarData
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
        try:
            self._pyomo_var_to_solver_var_map[var].setAttr(attr, val)
        except (self._gurobipy.GurobiError, AttributeError):
            self._solver_model.update()
            self._pyomo_var_to_solver_var_map[var].setAttr(attr, val)

    def get_model_attr(self, attr):
        """
        Get the value of an attribute on the Gurobi model.

        Parameters
        ----------
        attr: str
            The attribute to get. See Gurobi documentation for descriptions of the attributes.
            Options are:
                NumVars
                NumConstrs
                NumSOS
                NumQConstrs
                NumgGenConstrs
                NumNZs
                DNumNZs
                NumQNZs
                NumQCNZs
                NumIntVars
                NumBinVars
                NumPWLObjVars
                ModelName
                ModelSense
                ObjCon
                ObjVal
                ObjBound
                ObjBoundC
                PoolObjBound
                PoolObjVal
                MIPGap
                Runtime
                Status
                SolCount
                IterCount
                BarIterCount
                NodeCount
                IsMIP
                IsQP
                IsQCP
                IsMultiObj
                IISMinimal
                MaxCoeff
                MinCoeff
                MaxBound
                MinBound
                MaxObjCoeff
                MinObjCoeff
                MaxRHS
                MinRHS
                MaxQCCoeff
                MinQCCoeff
                MaxQCLCoeff
                MinQCLCoeff
                MaxQCRHS
                MinQCRHS
                MaxQObjCoeff
                MinQObjCoeff
                Kappa
                KappaExact
                FarkasProof
                TuneResultCount
                LicenseExpiration
                BoundVio
                BoundSVio
                BoundVioIndex
                BoundSVioIndex
                BoundVioSum
                BoundSVioSum
                ConstrVio
                ConstrSVio
                ConstrVioIndex
                ConstrSVioIndex
                ConstrVioSum
                ConstrSVioSum
                ConstrResidual
                ConstrSResidual
                ConstrResidualIndex
                ConstrSResidualIndex
                ConstrResidualSum
                ConstrSResidualSum
                DualVio
                DualSVio
                DualVioIndex
                DualSVioIndex
                DualVioSum
                DualSVioSum
                DualResidual
                DualSResidual
                DualResidualIndex
                DualSResidualIndex
                DualResidualSum
                DualSResidualSum
                ComplVio
                ComplVioIndex
                ComplVioSum
                IntVio
                IntVioIndex
                IntVioSum
        """
        try:
            return self._solver_model.getAttr(attr)
        except (self._gurobipy.GurobiError, AttributeError):
            self._solver_model.update()
            return self._solver_model.getAttr(attr)

    def get_var_attr(self, var, attr):
        """
        Get the value of an attribute on a gurobi var.

        Paramaters
        ----------
        var: pyomo.core.base.var._GeneralVarData
            The pyomo var for which the corresponding gurobi var attribute
            should be retrieved.
        attr: str
            The attribute to get. Options are:
                LB
                UB
                Obj
                VType
                VarName
                X
                Xn
                RC
                BarX
                Start
                VarHintVal
                VarHintPri
                BranchPriority
                VBasis
                PStart
                IISLB
                IISUB
                PWLObjCvx
                SAObjLow
                SAObjUp
                SALBLow
                SALBUp
                SAUBLow
                SAUBUp
                UnbdRay
        """
        try:
            return self._pyomo_var_to_solver_var_map[var].getAttr(attr)
        except (self._gurobipy.GurobiError, AttributeError):
            self._solver_model.update()
            return self._pyomo_var_to_solver_var_map[var].getAttr(attr)

    def get_linear_constraint_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi linear constraint.

        Paramaters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. Options are:
                Sense
                RHS
                ConstrName
                Pi
                Slack
                CBasis
                DStart
                Lazy
                IISConstr
                SARHSLow
                SARHSUp
                FarkasDual
        """
        try:
            return self._pyomo_con_to_solver_con_map[con].getAttr(attr)
        except (self._gurobipy.GurobiError, AttributeError):
            self._solver_model.update()
            return self._pyomo_con_to_solver_con_map[con].getAttr(attr)

    def get_sos_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi sos constraint.

        Paramaters
        ----------
        con: pyomo.core.base.sos._SOSConstraintData
            The pyomo SOS constraint for which the corresponding gurobi SOS constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. Options are:
                IISSOS
        """
        try:
            return self._pyomo_con_to_solver_con_map[con].getAttr(attr)
        except (self._gurobipy.GurobiError, AttributeError):
            self._solver_model.update()
            return self._pyomo_con_to_solver_con_map[con].getAttr(attr)

    def get_quadratic_constraint_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi quadratic constraint.

        Paramaters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. Options are:
                QCSense
                QCRHS
                QCName
                QCPi
                QCSlack
                IISQConstr
        """
        try:
            return self._pyomo_con_to_solver_con_map[con].getAttr(attr)
        except (self._gurobipy.GurobiError, AttributeError):
            self._solver_model.update()
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
            example:

            >>> from gurobipy import GRB
            >>> import pyomo.environ as pe
            >>> m = pe.ConcreteModel()
            >>> m.x = pe.Var(within=pe.Binary)
            >>> m.y = pe.Var(within=pe.Binary)
            >>> m.obj = pe.Objective(expr=m.x + m.y)
            >>> opt = pe.SolverFactory('gurobi_persistent')
            >>> opt.set_instance(m)
            >>> def my_callback(cb_m, cb_opt, cb_where):
            ...     if cb_where == GRB.Callback.MIPNODE:
            ...         status = cb_opt.cbGet(GRB.Callback.MIPNODE_STATUS)
            ...         if status == GRB.OPTIMAL:
            ...             cb_opt.cbGetNodeRel([cb_m.x, cb_m.y])
            ...             if cb_m.x.value + cb_m.y.value > 1.1:
            ...                 cb_opt.cbCut(pe.Constraint(expr=cb_m.x + cb_m.y <= 1))
            >>> opt.set_callback(my_callback)
            >>> opt.solve()
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

        if pyomo.core.expr.numvalue.is_fixed(con.body):
            raise ValueError('cbCut expected a non-trival constraint')

        gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(con.body, self._max_constraint_degree)

        if con.has_lb():
            if con.has_ub():
                raise ValueError('Range constraints are not supported in cbCut.')
            if not pyomo.core.expr.numvalue.is_fixed(con.lower):
                raise ValueError('Lower bound of constraint {0} is not constant.'.format(con))
        if con.has_ub():
            if not pyomo.core.expr.numvalue.is_fixed(con.upper):
                raise ValueError('Upper bound of constraint {0} is not constant.'.format(con))

        if con.equality:
            self._solver_model.cbCut(lhs=gurobi_expr, sense=self._gurobipy.GRB.EQUAL,
                                     rhs=pyomo.core.expr.numvalue.value(con.lower))
        elif con.has_lb() and (pyomo.core.expr.numvalue.value(con.lower) > -float('inf')):
            self._solver_model.cbCut(lhs=gurobi_expr, sense=self._gurobipy.GRB.GREATER_EQUAL,
                                     rhs=pyomo.core.expr.numvalue.value(con.lower))
        elif con.has_ub() and (pyomo.core.expr.numvalue.value(con.upper) < float('inf')):
            self._solver_model.cbCut(lhs=gurobi_expr, sense=self._gurobipy.GRB.LESS_EQUAL,
                                     rhs=pyomo.core.expr.numvalue.value(con.upper))
        else:
            raise ValueError('Constraint does not have a lower or an upper bound {0} \n'.format(con))

    def cbGet(self, what):
        return self._solver_model.cbGet(what)

    def cbGetNodeRel(self, vars):
        """
        Load the values of the specified variables from the node relaxation solution at the current node.

        Parameters
        ----------
        vars: Var or iterable of vars
        """
        if not isinstance(vars, collections.Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[i] for i in vars]
        var_values = self._solver_model.cbGetNodeRel(gurobi_vars)
        for i, v in enumerate(vars):
            v.value = var_values[i]

    def cbGetSolution(self, vars):
        """
        Load the values of the specified variables from the new MIP solution.

        Parameters
        ----------
        vars: iterable of vars
        """
        if not isinstance(vars, collections.Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[i] for i in vars]
        var_values = self._solver_model.cbGetSolution(gurobi_vars)
        for i, v in enumerate(vars):
            v.value = var_values[i]

    def cbLazy(self, con):
        """
        Add a lazy constraint within a callback. See gurobi docs for details.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The lazy constraint to add
        """
        if not con.active:
            raise ValueError('cbLazy expected an active constraint.')

        if pyomo.core.expr.numvalue.is_fixed(con.body):
            raise ValueError('cbLazy expected a non-trival constraint')

        gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(con.body, self._max_constraint_degree)

        if con.has_lb():
            if con.has_ub():
                raise ValueError('Range constraints are not supported in cbLazy.')
            if not pyomo.core.expr.numvalue.is_fixed(con.lower):
                raise ValueError('Lower bound of constraint {0} is not constant.'.format(con))
        if con.has_ub():
            if not pyomo.core.expr.numvalue.is_fixed(con.upper):
                raise ValueError('Upper bound of constraint {0} is not constant.'.format(con))

        if con.equality:
            self._solver_model.cbLazy(lhs=gurobi_expr, sense=self._gurobipy.GRB.EQUAL,
                                      rhs=pyomo.core.expr.numvalue.value(con.lower))
        elif con.has_lb() and (pyomo.core.expr.numvalue.value(con.lower) > -float('inf')):
            self._solver_model.cbLazy(lhs=gurobi_expr, sense=self._gurobipy.GRB.GREATER_EQUAL,
                                      rhs=pyomo.core.expr.numvalue.value(con.lower))
        elif con.has_ub() and (pyomo.core.expr.numvalue.value(con.upper) < float('inf')):
            self._solver_model.cbLazy(lhs=gurobi_expr, sense=self._gurobipy.GRB.LESS_EQUAL,
                                      rhs=pyomo.core.expr.numvalue.value(con.upper))
        else:
            raise ValueError('Constraint does not have a lower or an upper bound {0} \n'.format(con))

    def cbSetSolution(self, vars, solution):
        if not isinstance(vars, collections.Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[i] for i in vars]
        self._solver_model.cbSetSolution(gurobi_vars, solution)

    def cbUseSolution(self):
        return self._solver_model.cbUseSolution()

    def update(self):
        self._solver_model.update()

    def has_instance(self):
        """
        True if set_instance has been called and this solver interface has a pyomo model and a solver model.

        Returns
        -------
        tmp: bool
        """
        return self._pyomo_model is not None
