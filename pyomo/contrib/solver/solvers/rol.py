#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import datetime
import io
import logging
from typing import Tuple, List, Optional, Sequence, Mapping, Dict

from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.numeric_types import native_numeric_types
from pyomo.common.errors import InfeasibleConstraintException, ApplicationError
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.block import BlockData
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import Constraint, ConstraintData
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.staleflag import StaleFlagManager
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.solver.common.base import SolverBase, Availability
from pyomo.contrib.solver.common.config import SolverConfig
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoSolutionError,
)
from pyomo.contrib.solver.common.solution_loader import NoSolutionSolutionLoader
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.solution_loader import (
    SolutionLoaderBase,
    load_import_suffixes,
)
from pyomo.common.config import ConfigValue, ConfigDict
from pyomo.common.tee import capture_output, TeeStream
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AmplNLP
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.repn.plugins.nl_writer import NLWriter


logger = logging.getLogger(__name__)


pyrol, pyrol_available = attempt_import('pyrol')


if pyrol_available:
    from pyrol import Objective, Constraint, Problem, Solver, getCout, Bounds
    from pyrol.vectors import NumPyVector
    from pyrol.pyrol.Teuchos import ParameterList
else:
    Objective = object
    Constraint = object


class PynumeroObjective(Objective):
    def __init__(self, nlp: ExtendedNLP):
        super().__init__()
        self.nlp: ExtendedNLP = nlp
        self.zero_duals = np.zeros(self.nlp.n_constraints(), dtype=float)

    def value(self, x, tol):
        self.nlp.set_primals(x.array)
        return self.nlp.evaluate_objective()
    
    def gradient(self, g, x, tol):
        self.nlp.set_primals(x.array)
        g[:] = self.nlp.evaluate_grad_objective()

    def hessVec(self, hv, v, x, tol):
        # hack
        # set the duals to 0
        # evaluate the hessian of the lagrangian
        # with the duals set to 0, that should give 
        # the hessian of the objective
        # then do the matvec with v
        self.nlp.set_primals(x.array)
        self.nlp.set_duals(self.zero_duals)
        hess: coo_matrix = self.nlp.evaluate_hessian_lag()
        hv[:] = hess.dot(v.array)


class PynumeroEqConstraint(Constraint):
    def __init__(self, nlp: ExtendedNLP):
        super().__init__()
        self.nlp: ExtendedNLP = nlp
        self.zero_duals = np.zeros(self.nlp.n_constraints(), dtype=float)

    def value(self, c, x, tol):
        self.nlp.set_primals(x.array)
        c[:] = self.nlp.evaluate_eq_constraints()

    def applyJacobian(self, jv, v, x, tol):
        # print('apply jacobian eq')
        self.nlp.set_primals(x.array)
        jac: coo_matrix = self.nlp.evaluate_jacobian_eq()
        jv[:] = jac.dot(v.array)

    def applyAdjointJacobian(self, ajv, v, x, tol):
        # print('adjoint jacobian eq')
        self.nlp.set_primals(x.array)
        jac: coo_matrix = self.nlp.evaluate_jacobian_eq()
        ajv[:] = jac.transpose().dot(v.array)

    def applyAdjointHessian(self, ahuv, u, v, x, tol):
        # hack
        # set the duals to 0
        # evaluate the hessian of the lagrangian
        # with the duals set to 0, that should give the hessian of the objective
        # now set the duals of the equality contraints to u
        # compute the hessian of the lagrangian
        # subtract the hessian of the objective
        # now we should have the hessian applied with u
        # print('adjoint hessian eq')
        self.nlp.set_primals(x.array)
        self.nlp.set_duals(self.zero_duals)
        hess_obj: coo_matrix = self.nlp.evaluate_hessian_lag()
        self.nlp.set_duals_eq(u.array)
        hess: coo_matrix = self.nlp.evaluate_hessian_lag() - hess_obj
        ahuv[:] = hess.dot(v.array)


class PynumeroIneqConstraint(Constraint):
    def __init__(self, nlp: ExtendedNLP):
        super().__init__()
        self.nlp: ExtendedNLP = nlp
        self.zero_duals = np.zeros(self.nlp.n_constraints(), dtype=float)

    def value(self, c, x, tol):
        self.nlp.set_primals(x.array)
        c[:] = self.nlp.evaluate_ineq_constraints()

    def applyJacobian(self, jv, v, x, tol):
        # print('apply jacobian ineq')
        self.nlp.set_primals(x.array)
        jac: coo_matrix = self.nlp.evaluate_jacobian_ineq()
        jv[:] = jac.dot(v.array)

    def applyAdjointJacobian(self, ajv, v, x, tol):
        # print('adjoint jacobian ineq')
        self.nlp.set_primals(x.array)
        jac: coo_matrix = self.nlp.evaluate_jacobian_ineq()
        ajv[:] = jac.transpose().dot(v.array)

    def applyAdjointHessian(self, ahuv, u, v, x, tol):
        # hack
        # set the duals to 0
        # evaluate the hessian of the lagrangian
        # with the duals set to 0, that should give the hessian of the objective
        # now set the duals of the equality contraints to u
        # compute the hessian of the lagrangian
        # subtract the hessian of the objective
        # now we should have the hessian applied with u
        # print('adjoint hessian ineq')
        self.nlp.set_primals(x.array)
        self.nlp.set_duals(self.zero_duals)
        hess_obj: coo_matrix = self.nlp.evaluate_hessian_lag()
        self.nlp.set_duals_ineq(u.array)
        hess: coo_matrix = self.nlp.evaluate_hessian_lag() - hess_obj
        ahuv[:] = hess.dot(v.array)


class PyrolConfig(SolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.writer_config: ConfigDict = self.declare(
            'writer_config', NLWriter.CONFIG()
        )
        self.feasibility_tol: float = self.declare(
            'feasibility_tol', 
            ConfigValue(
                default=1e-6, 
                domain=float, 
                description="tolerance for checking for feasible solutions",
            ),
        )
        self.run_checks: bool = self.declare(
            'run_checks',
            ConfigValue(
                default=False,
                domain=bool,
                description="run pyrol model checks",
            )
        )


logger = logging.getLogger(__name__)


class PyrolSolutionLoader(SolutionLoaderBase):
    def __init__(
        self,
        primal_arr,
        dual_arr,
        pyomo_vars,
        pyomo_cons,
        pyomo_model,
    ) -> None:
        super().__init__()
        self._primal_arr = primal_arr
        self._dual_arr = dual_arr
        self._pyomo_vars = pyomo_vars
        self._pyomo_cons = pyomo_cons
        self._pyomo_model = pyomo_model

    def get_number_of_solutions(self) -> int:
        return 1

    def get_solution_ids(self) -> List:
        return [None]

    def load_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> None:
        for v, val in self.get_vars(vars_to_load=vars_to_load, solution_id=solution_id).items():
            v.value = val

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> Mapping[VarData, float]:
        if self.get_number_of_solutions() == 0:
            raise NoSolutionError()
        if vars_to_load is None:
            vars_to_load = self._pyomo_vars
        vars_to_load = ComponentSet(vars_to_load)
        res = ComponentMap()
        for v, val in zip(self._pyomo_vars, self._primal_arr):
            if v in vars_to_load:
                res[v] = val
        return res

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> Mapping[VarData, float]:
        return NotImplemented

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None, solution_id=None
    ) -> Dict[ConstraintData, float]:
        if self.get_number_of_solutions() == 0:
            raise NoSolutionError()
        if cons_to_load is None:
            cons_to_load = self._pyomo_cons
        cons_to_load = set(cons_to_load)
        res = {}
        for c, val in zip(self._pyomo_cons, self._dual_arr):
            if c in cons_to_load:
                res[c] = val
        return res

    def load_import_suffixes(self, solution_id=None):
        load_import_suffixes(self._pyomo_model, self, solution_id=solution_id)


def _parse_rol_output(solver_log):
    for line in solver_log.splitlines():
        if line.startswith('Optimization Terminated with Status:'):
            msg = line.split(':')[-1]
            msg = msg.strip()
            if msg == 'Converged':
                return True
            else:
                return False
    raise RuntimeError('did not find termination line')


class PyrolInterface(SolverBase):

    _available = None

    CONFIG = PyrolConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._nlp = None
        self._x = None

    def _clear(self):
        self._nlp = None
        self._x = None

    def available(self) -> Availability:
        if self._available is not None:
            return self._available
        
        if not pyrol_available:
            PyrolInterface._available = Availability.NotFound
        else:
            PyrolInterface._available = Availability.FullLicense

        return self._available
    
    def version(self) -> Tuple:
        raise NotImplementedError('not done')
        return tuple(int(i) for i in scip.__version__.split('.'))

    def solve(self, model: BlockData, **kwds) -> Results:
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        orig_config = self.config
        if not self.available():
            raise ApplicationError(
                f'{self.name} is not available: {self.available()}'
            )
        TempfileManager.push()
        try:
            config = self.config(value=kwds, preserve_implicit=True)

            # hack to work around legacy solver wrapper __setattr__
            # otherwise, this would just be self.config = config
            object.__setattr__(self, 'config', config)

            StaleFlagManager.mark_all_as_stale()

            if config.timer is None:
                config.timer = HierarchicalTimer()
            timer = config.timer

            ostreams = [io.StringIO()] + config.tee

            pyrol_model, solution_loader, has_obj = self._create_solver_model(model)

            params = ParameterList()
            params['General'] = ParameterList()
            params['General']['Output Level'] = 1
            solver = Solver(pyrol_model, params)
            stream = getCout()

            with capture_output(TeeStream(*ostreams), capture_fd=True):
                if self.config.run_checks:
                    timer.start('check pyrol problem')
                    pyrol_model.check(True, stream)
                    timer.stop('check pyrol problem')
                timer.start('optimize')
                solver.solve(stream)
                timer.stop('optimize')

            solver_log = ostreams[0].getvalue()
            results = self._postsolve(solution_loader, has_obj, solver_log)
        except InfeasibleConstraintException:
            results = self._get_infeasible_results()
        finally:
            # hack to work around legacy solver wrapper __setattr__
            # otherwise, this would just be self.config = orig_config
            object.__setattr__(self, 'config', orig_config)
            TempfileManager.pop()

        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        results.timing_info.start_timestamp = start_timestamp
        results.timing_info.wall_time = (end_timestamp - start_timestamp).total_seconds()
        results.timing_info.timer = timer
        return results

    def _get_infeasible_results(self):
        res = Results()
        res.solution_loader = NoSolutionSolutionLoader()
        res.solution_status = SolutionStatus.noSolution
        res.termination_condition = TerminationCondition.provenInfeasible
        res.incumbent_objective = None
        res.objective_bound = None
        res.iteration_count = None
        res.solver_config = self.config
        res.solver_name = self.name
        # res.solver_version = self.version()
        if self.config.raise_exception_on_nonoptimal_result:
            raise NoOptimalSolutionError()
        if self.config.load_solutions:
            raise NoFeasibleSolutionError()
        return res

    def _create_solver_model(self, model):
        timer = self.config.timer
        timer.start('create pyrol model')
        self._clear()

        timer.start('write nl file')
        nl_fname = TempfileManager.create_tempfile(suffix='.nl')
        # row_fname = TempfileManager.create_tempfile(suffix='.row')
        # col_fname = TempfileManager.create_tempfile(suffix='.col')
        with (
            open(nl_fname, 'w', newline='\n', encoding='utf-8') as nl_file,
            # open(row_fname, 'w', encoding='utf-8') as row_file,
            # open(col_fname, 'w', encoding='utf-8') as col_file,
        ):
            writer = NLWriter()
            writer.config.set_value(self.config.writer_config)
            nl_info = writer.write(
                model,
                nl_file,
                # row_file,
                # col_file,
            )
        timer.stop('write nl file')

        timer.start('create AmplNLP')
        nlp = AmplNLP(
            nl_file=nl_fname,
            # row_filename=row_fname,
            # col_filename=col_fname,
        )
        self._nlp = nlp
        timer.stop('create AmplNLP')

        timer.start('create pyrol problem')
        stream = getCout()
        obj = PynumeroObjective(nlp=nlp)
        self._x = x = NumPyVector(nlp.get_primals())
        problem = Problem(obj, x)
        if nlp.n_eq_constraints() > 0:
            eq_con = PynumeroEqConstraint(nlp=nlp)
            mult = NumPyVector(nlp.get_duals_eq())
            problem.addConstraint('equality_constraints', eq_con, mult)
        if nlp.n_ineq_constraints() > 0:
            ineq_con = PynumeroIneqConstraint(nlp=nlp)
            mult = NumPyVector(nlp.get_duals_ineq())
            lbs = NumPyVector(nlp.ineq_lb())
            ubs = NumPyVector(nlp.ineq_ub())
            bounds = Bounds(lbs, ubs)
            problem.addConstraint('inequality_constraints', ineq_con, mult, bounds)
        if np.any(np.isfinite(nlp.primals_lb())) or np.any(np.isfinite(nlp.primals_ub())):
            lbs = NumPyVector(nlp.primals_lb())
            ubs = NumPyVector(nlp.primals_ub())
            bounds = Bounds(lbs, ubs)
            problem.addBoundConstraint(bounds)
        timer.stop('create pyrol problem')

        # right now, the NLP object requires an objective
        has_obj = True
        solution_loader = PyrolSolutionLoader(
            primal_arr=x,
            dual_arr=None,
            pyomo_vars=nl_info.variables,
            pyomo_cons=nl_info.constraints,
            pyomo_model=model,
        )
        timer.stop('create pyrol model')
        return problem, solution_loader, has_obj

    def _check_feasibility(self):
        nlp: ExtendedNLP = self._nlp
        nlp.set_primals(self._x.array)
        err = 0
        lb_err = np.max(nlp.primals_lb() - nlp.get_primals())
        err = max(err, lb_err)
        ub_err = np.max(nlp.get_primals() - nlp.primals_ub())
        err = max(err, ub_err)
        eq_err = np.max(np.abs(nlp.evaluate_eq_constraints()))
        err = max(err, eq_err)
        ineq_lb_err = np.max(nlp.ineq_lb() - nlp.evaluate_ineq_constraints())
        err = max(err, ineq_lb_err)
        ineq_ub_err = np.max(nlp.evaluate_ineq_constraints() - nlp.ineq_ub())
        err = max(err, ineq_ub_err)
        return err

    def _postsolve(
        self, 
        solution_loader: PyrolSolutionLoader, 
        has_obj,
        solver_log,
    ):
        results = Results()
        results.solver_log = solver_log
        results.solution_loader = solution_loader
        success = _parse_rol_output(solver_log)
        if success:
            results.termination_condition = TerminationCondition.convergenceCriteriaSatisfied
            results.solution_status = SolutionStatus.optimal
        else:
            results.termination_condition = TerminationCondition.unknown
            err = self._check_feasibility()
            if err > self.config.feasibility_tol:
                results.solution_status = SolutionStatus.infeasible
            else:
                results.solution_status = SolutionStatus.feasible
        
        if (
            results.termination_condition 
            != TerminationCondition.convergenceCriteriaSatisfied
            and self.config.raise_exception_on_nonoptimal_result
        ):
            raise NoOptimalSolutionError()

        if results.solution_status in {SolutionStatus.optimal, SolutionStatus.feasible}:
            self._nlp.set_primals(self._x.array)
            results.incumbent_objective = self._nlp.evaluate_objective()
        else:
            results.incumbent_objective = None
        results.objective_bound = None

        self.config.timer.start('load solution')
        if self.config.load_solutions:
            if results.solution_status in {SolutionStatus.optimal, SolutionStatus.feasible}:
                solution_loader.load_solution()
            else:
                raise NoFeasibleSolutionError()
        self.config.timer.stop('load solution')

        results.solver_config = self.config
        results.solver_name = self.name
        # results.solver_version = self.version()

        return results
