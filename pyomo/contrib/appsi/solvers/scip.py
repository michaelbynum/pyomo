from pyomo.common.tempfiles import TempfileManager
from pyomo.common.fileutils import Executable
from pyomo.contrib.appsi.base import (
    Solver,
    Results,
    TerminationCondition,
    MIPSolverConfig,
    SolutionLoader,
)
from pyomo.common.log import LogStream
import logging
import subprocess
from pyomo.core.kernel.objective import minimize
import math
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.visitor import replace_expressions
from typing import Optional, Sequence, NoReturn, List, Mapping
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.tee import TeeStream
import sys
from typing import Dict
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.errors import PyomoException
import os
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.plugins.nl_writer import NLWriter
from pyomo.core.base.symbol_map import SymbolMap


logger = logging.getLogger(__name__)


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

        self.declare("executable", ConfigValue())
        self.declare("filename", ConfigValue(domain=str))
        self.declare("keepfiles", ConfigValue(domain=bool))
        self.declare("solver_output_logger", ConfigValue())
        self.declare("log_level", ConfigValue(domain=NonNegativeInt))

        self.executable = Executable("scipampl")
        self.filename = None
        self.keepfiles = False
        self.solver_output_logger = logger
        self.log_level = logging.INFO


class ScipResults(Results):
    def __init__(self):
        super().__init__()
        self.nl_filename = None
        self.sol_filename = None
        self.options_filename = None

    def __str__(self):
        s = super().__str__()
        if self.nl_filename is not None:
            s += f"nl_filename: {str(self.nl_filename)}\n"
        if self.sol_filename is not None:
            s += f"sol_filename: {str(self.sol_filename)}\n"
        if self.options_filename is not None:
            s += f"options_filename: {str(self.sol_filename)}\n"


class Scip(Solver):
    def __init__(self):
        self._config = ScipConfig()
        self._solver_options = dict()
        self._writer = NLWriter()
        self._filename = None
        self._dual_sol = dict()
        self._primal_sol = ComponentMap()
        self._reduced_costs = ComponentMap()

        self._writer.config.skip_trivial_constraints = True

    def available(self):
        if self.config.executable.path() is None:
            return self.Availability.NotFound
        return self.Availability.Available

    def version(self):
        results = subprocess.run(
            [str(self.config.executable), "--version"],
            timeout=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        version = results.stdout.splitlines()[0]
        version = version.split(" ")[2]
        version = version.strip()
        version = tuple(int(i) for i in version.split("."))
        return version

    def _nl_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + ".nl"

    def _sol_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + ".sol"

    def _options_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + ".set"

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val):
        self._config = val

    @property
    def scip_options(self):
        """
        Returns
        -------
        ipopt_options: dict
            A dictionary mapping solver options to values for those options. These
            are solver specific.
        """
        return self._solver_options

    @scip_options.setter
    def scip_options(self, val: Dict):
        self._solver_options = val

    def _write_options_file(self):
        scip_options = dict(self.scip_options)
        if self.config.time_limit is not None:
            scip_options["limits/time"] = self.config.time_limit
        if self.config.mip_gap is not None:
            scip_options["limits/gap"] = self.config.mip_gap

        f = open(self._options_filename(), "w")
        for k, val in scip_options.items():
            f.write(str(k) + " = " + str(val) + "\n")
        f.close()

    def solve(self, model, timer: HierarchicalTimer = None) -> ScipResults:
        StaleFlagManager.mark_all_as_stale()
        avail = self.available()
        if not avail:
            raise PyomoException(f"Solver {self.__class__} is not available ({avail}).")
        if timer is None:
            timer = HierarchicalTimer()
        try:
            TempfileManager.push()
            if self.config.filename is None:
                nl_filename = TempfileManager.create_tempfile(suffix=".nl")
                self._filename = nl_filename.split(".")[0]
            else:
                self._filename = self.config.filename
                TempfileManager.add_tempfile(self._nl_filename(), exists=False)
            TempfileManager.add_tempfile(self._sol_filename(), exists=False)
            TempfileManager.add_tempfile(self._options_filename(), exists=False)
            self._write_options_file()
            timer.start("write nl file")
            ostream = open(self._nl_filename(), "w")
            self._writer.config.symbolic_solver_labels = (
                self.config.symbolic_solver_labels
            )
            symbol_map, amplfunc_libraries = self._writer.write(model, ostream=ostream)
            ostream.close()
            timer.stop("write nl file")
            return_code = self._apply_solver(timer)
            res = self._get_results(return_code, symbol_map, timer)
            if self.config.report_timing:
                logger.info("\n" + str(timer))
            return res
        finally:
            # finally, clean any temporary files registered with the
            # temp file manager, created/populated *directly* by this
            # plugin.
            TempfileManager.pop(remove=not self.config.keepfiles)
            if not self.config.keepfiles:
                self._filename = None

    def _get_results(
        self, return_code, symbol_map: SymbolMap, timer: HierarchicalTimer
    ) -> ScipResults:
        if return_code != 0:
            if self.config.load_solution:
                raise RuntimeError(
                    "A feasible solution was not found, so no solution can be loaded."
                    "Please set opt.config.load_solution=False and check "
                    "results.termination_condition and "
                    "results.best_feasible_objective before loading a solution."
                )
            results = ScipResults()
            results.termination_condition = TerminationCondition.error
            results.best_feasible_objective = None
            self._primal_sol = None
            self._dual_sol = None
            self._reduced_costs = None
        else:
            timer.start("parse solution")
            results = self._parse_sol(symbol_map)
            timer.stop("parse solution")

        if self._writer.get_active_objective() is None:
            results.best_objective_bound = None
        else:
            if self._writer.get_active_objective().sense == minimize:
                results.best_objective_bound = -math.inf
            else:
                results.best_objective_bound = math.inf

        return results

    def _parse_sol(self, symbol_map: SymbolMap) -> ScipResults:
        results = ScipResults()

        f = open(self._sol_filename(), "r")
        all_lines = list(f.readlines())
        f.close()

        termination_line = all_lines[1].lower()
        if "optimal solution found" in termination_line:
            results.termination_condition = TerminationCondition.optimal
        elif "problem may be infeasible" in termination_line:
            results.termination_condition = TerminationCondition.infeasible
        elif "problem might be unbounded" in termination_line:
            results.termination_condition = TerminationCondition.unbounded
        elif "maximum number of iterations exceeded" in termination_line:
            results.termination_condition = TerminationCondition.maxIterations
        elif "maximum cpu time exceeded" in termination_line:
            results.termination_condition = TerminationCondition.maxTimeLimit
        else:
            results.termination_condition = TerminationCondition.unknown

        n_cons = len(solve_cons)
        n_vars = len(solve_vars)
        dual_lines = all_lines[12 : 12 + n_cons]
        primal_lines = all_lines[12 + n_cons : 12 + n_cons + n_vars]

        rc_upper_info_line = all_lines[12 + n_cons + n_vars + 1]
        assert rc_upper_info_line.startswith("suffix")
        n_rc_upper = int(rc_upper_info_line.split()[2])
        assert "ipopt_zU_out" in all_lines[12 + n_cons + n_vars + 2]
        upper_rc_lines = all_lines[
            12 + n_cons + n_vars + 3 : 12 + n_cons + n_vars + 3 + n_rc_upper
        ]

        rc_lower_info_line = all_lines[12 + n_cons + n_vars + 3 + n_rc_upper]
        assert rc_lower_info_line.startswith("suffix")
        n_rc_lower = int(rc_lower_info_line.split()[2])
        assert "ipopt_zL_out" in all_lines[12 + n_cons + n_vars + 3 + n_rc_upper + 1]
        lower_rc_lines = all_lines[
            12
            + n_cons
            + n_vars
            + 3
            + n_rc_upper
            + 2 : 12
            + n_cons
            + n_vars
            + 3
            + n_rc_upper
            + 2
            + n_rc_lower
        ]

        self._dual_sol = dict()
        self._primal_sol = ComponentMap()
        self._reduced_costs = ComponentMap()

        for ndx, dual in enumerate(dual_lines):
            dual = float(dual)
            con = solve_cons[ndx]
            self._dual_sol[con] = dual

        for ndx, primal in enumerate(primal_lines):
            primal = float(primal)
            var = solve_vars[ndx]
            self._primal_sol[var] = primal

        for rcu_line in upper_rc_lines:
            split_line = rcu_line.split()
            var_ndx = int(split_line[0])
            rcu = float(split_line[1])
            var = solve_vars[var_ndx]
            self._reduced_costs[var] = rcu

        for rcl_line in lower_rc_lines:
            split_line = rcl_line.split()
            var_ndx = int(split_line[0])
            rcl = float(split_line[1])
            var = solve_vars[var_ndx]
            if var in self._reduced_costs:
                if abs(rcl) > abs(self._reduced_costs[var]):
                    self._reduced_costs[var] = rcl
            else:
                self._reduced_costs[var] = rcl

        for var in solve_vars:
            if var not in self._reduced_costs:
                self._reduced_costs[var] = 0

        if (
            results.termination_condition == TerminationCondition.optimal
            and self.config.load_solution
        ):
            for v, val in self._primal_sol.items():
                v.set_value(val, skip_validation=True)

            if self._writer.get_active_objective() is None:
                results.best_feasible_objective = None
            else:
                results.best_feasible_objective = value(
                    self._writer.get_active_objective().expr
                )
        elif results.termination_condition == TerminationCondition.optimal:
            if self._writer.get_active_objective() is None:
                results.best_feasible_objective = None
            else:
                obj_expr_evaluated = replace_expressions(
                    self._writer.get_active_objective().expr,
                    substitution_map={
                        id(v): val for v, val in self._primal_sol.items()
                    },
                    descend_into_named_expressions=True,
                    remove_named_expressions=True,
                )
                results.best_feasible_objective = value(obj_expr_evaluated)
        elif self.config.load_solution:
            raise RuntimeError(
                "A feasible solution was not found, so no solution can be loaded."
                "Please set opt.config.load_solution=False and check "
                "results.termination_condition and "
                "resutls.best_feasible_objective before loading a solution."
            )

        results.solution_loader = PersistentSolutionLoader(solver=self)

        return results

    def _apply_solver(self, timer: HierarchicalTimer):
        config = self.config

        if config.time_limit is not None:
            timeout = config.time_limit + min(max(1.0, 0.01 * config.time_limit), 100)
        else:
            timeout = None

        ostreams = [
            LogStream(
                level=self.config.log_level, logger=self.config.solver_output_logger
            )
        ]
        if self.config.stream_solver:
            ostreams.append(sys.stdout)

        cmd = [
            str(config.executable),
            self._nl_filename(),
            "-AMPL",
            self._options_filename(),
        ]

        env = os.environ.copy()
        if "PYOMO_AMPLFUNC" in env:
            env["AMPLFUNC"] = "\n".join(
                filter(
                    None, (env.get("AMPLFUNC", None), env.get("PYOMO_AMPLFUNC", None))
                )
            )

        with TeeStream(*ostreams) as t:
            timer.start("subprocess")
            cp = subprocess.run(
                cmd,
                timeout=timeout,
                stdout=t.STDOUT,
                stderr=t.STDERR,
                env=env,
                universal_newlines=True,
            )
            timer.stop("subprocess")

        return cp.returncode

    def get_primals(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        res = ComponentMap()
        if vars_to_load is None:
            for v, val in self._primal_sol.items():
                res[v] = val
        else:
            for v in vars_to_load:
                res[v] = self._primal_sol[v]
        return res

    def get_duals(self, cons_to_load=None):
        if cons_to_load is None:
            return {k: v for k, v in self._dual_sol.items()}
        else:
            return {c: self._dual_sol[c] for c in cons_to_load}

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        if vars_to_load is None:
            return ComponentMap((k, v) for k, v in self._reduced_costs.items())
        else:
            return ComponentMap((v, self._reduced_costs[v]) for v in vars_to_load)
