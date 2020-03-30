#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import sys
from pyutilib.common import ApplicationError, WindowsError
from pyutilib.services import TempfileManager
from pyutilib.subprocess import run
from pyomo.common.config import ConfigValue, add_docstring_list, Path, NonNegativeFloat
from pyomo.common.fileutils import Executable, this_file_dir
from pyomo.common.timing import TicTocTimer
from pyomo.solver.base import MIPSolver, Results, TerminationCondition, SolutionLoader
# from pyomo.writer.cpxlp import ProblemWriter_cpxlp
from pyomo.repn.plugins.cpxlp import ProblemWriter_cpxlp
from pyomo.opt.base.solvers import SolverFactory
from pyomo.core.kernel.component_map import ComponentMap
import subprocess
import io
import logging
from pyomo.core.base.suffix import active_import_suffix_generator
try:
    import cPickle as pickle
except ImportError:
    import pickle


logger = logging.getLogger(__name__)


@SolverFactory.register('NEW_gurobi', doc='An interface to Gurobi')
class GurobiSolver(MIPSolver):
    CONFIG = MIPSolver.CONFIG()

    def __new__(cls, **kwds):
        if cls != GurobiSolver:
            return super(GurobiSolver, cls).__new__(cls, **kwds)

        solver_io = kwds.pop('solver_io', 'lp')
        if solver_io == 'lp':
            return GurobiSolver_LP(**kwds)
        elif solver_io == 'nl':
            raise NotImplementedError('nl interface is not supported yet.')
        elif solver_io == 'mps':
            raise NotImplementedError('mps interface is not supported yet.')
        # For the direct / persistent solvers, they are implemented in
        # other modules.  To simplify imports, we will defer to the
        # SolverFactory
        elif solver_io == 'persistent':
            return SolverFactory('gurobi_persistent_new', **kwds)
        elif solver_io in ('direct', 'python'):
            return SolverFactory('gurobi_persistent_new', **kwds)
        else:
            raise ValueError("Invalid solver_io for GurobiSolver: %s"
                             % (solver_io,))


class GurobiSolver_LP(GurobiSolver):
    CONFIG = GurobiSolver.CONFIG()
    CONFIG.declare("executable", ConfigValue(
        default='gurobi.bat' if sys.platform == 'win32' else 'gurobi.sh' ,
        domain=Executable,
    ))
    CONFIG.declare("problemfile", ConfigValue(
        default=None,
        domain=Path(),
    ))
    CONFIG.declare("logfile", ConfigValue(
        default=None,
        domain=Path(),
    ))
    CONFIG.declare("solnfile", ConfigValue(
        default=None,
        domain=Path(),
    ))
    CONFIG.declare("warmstart_file", ConfigValue(
        default=None,
        domain=Path(),
    ))

    # CONFIG.declare_from(ProblemWriter_cpxlp.CONFIG, skip={
    #     'allow_quadratic_objective', 'allow_quadratic_constraints',
    #     'allow_sos1', 'allow_sos2'})

    def available(self):
        return self.config.executable.path() is not None

    def license_status(self):
        """
        Runs a check for a valid Gurobi license using the
        given executable (default is 'gurobi_cl'). All
        output is hidden. If the test fails for any reason
        (including the executable being invalid), then this
        function will return False.
        """
        if not self.available():
            return False
        gurobi_cl = os.path.join(os.path.dirname(str(self.config.executable)),
                                 'gurobi_cl')
        try:
            rc = subprocess.call([gurobi_cl, "--license"],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
        except OSError:
            rc = 1
        return rc == 0

    def version(self):
        assert self.available()
        stream = io.StringIO()
        rc = run([str(self.config.executable)],
                 stdin=('from gurobipy import GRB; '
                        'print(GRB.VERSION_MAJOR, GRB.VERSION_MINOR, GRB.VERSION_TECHNICAL); '
                        'exit()'),
                 ostream=stream)
        res = stream.getvalue().strip().replace('(', '').replace(',', '').replace(')', '').split()
        res = tuple(int(i) for i in res)
        return res

    def solve(self, model, options=None, **config_options):
        """Solve a model""" + add_docstring_list("", GurobiSolver_LP.CONFIG)

        options = self.options(options)
        config = self.config(config_options)

        try:
            TempfileManager.push()
            return self._apply_solver(model, options, config)
        finally:
            # finally, clean any temporary files registered with the
            # temp file manager, created populated *directly* by this
            # plugin.
            TempfileManager.pop(remove=not config.keepfiles)

    def _apply_solver(self, model, options, config):
        T = TicTocTimer()
        if not config.problemfile:
            config.problemfile = TempfileManager.create_tempfile(
                suffix='.pyomo.lp')
        if not config.logfile:
            config.logfile = TempfileManager.create_tempfile(
                suffix='.gurobi.log')
        if not config.solnfile:
            config.solnfile = TempfileManager.create_tempfile(
                suffix='.gurobi.txt')
        print(config.problemfile)

        # Gurobi can support certain problem features
        # writer_config = ProblemWriter_cpxlp.CONFIG()
        # writer_config.allow_quadratic_objective = True
        # writer_config.allow_quadratic_constraints = True
        # writer_config.allow_sos1 = True
        # writer_config.allow_sos2 = True
        # Copy over the relevant values from the solver config
        # (skip_implicit alloes the config to have additional fields
        # that are ignored)
        # writer_config.set_value(config, skip_implicit=True)
        T.toc("gurobi setup complete")
        writer = ProblemWriter_cpxlp()
        fname, symbol_map = writer(model=model,
                                   output_filename=config.problemfile,
                                   solver_capability=lambda x: True,
                                   io_options=dict())
        assert fname == str(config.problemfile)
        T.toc("gurobi lp write complete")

        # Handle mapped options
        mipgap = config.mip_gap
        if mipgap is not None:
            options['MIPGap'] = mipgap
        options['LogFile'] = config.logfile

        # Extract the suffixes
        suffixes = list(name for name, comp in active_import_suffix_generator(model))

        # Run Gurobi
        data = pickle.dumps(
            (config.problemfile,
             config.solnfile,
             {'warmstart_file': config.warmstart_file,
              'relax_integrality': config.relax_integrality,},
              options.value(),
              suffixes), protocol=2)
        timelim = config.time_limit
        if timelim:
            timelim + min(max(1, 0.01*self._timelim), 100)
        cmd = [ str(config.executable),
                os.path.join(this_file_dir(), 'GUROBI_RUN.py')]
        try:
            T.toc("gurobi other preminaries done")
            rc, log = run(cmd, stdin=data, timelimit=timelim, tee=config.tee)
            T.toc("gurobi subprocess complete")
        except WindowsError:
            raise ApplicationError(
                'Could not execute the command: %s\tError message: %s'
                % (cmd, sys.exc_info()[1]))
        sys.stdout.flush()

        # Read in the results
        result_data = None
        with open(config.solnfile, 'rb') as SOLN:
            try:
                result_data = pickle.load(SOLN)
            except ValueError:
                logger.error(
                    "GurobiSolver_LP: no results data returned from the "
                    "Gurobi subprocess.  Look at the solver log for more "
                    "details (re-run with 'tee=True' to see the solver log.")
                raise

        results = Results(found_feasible_solution=result_data['found_feasible_solution'])
        results.solver.declare('wallclock_time', ConfigValue(default=None, domain=NonNegativeFloat, doc='The wallclock time reported by Gurobi'))
        results.solver.wallclock_time = result_data['solver']['wallclock_time']
        results.solver.termination_condition = TerminationCondition[result_data['solver']['termination_condition']]
        results.solver.best_feasible_objective = result_data['solver']['best_feasible_objective']
        results.solver.best_objective_bound = result_data['solver']['best_objective_bound']

        primals = ComponentMap()
        duals = dict()
        slacks = dict()
        rc = ComponentMap()
        if results.found_feasible_solution():
            _sol = result_data['solutions'][0]
            X = _sol['X']
            for ndx, vname in enumerate(_sol['VarName']):
                v = symbol_map.getObject(vname)
                primals[v] = X[ndx]
            if 'Rc' in _sol:
                Rc = _sol['Rc']
                for ndx, vname in enumerate(_sol['VarName']):
                    v = symbol_map.getObject(vname)
                    rc[v] = Rc[ndx]
            if 'Pi' in _sol:
                Pi = _sol['Pi']
                for ndx, cname in enumerate(_sol['ConstrName']):
                    if cname in symbol_map.bySymbol or cname in symbol_map.aliases:
                        c = symbol_map.getObject(cname)
                        duals[c] = Pi[ndx]
            if 'QCPi' in _sol:
                QCPi = _sol['QCPi']
                for ndx, cname in enumerate(_sol['QCName']):
                    if cname in symbol_map.bySymbol or cname in symbol_map.aliases:
                        c = symbol_map.getObject(cname)
                        duals[c] = QCPi[ndx]
            if 'Slack' in _sol:
                Slack = _sol['Slack']
                for ndx, cname in enumerate(_sol['ConstrName']):
                    if cname in symbol_map.bySymbol or cname in symbol_map.aliases:
                        c = symbol_map.getObject(cname)
                        if c in slacks:
                            if abs(Slack[ndx]) > abs(slacks[c]):
                                slacks[c] = Slack[ndx]
                        else:
                            slacks[c] = Slack[ndx]
            if 'QCSlack' in _sol:
                QCSlack = _sol['QCSlack']
                for ndx, cname in enumerate(_sol['QCName']):
                    if cname in symbol_map.bySymbol or cname in symbol_map.aliases:
                        c = symbol_map.getObject(cname)
                        if c in slacks:
                            if abs(QCSlack[ndx]) > abs(slacks[c]):
                                slacks[c] = QCSlack[ndx]
                        else:
                            slacks[c] = QCSlack[ndx]

        solution_loader = SolutionLoader(model=model, primals=primals, duals=duals, slacks=slacks, reduced_costs=rc)
        results.solution_loader = solution_loader

        if config.load_solution:
            solution_loader.load_solution()

        T.toc("gurobi solution load complete")
        return results
