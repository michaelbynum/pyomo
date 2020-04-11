#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyutilib.services import TempfileManager
from pyomo.common.config import (ConfigBlock,
                                 ConfigList,
                                 ConfigValue,
                                 add_docstring_list,
                                 Path)
from pyomo.common.fileutils import Executable
from pyomo.solver.base import Solver
from pyomo.opt.base.solvers import SolverFactory
import logging
import io
import subprocess
from pyomo.repn.plugins.ampl.ampl_ import ProblemWriter_nl
from pyomo.core.base.suffix import active_import_suffix_generator


logger = logging.getLogger(__name__)


@SolverFactory.register('NEW_ipopt', doc='Interface to Ipopt')
class IpoptSolver(Solver):
    CONFIG = Solver.CONFIG()

    def __new__(cls, **kwargs):
        if cls != IpoptSolver:
            return super(IpoptSolver, cls).__new__(cls, **kwargs)

        solver_io = kwargs.pop('solver_io', 'nl')
        if solver_io == 'nl':
            return IpoptSolver_NL(**kwargs)
        else:
            raise ValueError("Invalid solver_io for IpoptSolver: %s"
                             % (solver_io,))


class IpoptSolver_NL(IpoptSolver):
    CONFIG = IpoptSolver.CONFIG()
    CONFIG.declare('executable', ConfigValue(
        default='ipopt',
        domain=Executable
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

    def available(self):
        return self.config.executable.path() is not None

    def license_status(self):
        """
        Ipopt does not require a license to run so this method is really just
        an alias of the available method.
        """
        return self.available()

    def version(self):
        assert self.available()
        cp = subprocess.run([str(self.config.executable), '--version'],
                            capture_output=True, text=True)
        version = cp.stdout.split()[1]
        version = version.split('.')
        version = tuple(int(i) for i in version)
        return version

    def solve(self, model, options=None, **config_options):
        """Solve a model""" + add_docstring_list("", IpoptSolver_NL.CONFIG)

        options = self.options(options)
        config = self.config(config_options)

        try:
            TempfileManager.push()
            return self._apply_solver(model, options, config)
        finally:
            # finally, clean any temporary files registered with the
            # temp file manager, created/populated *directly* by this
            # plugin.
            TempfileManager.pop(remove=not config.keepfiles)

    def _apply_solver(self, model, options, config):
        if not config.problemfile:
            config.problemfile = TempfileManager.create_tempfile(
                suffix='.pyomo.nl')
        if not config.logfile:
            config.logfile = TempfileManager.create_tempfile(
                suffix='.ipopt.log')
        if not config.solnfile:
            config.solnfile = TempfileManager.create_tempfile(
                suffix='.ipopt.sol')

        # Write out the model
        writer = ProblemWriter_nl()
        fname, symbol_map = writer(model=model,
                                   filename=config.problemfile,
                                   solver_capability=lambda x: True,
                                   io_options=dict())
        assert fname == str(config.problemfile)

        # Extract Suffixes
        suffixes = list(name for name, comp in active_import_suffix_generator(model))

        # Run Ipopt
        cp = subprocess.run([str(config.executable),
                             '-o'+config.solnfile,
                             config.problemfile],
                            capture_output=True,
                            text=True,
                            timeout=config.time_limit)

        data = json.dumps(
            (config.problemfile,
             config.solnfile,
             options,
             suffixes))
        timelim = config.timelimit
        if timelim:
            timelim + min(max(1, 0.01 * self._timelim), 100)
        cmd = [config.executable,
        os.path.join(this_file_dir(), '_RUN.py')]
        try:
            rc, log = run(cmd, stdin=data, timelimit=timelim, tee=config.tee)
        except WindowsError:
            raise ApplicationError(
                'Could not execute the command: %s\tError message: %s'
                % (cmd, sys.exc_info()[1]))
        sys.stdout.flush()

        # Read in the results
        result_data = None
        with open(config.solnfile, 'r') as SOLN:
            try:
                result_data = json.load(SOLN)
            except ValueError:
                logger.error(
                    "IpoptSolver_NL: no results data returned from the "
                    "Ipopt subprocess.  Look at the solver log for more "
                    "details (re-run with 'tee=True' to see the solver log.")
                raise

        results = SolverResults()
        results.problem.update(result_data['problem'])
        results.solver.update(result_data['solver'])
        results.solver.name = 'gurobi_lp'

        if not config.load_solution:
            raise RuntimeError("TODO")
        elif result_data['solution']['points']:
            _sol = result_data['solution']['points'][0]
            X = _sol['X']
            for i, vname in enumerate(_sol['VarName']):
                v = symbol_map.getObject(vname)
                v.value = X[i]

        return results


IpoptSolver_NL.solve.__doc__ = add_docstring_list(IpoptSolver_NL.solve.__doc__, IpoptSolver.CONFIG)

if __name__ == '__main__':
    # Debugging
    a = IpoptSolver_NL()
    help(a.solve)