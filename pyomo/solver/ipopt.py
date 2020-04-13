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
from pyomo.solver.tee_thread import TeeThread
from pyomo.opt.plugins.sol import ResultsReader_sol


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
    """
    NL interface to Ipopt.
    """
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

    __doc__ = add_docstring_list(__doc__, CONFIG)

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
        """
        Solve a model
        """

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
        if config.tee and config.logfile is not None:
            out = open(config.logfile, 'wb')
            err = subprocess.STDOUT
            thread = TeeThread(config.logfile)
            thread.start()
            capture_output = False
        elif config.tee:
            out = None
            err = None
            capture_output = False
        elif config.logfile is not None:
            out = open(config.logfile, 'wb')
            err = subprocess.STDOUT
            capture_output = False
        else:
            capture_output = True
        if config.time_limit is not None:
            timeout = config.time_limit + min(max(1, 0.01 * config.time_limit), 100)
        else:
            timeout = None
        cp = subprocess.run([str(config.executable),
                             '-o'+config.solnfile,
                             config.problemfile],
                            timeout=timeout,
                            stdout=out,
                            stderr=err,
                            capture_output=capture_output)

        if config.tee and config.logfile is not None:
            thread.event.set()
            thread.join()
            out.close()
        elif config.tee:
            pass
        elif config.logfile is not None:
            out.close()
        else:
            pass

        cp.check_returncode()

        sol_reader = ResultsReader_sol()
        results = sol_reader(config.solnfile, suffixes=suffixes)

        return results


IpoptSolver_NL.solve.__doc__ = add_docstring_list(IpoptSolver_NL.solve.__doc__, IpoptSolver_NL.CONFIG)
