#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.errors import DeveloperError
from pyomo.common.deprecation import deprecated
import six
import abc
import enum

"""
An enumeration for checking the termination condition of solvers

Attributes:
    unknown    
"""


class TerminationCondition(enum.Enum):
    """
    An enumeration for checking the termination condition of solvers
    """
    unknown = 0
    maxTimeLimit = 1
    maxIterations = 2
    minFunctionValue = 3
    minStepLength = 4
    optimal = 5
    feasible = 6
    maxEvaluations = 7
    unbounded = 8
    infeasible = 9
    infeasibleOrUnbounded = 10
    error = 11
    Interrupt = 12
    licensingProblems = 13


class Solver(object):
    """A generic optimization solver"""

    CONFIG = ConfigBlock()
    CONFIG.declare('timelimit', ConfigValue(
        default=None,
        domain=NonNegativeFloat,
    ))
    CONFIG.declare('keepfiles', ConfigValue(
        default=False,
        domain=bool,
    ))
    CONFIG.declare('tee', ConfigValue(
        default=False,
        domain=bool,
    ))
    CONFIG.declare('load_solution', ConfigValue(
        default=True,
        domain=bool,
    ))


    def __init__(self, **kwds):
        self.config = self.CONFIG()
        self.options = ConfigBlock(implicit=True)

    def available(self):
        raise DeveloperError(
            "Derived Solver class %s failed to implement available()"
            % (self.__class__.__name__,))

    def license_status(self):
        raise DeveloperError(
            "Derived Solver class %s failed to implement license_status()"
            % (self.__class__.__name__,))

    def version(self):
        """
        Returns a tuple describing the solver version.
        """
        raise DeveloperError(
            "Derived Solver class %s failed to implement version()"
            % (self.__class__.__name__,))

    def solve(self, model, options=None, **config_options):
        raise DeveloperError(
            "Derived Solver class %s failed to implement solve()"
            % (self.__class__.__name__,))

    @deprecated("Casting a solver to bool() is deprecated.  Use available()",
                version='TBD')
    def __bool__(self):
        return self.available()

    ####################################################################
    #  The following are "hacks" to support the pyomo command (and
    #  compatability with existing solvers)

    #
    # Support "with" statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass


class MIPSolver(Solver):
    CONFIG = Solver.CONFIG()
    CONFIG.declare('mipgap', ConfigValue(
        default=None,
        domain=NonNegativeFloat,
    ))
    CONFIG.declare('relax_integrality', ConfigValue(
        default=False,
        domain=bool,
    ))


class SolutionLoader(six.with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def load_solution(self, model):
        pass


class ResultsBase(object):
    def __init__(self):
        self._optimal = False
        self.solution_loader = None
        self.solver = ConfigBlock()
        self.solver.declare('termination_condition',
                            ConfigValue(default=TerminationCondition.unknown))
        self.solver_info.declare('total_wallclock_time',
                                 ConfigValue(default=None,
                                             domain=NonNegativeFloat,
                                             doc="Total wallclock time spent in the call to solve, including time "
                                                 "required by Pyomo to process the model and the solution"))
        self.solver_info.declare('solver_wallclock_time',
                                 ConfigValue(default=None,
                                             domain=NonNegativeFloat,
                                             doc="Wallclock time required by the solver"))
        self.solver_info.declare('termination_condition')

    @property
    def optimal(self):
        return self._optimal

    @optimal.setter
    def optimal(self, val):
        assert val in {True, False}
        self._optimal = val
