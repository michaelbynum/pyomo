#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat, In
from pyomo.common.errors import DeveloperError
from pyomo.common.deprecation import deprecated
from pyomo.core.base.suffix import Suffix
import six
import abc
import enum


class TerminationCondition(enum.Enum):
    """
    An enumeration for checking the termination condition of solvers
    """
    unknown = 0
    """unknown serves as both a default value, and it is used when no other enum member makes sense"""

    maxTimeLimit = 1
    """The solver exited due to a time limit"""

    maxIterations = 2
    """The solver exited due to an iteration limit """

    objectiveLimit = 3
    """The solver exited due to an objective limit"""

    minStepLength = 4
    """The solver exited due to a minimum step length"""

    optimal = 5
    """The solver exited with the optimal solution"""

    unbounded = 8
    """The solver exited because the problem is unbounded"""

    infeasible = 9
    """The solver exited because the problem is infeasible"""

    infeasibleOrUnbounded = 10
    """The solver exited because the problem is either infeasible or unbounded"""

    error = 11
    """The solver exited due to an error"""

    interrupted = 12
    """The solver exited because it was interrupted"""

    licensingProblems = 13
    """The solver exited due to licensing problems"""


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

    def is_persistent(self):
        return False

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


class SolutionLoaderBase(six.with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def load_solution(self):
        """
        Load the solution into the model. This will load the values of the primal variables into
        the value attribute of the variables, the duals into the model.dual suffix if it exists,
        the slacks into the model.slack suffix if it exists, and reduced costs into the
        model.rc suffix if it exists.
        """
        pass

    @abc.abstractmethod
    def load_suffix(self, suffix):
        """
        Load the specified suffix into the model.

        Parameters
        ----------
        suffix: str
            The suffix to load. Options typically include 'dual', 'slack', and 'rc', but this is solver-dependent.
            Please see the documentation for the solver interface of interest.
        """

    @abc.abstractmethod
    def load_vars(self, vars_to_load=None):
        """
        Load the solution of the primal variables into the value attribut of the variables.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose solution should be loaded. If vars_to_load is None, then the solution
            to all primal variables will be loaded.
        """
        pass

    def load_duals(self, cons_to_load=None):
        """
        Load the duals into the model.dual suffix. If the model.dual suffix does not exist it will be created.

        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose duals should be loaded. If cons_to_load is None, then the duals for all
            constraints will be loaded.
        """
        raise NotImplementedError('{0} does not support the load_duals method'.format(type(self)))

    def load_slacks(self, cons_to_load=None):
        """
        Load the slacks into the model.slack suffix. If the model.slack suffix does not exist it will be created.

        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose slacks should be loaded. If cons_to_load is None, then the slacks for all
            constraints will be loaded.
        """
        raise NotImplementedError('{0} does not support the load_slacks method'.format(type(self)))

    def load_reduced_costs(self, vars_to_load=None):
        """
        Load the reduced costs into the model.rc suffix. If the model.rc suffix does not exist it will be created.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose reduced cost should be loaded. If vars_to_load is None, then all reduced costs
            will be loaded.
        """
        raise NotImplementedError('{0} does not support the load_reduced_costs method'.format(type(self)))


class SolutionLoader(SolutionLoaderBase):
    def __init__(self, model, primals, duals, slacks, reduced_costs):
        """
        Parameters
        ----------
        model: the pyomo model
        primals: ComponentMap
            maps Var to value
        duals: ComponentMap
            maps Constraint to dual value
        slacks: ComponentMap
            maps Constraint to slack value
        reduced_costs: ComponentMap
            maps Var to reduced cost
        """
        self._model = model
        self._primals = primals
        self._duals = duals
        self._slacks = slacks
        self._reduced_costs = reduced_costs

    def load_solution(self):
        for v, val in self._primals.items():
            v.value = val

        if hasattr(self._model, 'dual'):
            for c, val in self._duals.items():
                self._model.dual[c] = val

        if hasattr(self._model, 'slack'):
            for c, val in self._slacks.items():
                self._model.slack[c] = val

        if hasattr(self._model, 'rc'):
            for v, val in self._reduced_costs.items():
                self._model.rc[v] = val

    def load_suffix(self, suffix):
        if suffix == 'dual':
            self.load_duals()
        elif suffix == 'slack':
            self.load_slacks()
        elif suffix == 'rc':
            self.load_reduced_costs()
        else:
            raise ValueError('suffix not recognized')

    def load_vars(self, vars_to_load=None):
        if vars_to_load is None:
            for v, val in self._primals.items():
                v.value = val
        else:
            for v in vars_to_load:
                v.value = self._primals[v]

    def load_duals(self, cons_to_load=None):
        if not hasattr(self._model, 'dual'):
            self._model.dual = Suffix(direction=Suffix.IMPORT)

        if cons_to_load is None:
            for c, val in self._duals.items():
                self._model.dual[c] = val
        else:
            for c in cons_to_load:
                self._model.dual[c] = self._duals[c]

    def load_slacks(self, cons_to_load=None):
        if not hasattr(self._model, 'slack'):
            self._model.slack = Suffix(direction=Suffix.IMPORT)

        if cons_to_load is None:
            for c, val in self._slacks.items():
                self._model.slack[c] = val
        else:
            for c in cons_to_load:
                self._model.slack[c] = self._slacks[c]

    def load_reduced_costs(self, vars_to_load=None):
        if not hasattr(self._model, 'rc'):
            self._model.rc = Suffix(direction=Suffix.IMPORT)

        if vars_to_load is None:
            for v, val in self._reduced_costs.items():
                self._model.rc[v] = val
        else:
            for v in vars_to_load:
                self._model.rc[v] = self._reduced_costs[v]


class ResultsBase(six.with_metaclass(abc.ABCMeta, object)):
    """
    The results object has three primary roles:

    1. Report information from the solver. This is done through results.solver, which stores three or more attributes
    with information from the solver. At a minimum, results.solver must have a termination_condition,
    best_feasible_objective, and best_objective_bound.

    2. The results object has a method called found_feasible_solution, which returns True if at least one
    feasible solution was found and False otherwise.

    3. The results object has an attribute called solution_loader which should be an instance of SolutionLoader. The
    SolutionLoader has a method called load_solution which handles loading solutions into the model.

    Here is an example workflow:
    >>> import pyomo.environ as pe
    >>> m = pe.ConcreteModel()
    >>> m.x = pe.Var()
    >>> m.obj = pe.Objective(expr=m.x**2)
    >>> opt = pe.SolverFactory('my_solver')
    >>> results = opt.solve(m, load_solution=False)
    >>> if results.solver.termination_condition == TerminationCondition.optimal:
    >>>     print('optimal solution found: ', results.solver.best_feasible_objective)
    >>>     results.solution_loader.load_solution()
    >>>     print('the optimal value of x is ', m.x.value)
    >>> elif results.found_feasible_solution():
    >>>     print('sub-optimal but feasible solution found: ', results.solver.best_feasible_objective)
    >>>     results.solution_loader.load_vars(vars_to_load=[m.x])
    >>>     print('The value of x in the feasible solution is ', m.x.value)
    >>> elif results.solver.termination_condition in {TerminationCondition.maxIterations,
    ...                                               TerminationCondition.maxTimeLimit}:
    >>>     print('No feasible solution was found. The best lower bound found was ',
    ...           results.solver.best_objective_bound)
    >>> else:
    >>>     print('The following termination condition was encountered: ',
    ...           results.solver.termination_condition)
    """
    def __init__(self):
        self.solution_loader = None
        self.solver = ConfigBlock()
        self.solver.declare('termination_condition',
                            ConfigValue(default=TerminationCondition.unknown,
                                        domain=In(TerminationCondition),
                                        doc="The reason the solver exited. This is a member of the "
                                            "TerminationCondition enum."))
        self.solver.declare('best_feasible_objective',
                            ConfigValue(default=None,
                                        domain=float,
                                        doc="If a feasible solution was found, this is the objective value of "
                                            "the best solution found. If no feasible solution was found, this is"
                                            "None."))
        self.solver.declare('best_objective_bound',
                            ConfigValue(default=None,
                                        domain=float,
                                        doc="The best objective bound found. For minimization problems, this is "
                                            "the lower bound. For maximization problems, this is the upper bound."
                                            "For solvers that do not provide an objective bound, this should be -inf "
                                            "(minimization) or inf (maximization)"))
        self.problem = ConfigBlock()

    @abc.abstractmethod
    def found_feasible_solution(self):
        """
        Returns
        -------
        found_feasible_solution: bool
            True if at least one feasible solution was found. False otherwise.
        """
        pass


class Results(ResultsBase):
    def __init__(self, found_feasible_solution):
        super(Results, self).__init__()
        self._found_feasible_solution = found_feasible_solution

    def found_feasible_solution(self):
        return self._found_feasible_solution
