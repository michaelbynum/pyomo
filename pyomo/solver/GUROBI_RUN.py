#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""This script is run using the Gurobi/system python. Do not assume any
third party packages are available!

"""
import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle

from gurobipy import *

if sys.version_info[0] < 3:
    from itertools import izip as zip

GUROBI_VERSION = gurobi.version()
NUM_SOLNS = 1

# NOTE: this function / module is independent of Pyomo, and only relies
#       on the GUROBI python bindings. consequently, nothing in this
#       function should throw an exception that is expected to be
#       handled by Pyomo - it won't be.  rather, print an error message
#       and return - the caller will know to look in the logs in case of
#       a failure.

def _is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


def gurobi_run(model_file, pyomo_options, options, suffixes):
    # figure out what suffixes we need to extract.
    extract_duals = False
    extract_slacks = False
    extract_reduced_costs = False
    for suffix in suffixes:
        if "dual" == suffix:
            extract_duals = True
        elif "slack" == suffix:
            extract_slacks = True
        elif "rc" == suffix:
            extract_reduced_costs = True
        else:
            print("***The GUROBI solver plugin cannot extract solution suffix="
                  + suffix)
            return

    # Load the lp model
    model = read(model_file)

    # if the use wants to extract duals or reduced costs and the
    # model has quadratic constraints then we need to set the
    # QCPDual param to 1 (which apparently makes the solve more
    # expensive in the quadratic case). If we do not set this param
    # and and we attempt to access these suffixes in the solution
    # printing the module will crash (when we have a QCP)
    if GUROBI_VERSION[0] >= 5:
        if (extract_reduced_costs is True) or (extract_duals is True):
            model.setParam(GRB.Param.QCPDual,1)

    if model is None:
        print("***The GUROBI solver plugin failed to load the input LP file="
              + model_file)
        return


    warmstart_file = pyomo_options.pop('warmstart_file', None)
    if warmstart_file:
        model.read(warmstart_file)

    if pyomo_options.pop('relax_integrality', False):
        for v in model.getVars():
            if v.vType != GRB.CONTINUOUS:
                v.vType = GRB.CONTINUOUS
        model.update()

    if pyomo_options:
        print("***The GUROBI solver plugin does not understand the "
              "following Pyomo options:\n\t"
              + "\n\t".join("%s: %s" % _ for _ in pyomo_options.items()))
        return

    # set all other solver parameters, if specified.
    # GUROBI doesn't throw an exception if an unknown
    # key is specified, so you have to stare at the
    # output to see if it was accepted.
    for key, value in options.items():
        # When options come from the pyomo command, all
        # values are string types, so we try to cast
        # them to a numeric value in the event that
        # setting the parameter fails.
        try:
            model.setParam(key, value)
        except TypeError:
            # we place the exception handling for checking
            # the cast of value to a float in another
            # function so that we can simply call raise here
            # instead of except TypeError as e / raise e,
            # because the latter does not preserve the
            # Gurobi stack trace
            if not _is_numeric(value):
                raise
            model.setParam(key, float(value))


    # optimize the model
    model.optimize()

    results = dict()
    results['solver'] = dict()
    results['solver']['wallclock_time'] = model.Runtime
    grb = GRB
    status = model.Status

    if status == grb.LOADED:  # problem is loaded, but no solution
        results['solver']['termination_condition'] = 'unknown'
    elif status == grb.OPTIMAL:  # optimal
        results['solver']['termination_condition'] = 'optimal'
    elif status == grb.INFEASIBLE:
        results['solver']['termination_condition'] = 'infeasible'
    elif status == grb.INF_OR_UNBD:
        results['solver']['termination_condition'] = 'infeasibleOrUnbounded'
    elif status == grb.UNBOUNDED:
        results['solver']['termination_condition'] = 'unbounded'
    elif status == grb.CUTOFF:
        results['solver']['termination_condition'] = 'objectiveLimit'
    elif status == grb.ITERATION_LIMIT:
        results['solver']['termination_condition'] = 'maxIterations'
    elif status == grb.NODE_LIMIT:
        results['solver']['termination_condition'] = 'maxIterations'
    elif status == grb.TIME_LIMIT:
        results['solver']['termination_condition'] = 'maxTimeLimit'
    elif status == grb.SOLUTION_LIMIT:
        results['solver']['termination_condition'] = 'unknown'
    elif status == grb.INTERRUPTED:
        results['solver']['termination_condition'] = 'interrupted'
    elif status == grb.NUMERIC:
        results['solver']['termination_condition'] = 'unknown'
    elif status == grb.SUBOPTIMAL:
        results['solver']['termination_condition'] = 'unknown'
    elif status == grb.USER_OBJ_LIMIT:
        results['solver']['termination_condition'] = 'objectiveLimit'
    else:
        results['solver']['termination_condition'] = 'unknown'

    try:
        results['solver']['best_feasible_objective'] = model.ObjVal
    except (GurobiError, AttributeError):
        results['solver']['best_feasible_objective'] = None
    try:
        results['solver']['best_objective_bound'] = model.ObjBound
    except (GurobiError, AttributeError):
        results['solver']['best_objective_bound'] = None

    if model.SolCount > 0:
        results['found_feasible_solution'] = True
    else:
        results['found_feasible_solution'] = False

    solutions = results['solutions'] = []

    is_discrete = model.getAttr(GRB.Attr.IsMIP)
    if is_discrete:
        if extract_reduced_costs:
            print('Cannot get reduced costs for MIP.')
        if extract_duals:
            print('Cannot get duals for MIP.')
        extract_reduced_costs = False
        extract_duals = False
    for solID in range(min(model.SolCount, NUM_SOLNS)):
        model.setParam('SolutionNumber', solID)
        vars = model.getVars()
        _sol = dict()
        _sol['X'] = model.getAttr("X", vars)
        _sol['VarName'] = model.getAttr("VarName", vars)
        if extract_reduced_costs:
            _sol['Rc'] = model.getAttr("Rc", vars)

        if extract_slacks or extract_duals:
            cons = model.getConstrs()
            qcons = model.getQConstrs()
            _sol['ConstrName'] = model.getAttr("ConstrName", cons)
            if GUROBI_VERSION[0] >= 5:
                _sol['QCName'] = model.getAttr("QCName", qcons)
        if extract_duals:
            _sol['Pi'] = model.getAttr("Pi", cons)
            if GUROBI_VERSION[0] >= 5:
                _sol['QCPi'] = model.getAttr("QCPi", qcons)
        if extract_slacks:
            _sol['Slack'] = model.getAttr("Slack", cons)
            if GUROBI_VERSION[0] >= 5:
                _sol['QCSlack'] = model.getAttr("QCSlack", qcons)

        solutions.append(_sol)

    return results


if __name__ == '__main__':
    model_file, soln_file, pyomo_options, options, suffixes = \
        pickle.load(sys.stdin)
    results = gurobi_run(model_file, pyomo_options, options, suffixes)
    with open(soln_file, 'wb') as SOLN:
        pickle.dump(results, SOLN, protocol=2)

