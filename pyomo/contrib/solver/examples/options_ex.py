import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory
import argparse


def main():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-1, 1))
    m.y = pyo.Var(bounds=(-2, 2))
    m.obj = pyo.Objective(expr=m.y)
    m.c1 = pyo.Constraint(expr=m.y - m.x >= 0)
    m.c2 = pyo.Constraint(expr=m.y + m.x - 2 >= 0)

    # create the solver
    opt = SolverFactory('gurobi_direct')
    
    # set tee = True to show the solver log
    opt.config.tee = True

    # we can override what is in the config with keyword arguments
    print('\n\n')
    print('starting fist solve')
    results = opt.solve(m, tee=False)
    print('done first solve')

    # lets solve again without the keyword arguments
    print('\n\n')
    print('starting second solve')
    results = opt.solve(m)
    print('done second solve')

    # setting solver specific options
    print('\n\n')
    opt.config.solver_options['presolve'] = 0
    print('starting third solve')
    results = opt.solve(m)
    print('done third solve')

    # again, let's use keyword arguments
    print('\n\n')
    print('starting fourth solve')
    results = opt.solve(m, solver_options={'presolve': 1})
    print('done fourth solve')


if __name__ == '__main__':
    main()
