import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory
import argparse


def main(solver_name):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-1, 1))
    m.y = pyo.Var(bounds=(-2, 2))
    m.obj = pyo.Objective(expr=3 * m.x + 4 * m.y)

    # create the solver
    opt = SolverFactory(solver_name)

    # show the pyomo and solver options
    print('\n\n********************* config ****************')
    opt.config.display()
    
    # solve
    print('\n\n')
    results = opt.solve(m, tee=True)

    print('\n\n*************** results **************')
    results.display()


if __name__ == '__main__':
    # choose a solver
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', default='highs', type=str, required=False, help="name of the solver")
    args = parser.parse_args()

    main(args.solver)
