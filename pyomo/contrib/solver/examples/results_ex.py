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
    opt = SolverFactory('highs')

    results = opt.solve(m, load_solutions=False, raise_exception_on_nonoptimal_result=False)
    print(results.solver_log)

    # if a feasible solution was found, print the primals and duals
    if results.incumbent_objective is not None:
        # the solution_loader object can be used to extract primal and dual values
        primals = results.solution_loader.get_primals()
        duals = results.solution_loader.get_duals()
        rc = results.solution_loader.get_reduced_costs()

        print('primals:')
        for k, v in primals.items():
            print(f'  {k}: {v}')
        print('duals:')
        for k, v in duals.items():
            print(f'  {k}: {v}')
        print('reduced costs:')
        for k, v in rc.items():
            print(f'  {k}: {v}')

        # we can also load the primal values into the model
        results.solution_loader.load_vars()
        m.pprint()


if __name__ == '__main__':
    main()
