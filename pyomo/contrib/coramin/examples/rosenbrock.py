import pyomo.environ as pe
from pyomo.contrib import coramin
from pyomo.contrib.solver.common.factory import SolverFactory


def create_nlp(a, b):
    # Create the nlp
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(-20.0, 20.0))
    m.y = pe.Var(bounds=(-20.0, 20.0))

    m.objective = pe.Objective(expr=(a - m.x) ** 2 + b * (m.y - m.x**2) ** 2)

    return m


def main():
    a = 1
    b = 1
    nlp = create_nlp(a, b)
    rel = coramin.relaxations.relax(nlp)

    nlp_opt = SolverFactory('ipopt')
    rel_opt = SolverFactory('gurobi_persistent')

    res = nlp_opt.solve(nlp, tee=False)
    ub = res.incumbent_objective

    res = rel_opt.solve(rel, tee=False)
    lb = res.objective_bound

    print('lb: ', lb)
    print('ub: ', ub)

    print('nlp results:')
    print('--------------------------')
    nlp.x.pprint()
    nlp.y.pprint()

    print('relaxation results:')
    print('--------------------------')
    rel.x.pprint()
    rel.y.pprint()


if __name__ == '__main__':
    main()
