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
    b = 100
    nlp = create_nlp(a, b)
    rel = coramin.relaxations.relax(nlp)
    rel.pprint()

    nlp_opt = SolverFactory('ipopt')
    rel_opt = SolverFactory('highs')

    res = nlp_opt.solve(nlp, tee=False)
    ub = res.incumbent_objective
    print(f'upper bound: {ub};    x: {nlp.x.value};    y: {nlp.y.value}')

    res = rel_opt.solve(rel, tee=False)
    lb = res.objective_bound
    print(f'lower bound: {lb};    x: {nlp.x.value};    y: {nlp.y.value}')

    for iter_ndx in range(30):
        for r in coramin.relaxations.relaxation_data_objects(rel, descend_into=True, active=True):
            v = r.get_rhs_vars()[0]
            r.add_oa_point(var_values=(v.value,))
            r.rebuild()

        res = rel_opt.solve(rel, tee=False)
        lb = res.objective_bound
        print(f'iter: {iter_ndx};    lower bound: {lb};    x: {nlp.x.value};    y: {nlp.y.value}')


if __name__ == '__main__':
    main()
