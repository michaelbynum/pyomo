import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.common.timing import HierarchicalTimer


def main():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.p = pyo.Param(mutable=True, initialize=0)
    m.obj = pyo.Objective(expr=m.y)
    m.c1 = pyo.Constraint(expr=m.y >= m.x + m.p)
    m.c2 = pyo.Constraint(expr=m.y >= -m.x)

    timer = HierarchicalTimer()
    opt = SolverFactory('highs')
    opt.config.timer = timer
    timer.start('total')
    for p in range(100):
        m.p.value = p
        results = opt.solve(m)
    timer.stop('total')
    print(opt.config.timer)


if __name__ == '__main__':
    main()
