import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory
import sys
import io
import logging


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-1, 1))
    m.y = pyo.Var(bounds=(-2, 2))
    m.obj = pyo.Objective(expr=m.y)
    m.c1 = pyo.Constraint(expr=m.y - m.x >= 0)
    m.c2 = pyo.Constraint(expr=m.y + m.x - 2 >= 0)

    # create the solver
    opt = SolverFactory('highs')

    out = io.StringIO()
    opt.config.tee = [sys.stdout, out, logger]
    results = opt.solve(m)
    print(out.getvalue())


if __name__ == '__main__':
    main()
