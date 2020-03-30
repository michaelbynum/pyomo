import pyomo.environ as pe
import pyutilib.th as unittest


opt = pe.SolverFactory('NEW_gurobi')
gurobi_available = opt.available()


@unittest.skipIf(not gurobi_available, 'Gurobi is not available')
class TestGurobi(unittest.TestCase):
    def test_duals(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y - m.x >= 0)
        m.c2 = pe.Constraint(expr=m.y + m.x - 2 >= 0)
        m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)
        self.assertAlmostEqual(m.dual[m.c1], 0.5)
        self.assertAlmostEqual(m.dual[m.c2], 0.5)

    def test_range_slacks(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.obj = pe.Objective(expr=m.x**2)
        m.c = pe.Constraint(expr=(-2, m.x, 1))
        m.slack = pe.Suffix(direction=pe.Suffix.IMPORT)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.slack[m.c], -2)

        del m.c
        m.c = pe.Constraint(expr=(-1, m.x, 2))
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.slack[m.c], 2)
