import pyutilib.th as unittest
import pyomo.environ as pyo
from pyomo.solver.gurobi_persistent import GurobiPersistentNew
from pyomo.solver.base import TerminationCondition
from pyomo.core.expr.numeric_expr import LinearExpression
try:
    import gurobipy
    _tmp = gurobipy.Model()
    import numpy as np
    gurobipy_available = True
except:
    gurobipy_available = False


@unittest.skipIf(not gurobipy_available, 'gurobipy is not available')
class TestGurobiPersistentSimpleLPUpdates(unittest.TestCase):
    def setUp(self):
        self.m = pyo.ConcreteModel()
        m = self.m
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.p1 = pyo.Param(mutable=True)
        m.p2 = pyo.Param(mutable=True)
        m.p3 = pyo.Param(mutable=True)
        m.p4 = pyo.Param(mutable=True)
        m.obj = pyo.Objective(expr=m.x + m.y)
        m.c1 = pyo.Constraint(expr=m.y - m.p1 * m.x >= m.p2)
        m.c2 = pyo.Constraint(expr=m.y - m.p3 * m.x >= m.p4)

    def get_solution(self):
        p1 = self.m.p1.value
        p2 = self.m.p2.value
        p3 = self.m.p3.value
        p4 = self.m.p4.value
        A = np.array([[1, -p1],
                      [1, -p3]])
        rhs = np.array([p2,
                        p4])
        sol = np.linalg.solve(A, rhs)
        x = float(sol[1])
        y = float(sol[0])
        return x, y

    def set_params(self, p1, p2, p3, p4):
        self.m.p1.value = p1
        self.m.p2.value = p2
        self.m.p3.value = p3
        self.m.p4.value = p4

    def test_lp(self):
        self.set_params(-1, -2, 0.1, -2)
        x, y = self.get_solution()
        opt = GurobiPersistentNew()
        res = opt.solve(self.m)
        self.assertAlmostEqual(x + y, res.solver.best_feasible_objective)
        self.assertAlmostEqual(x + y, res.solver.best_objective_bound)
        self.assertEqual(res.solver.termination_condition, TerminationCondition.optimal)
        self.assertTrue(res.found_feasible_solution())
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)

        self.set_params(-1.25, -1, 0.5, -2)
        res = opt.solve(self.m, load_solution=False)
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)
        x, y = self.get_solution()
        self.assertNotAlmostEquals(x, self.m.x.value)
        self.assertNotAlmostEquals(y, self.m.y.value)
        res.solution_loader.load_solution()
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)

        self.m.x.value = None
        self.m.y.value = None
        res.solution_loader.load_vars()
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)

    def test_no_mutable_params(self):
        self.set_params(-1, -2, 0.1, -2)
        x, y = self.get_solution()
        opt = GurobiPersistentNew()
        opt.config.check_for_updated_mutable_params_in_constraints = False
        opt.config.check_for_updated_mutable_params_in_objective = False
        res = opt.solve(self.m)
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)


@unittest.skipIf(not gurobipy_available, 'gurobipy is not available')
class TestGurobiPersistent(unittest.TestCase):
    def test_range_constraints(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.xl = pyo.Param(initialize=-1, mutable=True)
        m.xu = pyo.Param(initialize=1, mutable=True)
        m.c = pyo.Constraint(expr=pyo.inequality(m.xl, m.x, m.xu))
        m.obj = pyo.Objective(expr=m.x)

        opt = GurobiPersistentNew()
        opt.set_instance(m)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -1)

        m.xl.value = -3
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -3)

        del m.obj
        m.obj = pyo.Objective(expr=m.x, sense=pyo.maximize)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)

        m.xu.value = 3
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 3)

    def test_quadratic_constraint_with_params(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Param(initialize=1, mutable=True)
        m.b = pyo.Param(initialize=1, mutable=True)
        m.c = pyo.Param(initialize=1, mutable=True)
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.con = pyo.Constraint(expr=m.y >= m.a*m.x**2 + m.b*m.x + m.c)

        opt = GurobiPersistentNew()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(m.y.value, m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)

        m.a.value = 2
        m.b.value = 4
        m.c.value = -1
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(m.y.value, m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)

    def test_quadratic_objective(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Param(initialize=1, mutable=True)
        m.b = pyo.Param(initialize=1, mutable=True)
        m.c = pyo.Param(initialize=1, mutable=True)
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.a*m.x**2 + m.b*m.x + m.c)

        opt = GurobiPersistentNew()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(res.solver.best_feasible_objective,
                               m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)

        m.a.value = 2
        m.b.value = 4
        m.c.value = -1
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(res.solver.best_feasible_objective,
                               m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)

    def test_var_bounds(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1, 1))
        m.obj = pyo.Objective(expr=m.x)

        opt = GurobiPersistentNew()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -1)

        m.x.setlb(-3)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -3)

        del m.obj
        m.obj = pyo.Objective(expr=m.x, sense=pyo.maximize)

        opt = GurobiPersistentNew()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)

        m.x.setub(3)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 3)

    def test_fixed_var(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Param(initialize=1, mutable=True)
        m.b = pyo.Param(initialize=1, mutable=True)
        m.c = pyo.Param(initialize=1, mutable=True)
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.con = pyo.Constraint(expr=m.y >= m.a*m.x**2 + m.b*m.x + m.c)

        m.x.fix(1)
        opt = GurobiPersistentNew()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 3)

        m.x.value = 2
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 7)

        m.x.unfix()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(m.y.value, m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)


@unittest.skipIf(not gurobipy_available, 'gurobipy is not available')
class TestGurobiWalker(unittest.TestCase):
    def test_linear_expression(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=0 <= LinearExpression(constant=-1,
                                                         linear_coefs=[1, -2],
                                                         linear_vars=[m.y, m.x]))
        m.c2 = pyo.Constraint(expr=0 <= LinearExpression(constant=-1,
                                                         linear_coefs=[1, 2],
                                                         linear_vars=[m.y, m.x]))
        opt = GurobiPersistentNew()
        opt.config.check_for_updated_mutable_params_in_constraints = False
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

    def test_native_types(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.p = pyo.Param(initialize=2)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=0 <= 2*(0.5*m.y - m.x - 0.5))
        m.c2 = pyo.Constraint(expr=0 <= m.y + m.p*m.x - 1)
        opt = GurobiPersistentNew()
        opt.config.check_for_updated_mutable_params_in_constraints = False
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

    def test_quadratic(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Param(initialize=1)
        m.b = pyo.Param(initialize=1)
        m.c = pyo.Param(initialize=1)
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.con = pyo.Constraint(expr=m.y >= m.a*m.x**2 + m.b*m.x + m.c)

        opt = GurobiPersistentNew()
        opt.config.check_for_updated_mutable_params_in_constraints = False
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(m.y.value, m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)
