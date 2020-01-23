import pyutilib.th as unittest
import pyomo.environ as pyo
from pyomo.solver.gurobi_persistent import GurobiPersistentNew
from pyomo.solver.base import TerminationCondition
try:
    import gurobipy
    m = gurobipy.Model()
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
        res = opt.solve(self.m, load_solutions=False)
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
