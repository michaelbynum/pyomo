import pyomo.environ as pe
import pyutilib.th as unittest
from parameterized import parameterized
from pyomo.solver.base import TerminationCondition


all_solvers = [('gurobi_lp', pe.SolverFactory('NEW_gurobi'),), ('gurobi_persistent', pe.SolverFactory('gurobi_persistent_new'),)]


class TestSolvers(unittest.TestCase):
    @parameterized.expand(input=all_solvers)
    def test_results_1(self, name, opt):
        if not opt.available():
            raise unittest.SkipTest(name + ' is not available')
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x)
        m.c2 = pe.Constraint(expr=m.y >= -m.x + 2)
        res = opt.solve(m)
        self.assertEqual(res.solver.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)
        self.assertTrue(res.found_feasible_solution())
        self.assertAlmostEqual(res.solver.best_feasible_objective, 1)

    @parameterized.expand(input=all_solvers)
    def test_results_2(self, name, opt):
        if not opt.available():
            raise unittest.SkipTest(name + ' is not available')
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x)
        m.c2 = pe.Constraint(expr=m.y <= m.x - 1)
        with self.assertRaises(Exception):
            res = opt.solve(m)
        res = opt.solve(m, load_solution=False)
        self.assertNotEqual(res.solver.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, None)
        self.assertAlmostEqual(m.y.value, None)
        self.assertFalse(res.found_feasible_solution())
        self.assertEqual(res.solver.best_feasible_objective, None)

    @parameterized.expand(input=all_solvers)
    def test_solution_loader(self, name, opt):
        if not opt.available():
            raise unittest.SkipTest(name + ' is not available')
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x)
        m.c2 = pe.Constraint(expr=m.y >= -m.x + 2)
        res = opt.solve(m, load_solution=False)
        self.assertEqual(res.solver.termination_condition, TerminationCondition.optimal)
        self.assertEqual(m.x.value, None)
        self.assertEqual(m.y.value, None)
        self.assertTrue(res.found_feasible_solution())
        self.assertAlmostEqual(res.solver.best_feasible_objective, 1)
        res.solution_loader.load_solution()
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)

    @parameterized.expand(input=all_solvers)
    def test_license_status(self, name, opt):
        if not opt.available():
            raise unittest.SkipTest(name + ' is not available')
        res = opt.license_status()
        self.assertIn(res, {True, False})

    @parameterized.expand(input=all_solvers)
    def test_version(self, name, opt):
        if not opt.available():
            raise unittest.SkipTest(name + ' is not available')
        version = opt.version()
        self.assertNotEqual(version, None)
        for i in version:
            self.assertEqual(type(i), int)
