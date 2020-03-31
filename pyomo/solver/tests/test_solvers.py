import pyomo.environ as pe
import pyutilib.th as unittest
from parameterized import parameterized
from pyomo.solver.base import TerminationCondition, MIPSolver


all_solvers = [('gurobi_lp', pe.SolverFactory('NEW_gurobi'),), ('gurobi_persistent', pe.SolverFactory('gurobi_persistent_new'),)]


"""
The tests in this file are used to ensure basic functionality/API works with all solvers

Feature                                    Tested
-------                                    ------
config time_limit                          x
config tee                                 
config load_solution True                  x
config load_solution False                 x     
results termination condition optimal      x
results termination condition infeasible   x
results found_feasible_solution            x partial; tested optimal and infeasible; still need to test feasible but not optimal 
solution_loader primals                    x
solution_loader dual                       x
solution_loader slack                      x
solution_loader rc                         x
solution_loader infeasible
solution_loader feasible                                         
available
license_status
version
"""


class TestSolvers(unittest.TestCase):
    @parameterized.expand(input=all_solvers)
    def test_results(self, name, opt):
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
    def test_results_infeasible(self, name, opt):
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
    def test_duals(self, name, opt):
        if not opt.available():
            raise unittest.SkipTest(name + ' is not available')
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

        del m.dual
        m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        res = opt.solve(m, load_solution=False)
        self.assertNotIn(m.c1, m.dual)
        self.assertNotIn(m.c2, m.dual)
        res.solution_loader.load_solution()
        self.assertAlmostEqual(m.dual[m.c1], 0.5)
        self.assertAlmostEqual(m.dual[m.c2], 0.5)
        del m.dual
        m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        res.solution_loader.load_duals(cons_to_load=[m.c1])
        self.assertAlmostEqual(m.dual[m.c1], 0.5)
        self.assertNotIn(m.c2, m.dual)

    @parameterized.expand(input=all_solvers)
    def test_slacks(self, name, opt):
        if not opt.available():
            raise unittest.SkipTest(name + ' is not available')
        m = pe.ConcreteModel()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= 0)
        m.c2 = pe.Constraint(expr=m.y <= 2)
        m.slack = pe.Suffix(direction=pe.Suffix.IMPORT)

        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 0)
        self.assertAlmostEqual(m.slack[m.c1], 0)
        self.assertAlmostEqual(m.slack[m.c2], 2)

        del m.slack
        m.slack = pe.Suffix(direction=pe.Suffix.IMPORT)
        res = opt.solve(m, load_solution=False)
        self.assertNotIn(m.c1, m.slack)
        self.assertNotIn(m.c2, m.slack)
        res.solution_loader.load_solution()
        self.assertAlmostEqual(m.slack[m.c1], 0)
        self.assertAlmostEqual(m.slack[m.c2], 2)
        del m.slack
        m.slack = pe.Suffix(direction=pe.Suffix.IMPORT)
        res.solution_loader.load_slacks(cons_to_load=[m.c1])
        self.assertAlmostEqual(m.slack[m.c1], 0)
        self.assertNotIn(m.c2, m.slack)

    @parameterized.expand(input=all_solvers)
    def test_rc(self, name, opt):
        if not opt.available():
            raise unittest.SkipTest(name + ' is not available')
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var(bounds=(-10, 10))
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y - m.x >= 0)
        m.rc = pe.Suffix(direction=pe.Suffix.IMPORT)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -1)
        self.assertAlmostEqual(m.y.value, -1)
        self.assertAlmostEqual(m.rc[m.x], 1)
        self.assertAlmostEqual(m.rc[m.y], 0)

        del m.rc
        m.rc = pe.Suffix(direction=pe.Suffix.IMPORT)
        res = opt.solve(m, load_solution=False)
        self.assertNotIn(m.x, m.rc)
        self.assertNotIn(m.y, m.rc)
        res.solution_loader.load_solution()
        self.assertAlmostEqual(m.rc[m.x], 1)
        self.assertAlmostEqual(m.rc[m.y], 0)
        del m.rc
        m.rc = pe.Suffix(direction=pe.Suffix.IMPORT)
        res.solution_loader.load_reduced_costs(vars_to_load=[m.x])
        self.assertAlmostEqual(m.rc[m.x], 1)
        self.assertNotIn(m.y, m.rc)

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
        m.x.value = None
        m.y.value = None
        res.solution_loader.load_vars(vars_to_load=[m.x])
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, None)

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

    @parameterized.expand(input=all_solvers)
    def test_time_limit(self, name, opt):
        if not opt.available():
            raise unittest.SkipTest(name + ' is not available')
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x)
        m.c2 = pe.Constraint(expr=m.y >= -m.x + 2)
        res = opt.solve(m, load_solution=False, time_limit=1e-10)
        self.assertNotEqual(res.solver.termination_condition, TerminationCondition.optimal)
        self.assertEqual(res.solver.termination_condition, TerminationCondition.maxTimeLimit)
