import pyutilib.th as unittest
import pyomo.environ as pyo
from pyomo.solver.gurobi_persistent import GurobiPersistentNew
from pyomo.solver.base import TerminationCondition
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.taylor_series import taylor_series_expansion
try:
    import gurobipy
    _tmp = gurobipy.Model()
    import numpy as np
    gurobipy_available = True
except ImportError:
    gurobipy_available = False

pe = pyo


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

    def test_linear_constraint_attr(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c = pe.Constraint(expr=m.x + m.y == 1)

        opt = pe.SolverFactory('gurobi_persistent_new')
        opt.set_instance(m)
        opt.set_linear_constraint_attr(m.c, 'Lazy', 1)
        self.assertEqual(opt.get_linear_constraint_attr(m.c, 'Lazy'), 1)

    def test_quadratic_constraint_attr(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c = pe.Constraint(expr=m.y >= m.x**2)

        opt = pe.SolverFactory('gurobi_persistent_new')
        opt.set_instance(m)
        self.assertEqual(opt.get_quadratic_constraint_attr(m.c, 'QCRHS'), 0)

    def test_var_attr(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(within=pe.Binary)

        opt = pe.SolverFactory('gurobi_persistent_new')
        opt.set_instance(m)
        opt.set_var_attr(m.x, 'Start', 1)
        self.assertEqual(opt.get_var_attr(m.x, 'Start'), 1)

    def test_callback(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(0, 4))
        m.y = pe.Var(within=pe.Integers, bounds=(0, None))
        m.obj = pe.Objective(expr=2*m.x + m.y)
        m.cons = pe.ConstraintList()

        def _add_cut(xval):
            m.x.value = xval
            return m.cons.add(m.y >= taylor_series_expansion((m.x - 2)**2))

        _add_cut(0)
        _add_cut(4)

        opt = pe.SolverFactory('gurobi_persistent_new')
        opt.set_instance(m)
        opt.set_gurobi_param('PreCrush', 1)
        opt.set_gurobi_param('LazyConstraints', 1)

        def _my_callback(cb_m, cb_opt, cb_where):
            if cb_where == gurobipy.GRB.Callback.MIPSOL:
                cb_opt.cbGetSolution(vars=[m.x, m.y])
                if m.y.value < (m.x.value - 2)**2 - 1e-6:
                    cb_opt.cbLazy(_add_cut(m.x.value))

        opt.set_callback(_my_callback)
        opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)


@unittest.skipIf(not gurobipy_available, 'gurobipy is not available')
class TestManualModel(unittest.TestCase):
    def setUp(self):
        opt = pe.SolverFactory('gurobi_persistent_new')
        opt.config.check_for_updated_mutable_params_in_constraints = False
        opt.config.check_for_updated_mutable_params_in_objective = False
        opt.config.check_for_new_or_removed_constraints = False
        opt.config.update_constraints = False
        opt.config.check_for_new_or_removed_vars = False
        opt.config.update_vars = False
        opt.config.update_named_expressions = False
        opt.config.check_for_fixed_vars = False
        self.opt = opt

    def test_basics(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= 2*m.x + 1)

        opt = self.opt
        opt.set_instance(m)

        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -10)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 10)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        opt.load_duals()
        self.assertAlmostEqual(m.dual[m.c1], -0.4)
        del m.dual

        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 2)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        res = opt.solve(m, load_solution=False)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        opt.load_vars()
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

        opt.remove_constraint(m.c2)
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        self.assertEqual(opt.get_gurobi_param_info('FeasibilityTol')[2], 1e-6)
        res = opt.solve(m, options={'FeasibilityTol': 1e-7})
        self.assertEqual(opt.get_gurobi_param_info('FeasibilityTol')[2], 1e-7)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)

        m.x.setlb(-5)
        m.x.setub(5)
        opt.update_var(m.x)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)

        m.x.fix(0)
        opt.update_var(m.x)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 0)

        m.x.unfix()
        opt.update_var(m.x)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)

        m.c2 = pe.Constraint(expr=m.y >= m.x**2)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 1)

        opt.remove_constraint(m.c2)
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        m.z = pe.Var()
        opt.add_var(m.z)
        self.assertEqual(opt.get_model_attr('NumVars'), 3)
        opt.remove_var(m.z)
        del m.z
        self.assertEqual(opt.get_model_attr('NumVars'), 2)

    def test_update1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)

        opt.remove_constraint(m.c1)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)

        opt.add_constraint(m.c1)
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)

    def test_update2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c2 = pe.Constraint(expr=m.x + m.y == 1)

        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)

        opt.remove_constraint(m.c2)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)

        opt.add_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)

    def test_update3(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
        m.c2 = pe.Constraint(expr=m.y >= m.x**2)
        opt.add_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
        opt.remove_constraint(m.c2)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)

    def test_update4(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x + m.y)

        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)
        m.c2 = pe.Constraint(expr=m.y >= m.x)
        opt.add_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)
        opt.remove_constraint(m.c2)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)

    def test_update5(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1,2,3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)

        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)

        opt.remove_sos_constraint(m.c1)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)

        opt.add_sos_constraint(m.c1)
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)

    def test_update6(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1,2,3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)

        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
        m.c2 = pe.SOSConstraint(var=m.x, sos=2)
        opt.add_sos_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
        opt.remove_sos_constraint(m.c2)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)

    def test_update7(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()

        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 0)

        opt.remove_var(m.x)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)

        opt.add_var(m.x)
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 2)

        opt.remove_var(m.x)
        opt.update()
        opt.add_var(m.x)
        opt.remove_var(m.x)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)


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
