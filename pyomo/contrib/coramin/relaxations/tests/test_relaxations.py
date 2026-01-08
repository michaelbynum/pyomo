from pyomo.common.unittest import TestCase, skipIf
import pyomo.environ as pyo
from pyomo.contrib import coramin
import math
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_ad


def find_intersection(m1, m2, b1, b2):
    # find the point at which two lines intersect
    # y = m1*x + b1
    # y = m2*x + b2
    # m1*x + b1 = m2*x + b2
    # m1*x - m2*x = b2 - b1
    # x*(m1 - m2) = b2 - b1
    # x = (b2 - b1) / (m1 - m2)
    x = (b2 - b1) / (m1 - m2)
    y = m1*x + b1
    return x, y


def get_tangent(expr, x):
    m = reverse_ad(expr)[x]
    yval = pyo.value(expr)
    xval = pyo.value(x)
    # yval = m*xval + b
    # b = yval - m*xval
    b = yval - m*xval
    return m, b


def line_from_points(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m*x1
    return m, b


class TestUnivariateConcave(TestCase):
    def setUp(self) -> None:
        self.abs_tol = 1e-6
        self.rel_tol = 1e-6

    def assert_at_least_one_active(self, block):
        found_active = False
        for con in block.component_data_objects(pyo.Constraint, active=True, descend_into=True):
            lb = pyo.value(con.lb)
            ub = pyo.value(con.ub)
            body = pyo.value(con.body)
            if lb is not None and math.isclose(lb, body, abs_tol=self.abs_tol, rel_tol=self.rel_tol):
                found_active = True
            if ub is not None and math.isclose(ub, body, abs_tol=self.abs_tol, rel_tol=self.rel_tol):
                found_active = True
            if found_active:
                break
        self.assertTrue(found_active)

    def assert_none_active(self, block):
        found_active = False
        for con in block.component_data_objects(pyo.Constraint, active=True, descend_into=True):
            lb = pyo.value(con.lb)
            ub = pyo.value(con.ub)
            body = pyo.value(con.body)
            if lb is not None and math.isclose(lb, body, abs_tol=self.abs_tol, rel_tol=self.rel_tol):
                found_active = True
            if ub is not None and math.isclose(ub, body, abs_tol=self.abs_tol, rel_tol=self.rel_tol):
                found_active = True
            if found_active:
                break
        self.assertFalse(found_active)

    def assert_all_satisfied(self, block):
        found_violated = False
        for con in block.component_data_objects(pyo.Constraint, active=True, descend_into=True):
            lb = pyo.value(con.lb)
            ub = pyo.value(con.ub)
            body = pyo.value(con.body)
            if lb is not None and (lb - self.abs_tol - lb*self.rel_tol > body):
                found_violated = True
                break
            if ub is not None and (ub + self.abs_tol + ub*self.rel_tol < body):
                found_violated = True
                break
        self.assertFalse(found_violated)

    def assert_at_least_one_violated(self, block):
        found_violated = False
        for con in block.component_data_objects(pyo.Constraint, active=True, descend_into=True):
            lb = pyo.value(con.lb)
            ub = pyo.value(con.ub)
            body = pyo.value(con.body)
            if lb is not None and (lb - self.abs_tol - lb*self.rel_tol > body):
                found_violated = True
                break
            if ub is not None and (ub + self.abs_tol + ub*self.rel_tol < body):
                found_violated = True
                break
        self.assertTrue(found_violated)

    def _check_tight(self, block, nl_expr, y, direction=coramin.enums.RelaxationSide.BOTH):
        y.value = pyo.value(nl_expr)
        self.assert_all_satisfied(block)
        self.assert_at_least_one_active(block)
        if direction in {coramin.enums.RelaxationSide.BOTH, coramin.enums.RelaxationSide.OVER}:
            y.value = pyo.value(nl_expr) + 0.1
            self.assert_at_least_one_violated(block)
        if direction in {coramin.enums.RelaxationSide.BOTH, coramin.enums.RelaxationSide.UNDER}:
            y.value = pyo.value(nl_expr) - 0.1
            self.assert_at_least_one_violated(block)

    def _check_not_tight(self, block, nl_expr, y):
        y.value = pyo.value(nl_expr)
        self.assert_all_satisfied(block)
        self.assert_none_active(block)

    def test_build_and_refine(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 3))
        m.y = pyo.Var()
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        nl_expr = pyo.log(m.x + 1)
        m.c.build(
            x=m.x,
            aux_var=m.y,
            shape=coramin.enums.FunctionShape.CONCAVE,
            f_x_expr=nl_expr,
            relaxation_side=coramin.enums.RelaxationSide.BOTH,
        )

        # we should have a tangent at x=0, a tangent at x=3, and a secant between x=0 and x=3
        m.x.value = 0
        self._check_tight(m.c, nl_expr, m.y)

        m.x.value = 3
        self._check_tight(m.c, nl_expr, m.y)

        # intersection point of two tangents
        m.x.value = 0
        m1, b1 = get_tangent(nl_expr, m.x)
        m.x.value = 3
        m2, b2 = get_tangent(nl_expr, m.x)
        xhat, yhat = find_intersection(m1=m1, m2=m2, b1=b1, b2=b2)
        m.x.value = xhat
        m.y.value = yhat
        self.assert_all_satisfied(m.c)
        self.assert_at_least_one_active(m.c)
        m.y.value += 0.1
        self.assert_at_least_one_violated(m.c)

        # check mid-point of the secant
        slope, intercept = line_from_points(x1=0, y1=math.log(1), x2=3, y2=math.log(4))
        mid_x = 1.5
        mid_y = slope * mid_x + intercept
        m.x.value = mid_x
        m.y.value = mid_y
        self.assert_all_satisfied(m.c)
        self.assert_at_least_one_active(m.c)
        m.y.value -= 0.1
        self.assert_at_least_one_violated(m.c)

        # add a tangent at xhat and make sure that point becomes violated
        m.x.value = xhat
        m.y.value = yhat
        m.c.add_cut()
        self._check_tight(m.c, nl_expr, m.y, direction=coramin.enums.RelaxationSide.OVER)

        # change the bounds and rebuild
        m.x.setlb(2.5)
        m.c.rebuild()
        m.x.value = 2.5
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 3
        self._check_tight(m.c, nl_expr, m.y)

    def test_oa_points(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 3))
        m.y = pyo.Var()
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        nl_expr = pyo.log(m.x + 1)
        m.c.build(
            x=m.x,
            aux_var=m.y,
            shape=coramin.enums.FunctionShape.CONCAVE,
            f_x_expr=nl_expr,
            relaxation_side=coramin.enums.RelaxationSide.BOTH,
        )

        m.x.value = 0
        self._check_tight(m.c, nl_expr, m.y)

        m.x.value = 3
        self._check_tight(m.c, nl_expr, m.y)

        m.x.value = 1.5
        self._check_not_tight(m.c, nl_expr, m.y)

        m.c.add_oa_point()
        m.c.rebuild()
        self.assertEqual(m.x.value, 1.5)
        self._check_tight(m.c, nl_expr, m.y, direction=coramin.enums.RelaxationSide.OVER)

        m.c.clear_oa_points()
        m.c.rebuild()
        m.x.value = 0
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 3
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 1.5
        self._check_not_tight(m.c, nl_expr, m.y)

        m.x.value = 0
        m.c.add_oa_point((1.5,))
        m.c.rebuild()
        m.x.value = 0
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 3
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 1.5
        self._check_tight(m.c, nl_expr, m.y, direction=coramin.enums.RelaxationSide.OVER)

        m.c.push_oa_points()
        m.c.clear_oa_points()
        m.c.rebuild()
        m.x.value = 0
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 3
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 1.5
        self._check_not_tight(m.c, nl_expr, m.y)

        m.c.pop_oa_points()
        m.c.rebuild()
        m.x.value = 0
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 3
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 1.5
        self._check_tight(m.c, nl_expr, m.y, direction=coramin.enums.RelaxationSide.OVER)

        m.c.push_oa_points(key='foo')
        m.c.clear_oa_points()
        m.c.rebuild()
        m.x.value = 0
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 3
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 1.5
        self._check_not_tight(m.c, nl_expr, m.y)

        m.c.pop_oa_points(key='foo')
        m.c.rebuild()
        m.x.value = 0
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 3
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 1.5
        self._check_tight(m.c, nl_expr, m.y, direction=coramin.enums.RelaxationSide.OVER)

        m.c.clear_oa_points()
        m.c.rebuild()
        m.x.value = 0
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 3
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 1.5
        self._check_not_tight(m.c, nl_expr, m.y)

        var_vals = pyo.ComponentMap()
        var_vals[m.x] = 1.5
        m.c.add_oa_point(var_vals)
        m.c.rebuild()
        m.x.value = 0
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 3
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 1.5
        self._check_tight(m.c, nl_expr, m.y, direction=coramin.enums.RelaxationSide.OVER)

    def test_unbounded_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(None, None))
        m.y = pyo.Var()
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        nl_expr = pyo.exp(m.x + 1)
        m.c.build(
            x=m.x,
            aux_var=m.y,
            shape=coramin.enums.FunctionShape.CONVEX,
            f_x_expr=nl_expr,
            relaxation_side=coramin.enums.RelaxationSide.BOTH,
        )
        
        m.x.value = -2
        self._check_not_tight(m.c, nl_expr, m.y)
        m.x.value = 0
        self._check_not_tight(m.c, nl_expr, m.y)
        m.x.value = 2
        self._check_not_tight(m.c, nl_expr, m.y)

        m.x.setlb(-2)
        m.c.rebuild()
        m.x.value = -2
        self._check_tight(m.c, nl_expr, m.y, direction=coramin.enums.RelaxationSide.UNDER)
        m.x.value = 0
        self._check_not_tight(m.c, nl_expr, m.y)
        m.x.value = 2
        self._check_not_tight(m.c, nl_expr, m.y)

        m.x.setlb(None)
        m.x.setub(2)
        m.c.rebuild()
        m.x.value = -2
        self._check_not_tight(m.c, nl_expr, m.y)
        m.x.value = 0
        self._check_not_tight(m.c, nl_expr, m.y)
        m.x.value = 2
        self._check_tight(m.c, nl_expr, m.y, direction=coramin.enums.RelaxationSide.UNDER)

        m.x.setlb(-2)
        m.c.rebuild()
        m.x.value = -2
        self._check_tight(m.c, nl_expr, m.y)
        m.x.value = 0
        self._check_not_tight(m.c, nl_expr, m.y)
        m.x.value = 2
        self._check_tight(m.c, nl_expr, m.y)

    def test_small_and_large_coefs(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0.01, 100))
        m.y = pyo.Var()
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        nl_expr = pyo.log(m.x)
        m.c.build(
            x=m.x,
            aux_var=m.y,
            shape=coramin.enums.FunctionShape.CONCAVE,
            f_x_expr=nl_expr,
            relaxation_side=coramin.enums.RelaxationSide.OVER,
            large_coef = 10,
            small_coef = 0.1,
        )

        m.x.value = m.x.lb
        self._check_not_tight(m.c, nl_expr, m.y)
        m.x.value = m.x.ub
        self.assert_all_satisfied(m.c)
        m.x.value = 99
        m.c.add_cut(check_violation=False)
        self.assert_all_satisfied(m.c)
        m.x.value = 98
        m.c.add_cut(check_violation=False)
        self.assert_all_satisfied(m.c)
        self.assertEqual(m.c._oa_params[2].value, 0)
        self.assertEqual(m.c._oa_params[4].value, 0)
        self.assertEqual(m.c._oa_params[6].value, 0)
        self.assertGreaterEqual(m.c._oa_params[3].value, math.log(100))
        self.assertGreaterEqual(m.c._oa_params[5].value, math.log(100))
        self.assertGreaterEqual(m.c._oa_params[7].value, math.log(100))

    def test_pprint(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 3))
        m.y = pyo.Var()
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        m.c.build(
            x=m.x,
            aux_var=m.y,
            shape=coramin.enums.FunctionShape.CONCAVE,
            f_x_expr=pyo.log(m.x + 1),
            relaxation_side=coramin.enums.RelaxationSide.BOTH,
        )

        # For now, we will just make sure no exception is raised
        m.c.pprint()
        m.c.pprint(verbose=True)
        
