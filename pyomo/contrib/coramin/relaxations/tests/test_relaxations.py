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

    def test_build(self):
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

        # we should have a tangent at 1, a tangent at 4, and a secant between 1 and 4
        m.x.value = 0
        m.y.value = math.log(1)
        self.assert_all_satisfied(m.c)
        self.assert_at_least_one_active(m.c)
        m.y.value = math.log(1) + 0.1
        self.assert_at_least_one_violated(m.c)

        m.x.value = 3
        m.y.value = math.log(4)
        self.assert_all_satisfied(m.c)
        self.assert_at_least_one_active(m.c)
        m.y.value = math.log(4) + 0.1
        self.assert_at_least_one_violated(m.c)

        # intersection point of two tangents
        m.x.value = 0
        expr = pyo.log(m.x + 1)
        m1, b1 = get_tangent(expr, m.x)
        m.x.value = 3
        m2, b2 = get_tangent(expr, m.x)
        xhat, yhat = find_intersection(m1=m1, m2=m2, b1=b1, b2=b2)
        m.x.value = xhat
        m.y.value = yhat
        self.assert_all_satisfied(m.c)
        self.assert_at_least_one_active(m.c)
        m.y.value += 0.1
        self.assert_at_least_one_violated(m.c)

        # check the secant
        m.x.value = 0
        m.y.value = math.log(1) - 0.1
        self.assert_at_least_one_violated(m.c)

        m.x.value = 3
        m.y.value = math.log(4) - 0.1
        self.assert_at_least_one_violated(m.c)

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
        self.assert_at_least_one_violated(m.c)
        m.y.value = math.log(m.x.value + 1)
        self.assert_all_satisfied(m.c)
        self.assert_at_least_one_active(m.c)
        m.y.value += 0.1
        self.assert_at_least_one_violated(m.c)

        # make sure pprint works
        m.c.pprint()
        m.c.pprint(verbose=True)

        # change the bounds and rebuild
        m.x.setlb(2.5)
        
