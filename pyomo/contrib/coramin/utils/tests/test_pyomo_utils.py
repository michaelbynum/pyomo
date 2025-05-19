from pyomo.contrib.coramin.utils.pyomo_utils import (
    get_objective, 
    identify_variables_with_cache, 
    active_vars,
    active_cons, 
    simplify_expr,
    _var_cache,
)
from pyomo.common.unittest import TestCase, skipIf
import pyomo.environ as pe
from pyomo.common.collections import ComponentSet
import gc
from pyomo.contrib.simplification.simplify import ginac_available
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.common.errors import PyomoException


class TestGetObjective(TestCase):
    def test_get_objective(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c = pe.Constraint(expr=m.y >= pe.exp(m.x))
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)

        self.assertIs(get_objective(m), m.obj)

        m.obj2 = pe.Objective(expr=m.y)
        with self.assertRaises(PyomoException):
            get_objective(m)

        m.obj.deactivate()
        self.assertIs(get_objective(m), m.obj2)


class TestIdentifyVariables(TestCase):
    def test_identify_var_cache(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c = pe.Constraint(expr=m.y >= pe.exp(m.x))

        vset = ComponentSet(identify_variables_with_cache(con=m.c, include_fixed=True))
        self.assertIn(m.x, vset)
        self.assertIn(m.y, vset)
        self.assertEqual(len(vset), 2)

        # make sure it works twice in a row with the cache
        self.assertIn(m.c, _var_cache)
        vset = ComponentSet(identify_variables_with_cache(con=m.c, include_fixed=True))
        self.assertIn(m.x, vset)
        self.assertIn(m.y, vset)
        self.assertEqual(len(vset), 2)

        # check fixed variables
        m.y.fix(1)
        vset = ComponentSet(identify_variables_with_cache(con=m.c, include_fixed=True))
        self.assertIn(m.x, vset)
        self.assertIn(m.y, vset)
        self.assertEqual(len(vset), 2)
        vset = ComponentSet(identify_variables_with_cache(con=m.c, include_fixed=False))
        self.assertIn(m.x, vset)
        self.assertNotIn(m.y, vset)
        self.assertEqual(len(vset), 1)

        # delete the constraint, run the garbage collector, 
        # and ensure the constraint is no longer in the cache
        del m.c
        gc.collect()
        self.assertEqual(len(_var_cache), 0)

    def test_active_vars_and_cons(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.c = pe.Constraint(expr=m.y >= pe.exp(m.x))

        vset = ComponentSet(active_vars(m=m, include_fixed=True))
        self.assertEqual(len(vset), 2)
        self.assertIn(m.x, vset)
        self.assertIn(m.y, vset)
        self.assertNotIn(m.z, vset)

        m.obj = pe.Objective(expr=m.z**2)

        vset = ComponentSet(active_vars(m=m, include_fixed=True))
        self.assertEqual(len(vset), 3)
        self.assertIn(m.x, vset)
        self.assertIn(m.y, vset)
        self.assertIn(m.z, vset)

        cset = ComponentSet(active_cons(m=m))
        self.assertIn(m.c, cset)
        self.assertEqual(len(cset), 1)


@skipIf(not (sympy_available or ginac_available), 'neither sympy nor ginac is available')
class TestSimplify(TestCase):
    def test_simplify_expr(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        e1 = m.x + m.y - m.x
        e2 = m.x + 1.7 - m.x
        e1_simp = simplify_expr(e1)
        e2_simp = simplify_expr(e2)
        self.assertIs(e1_simp, m.y)
        self.assertEqual(e2_simp, 1.7)
