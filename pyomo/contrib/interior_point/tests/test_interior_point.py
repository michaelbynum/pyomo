import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.common.dependencies import attempt_import

np, numpy_availalbe = attempt_import('numpy', 'Interior point requires numpy', minimum_version='1.13.0')
scipy, scipy_available = attempt_import('scipy', 'Interior point requires scipy')
mumps, mumps_available = attempt_import('mumps', 'Interior point requires mumps')
if not (numpy_availalbe and scipy_available):
    raise unittest.SkipTest('Interior point tests require numpy and scipy')

import numpy as np

from pyomo.contrib.pynumero.asl import AmplInterface
asl_available = AmplInterface.available()
import pyomo.contrib.interior_point as ip
from pyomo.contrib.interior_point.interior_point import (process_init,
                                                         process_init_duals_lb,
                                                         process_init_duals_ub,
                                                         _fraction_to_the_boundary_helper_lb,
                                                         _fraction_to_the_boundary_helper_ub)
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
ma27_available = MA27Interface.available()


@unittest.skipIf(not asl_available, 'asl is not available')
class TestSolveInteriorPoint(unittest.TestCase):
    def _test_solve_interior_point_1(self, linear_solver):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y == pe.exp(m.x))
        m.c2 = pe.Constraint(expr=m.y >= (m.x - 1)**2)
        interface = ip.interface.InteriorPointInterface(m)
        ip_solver = ip.interior_point.InteriorPointSolver(linear_solver)
        status = ip_solver.solve(interface)
        self.assertEqual(status, ip.interior_point.InteriorPointStatus.optimal)
        x = interface.get_primals()
        duals_eq = interface.get_duals_eq()
        duals_ineq = interface.get_duals_ineq()
        self.assertAlmostEqual(x[0], 0)
        self.assertAlmostEqual(x[1], 1)
        self.assertAlmostEqual(duals_eq[0], -1-1.0/3.0)
        self.assertAlmostEqual(duals_ineq[0], 2.0/3.0)
        interface.load_primals_into_pyomo_model()
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

    def _test_solve_interior_point_2(self, linear_solver):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(1, 4))
        m.obj = pe.Objective(expr=m.x**2)
        interface = ip.interface.InteriorPointInterface(m)
        ip_solver = ip.interior_point.InteriorPointSolver(linear_solver)
        status = ip_solver.solve(interface)
        self.assertEqual(status, ip.interior_point.InteriorPointStatus.optimal)
        interface.load_primals_into_pyomo_model()
        self.assertAlmostEqual(m.x.value, 1)

    def test_ip1_scipy(self):
        solver = ip.linalg.scipy_interface.ScipyInterface()
        solver.compute_inertia = True
        self._test_solve_interior_point_1(solver)

    def test_ip2_scipy(self):
        solver = ip.linalg.scipy_interface.ScipyInterface()
        solver.compute_inertia = True
        self._test_solve_interior_point_2(solver)

    @unittest.skipIf(not mumps_available, 'Mumps is not available')
    def test_ip1_mumps(self):
        solver = ip.linalg.mumps_interface.MumpsInterface()
        self._test_solve_interior_point_1(solver)

    @unittest.skipIf(not mumps_available, 'Mumps is not available')
    def test_ip2_mumps(self):
        solver = ip.linalg.mumps_interface.MumpsInterface()
        self._test_solve_interior_point_2(solver)

    @unittest.skipIf(not ma27_available, 'MA27 is not available')
    def test_ip1_ma27(self):
        solver = ip.linalg.ma27_interface.InteriorPointMA27Interface()
        self._test_solve_interior_point_1(solver)

    @unittest.skipIf(not ma27_available, 'MA27 is not available')
    def test_ip2_ma27(self):
        solver = ip.linalg.ma27_interface.InteriorPointMA27Interface()
        self._test_solve_interior_point_2(solver)


class TestProcessInit(unittest.TestCase):
    def testprocess_init(self):
        lb = np.array([-np.inf, -np.inf,     -2, -2], dtype=np.double)
        ub = np.array([ np.inf,       2, np.inf,  2], dtype=np.double)

        x = np.array([       0,       0,      0,  0], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([0, 0, 0, 0], dtype=np.double)))

        x = np.array([      -2,      -2,     -2,  -2], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([-2, -2, -1, 0], dtype=np.double)))

        x = np.array([      -3,      -3,     -3,  -3], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([-3, -3, -1, 0], dtype=np.double)))

        x = np.array([       2,       2,      2,   2], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([2, 1, 2, 0], dtype=np.double)))

        x = np.array([       3,       3,      3,   3], dtype=np.double)
        process_init(x, lb, ub)
        self.assertTrue(np.allclose(x, np.array([3, 1, 3, 0], dtype=np.double)))

    def testprocess_init_duals(self):
        x = np.array([0, 0, 0, 0], dtype=np.double)
        lb = np.array([-5, 0, -np.inf, 2], dtype=np.double)
        process_init_duals_lb(x, lb)
        self.assertTrue(np.allclose(x, np.array([1, 1, 0, 1], dtype=np.double)))

        x = np.array([-1, -1, -1, -1], dtype=np.double)
        process_init_duals_lb(x, lb)
        self.assertTrue(np.allclose(x, np.array([1, 1, 0, 1], dtype=np.double)))

        x = np.array([2, 2, 2, 2], dtype=np.double)
        ub = np.array([-5, 0, np.inf, 2], dtype=np.double)
        process_init_duals_ub(x, ub)
        self.assertTrue(np.allclose(x, np.array([2, 2, 0, 2], dtype=np.double)))

        
class TestFractionToTheBoundary(unittest.TestCase):
    def test_fraction_to_the_boundary_helper_lb(self):
        tau = 0.9
        x = np.array([0, 0, 0, 0], dtype=np.double)
        xl = np.array([-np.inf, -1, -np.inf, -1], dtype=np.double)

        delta_x = np.array([-0.1, -0.1, -0.1, -0.1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([-1, -1, -1, -1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl)
        self.assertAlmostEqual(alpha, 0.9)

        delta_x = np.array([-10, -10, -10, -10], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl)
        self.assertAlmostEqual(alpha, 0.09)

        delta_x = np.array([1, 1, 1, 1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([-10, 1, -10, 1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([-10, -1, -10, -1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl)
        self.assertAlmostEqual(alpha, 0.9)

        delta_x = np.array([1, -10, 1, -1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl)
        self.assertAlmostEqual(alpha, 0.09)

    def test_fraction_to_the_boundary_helper_ub(self):
        tau = 0.9
        x = np.array([0, 0, 0, 0], dtype=np.double)
        xu = np.array([np.inf, 1, np.inf, 1], dtype=np.double)

        delta_x = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([1, 1, 1, 1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
        self.assertAlmostEqual(alpha, 0.9)

        delta_x = np.array([10, 10, 10, 10], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
        self.assertAlmostEqual(alpha, 0.09)

        delta_x = np.array([-1, -1, -1, -1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([10, -1, 10, -1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
        self.assertAlmostEqual(alpha, 1)

        delta_x = np.array([10, 1, 10, 1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
        self.assertAlmostEqual(alpha, 0.9)

        delta_x = np.array([-1, 10, -1, 1], dtype=np.double)
        alpha = _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu)
        self.assertAlmostEqual(alpha, 0.09)
