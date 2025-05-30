#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.core.base import ConcreteModel, Var, Constraint, Objective
from pyomo.common.dependencies import attempt_import

np, numpy_available = attempt_import(
    'numpy', 'Interior point requires numpy', minimum_version='1.13.0'
)
scipy, scipy_available = attempt_import('scipy', 'Interior point requires scipy')
mumps, mumps_available = attempt_import('mumps', 'Interior point requires mumps')

if not (numpy_available and scipy_available):
    raise unittest.SkipTest('Interior point tests require numpy and scipy')

from pyomo.contrib.pynumero.asl import AmplInterface

asl_available = AmplInterface.available()
if not asl_available:
    raise unittest.SkipTest('Regularization tests require ASL')
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.interior_point.interior_point import InteriorPointSolver
from pyomo.contrib.interior_point.linalg.mumps_interface import MumpsInterface


def make_model_tri(n, small_val=1e-7, big_val=1e2):
    m = ConcreteModel()
    m.x = Var(range(n), initialize=0.5)

    def c_rule(m, i):
        return big_val * m.x[i - 1] + small_val * m.x[i] + big_val * m.x[i + 1] == 1

    m.c = Constraint(range(1, n - 1), rule=c_rule)

    m.obj = Objective(expr=small_val * sum((m.x[i] - 1) ** 2 for i in range(n)))

    return m


class TestReallocation(unittest.TestCase):
    def _test_ip_with_reallocation(self, linear_solver, interface):
        ip_solver = InteriorPointSolver(
            linear_solver,
            max_reallocation_iterations=3,
            reallocation_factor=1.1,
            # The small factor is to ensure that multiple iterations of
            # reallocation are performed. The bug in the previous
            # implementation only occurred if 2+ reallocation iterations
            # were needed (max_reallocation_iterations >= 3).
            max_iter=1,
        )
        ip_solver.set_interface(interface)

        ip_solver.solve(interface)

        return ip_solver

    @unittest.skipIf(not mumps_available, 'Mumps is not available')
    def test_mumps(self):
        n = 20000
        m = make_model_tri(n, small_val=1e-7)
        interface = InteriorPointInterface(m)
        linear_solver = MumpsInterface()
        # Default memory "buffer" factor: 20
        linear_solver.set_icntl(14, 20)

        kkt = interface.evaluate_primal_dual_kkt_matrix()
        res = linear_solver.do_symbolic_factorization(kkt)
        predicted = linear_solver.get_infog(16)

        linear_solver.set_icntl(23, 8)

        self._test_ip_with_reallocation(linear_solver, interface)
        # In Mumps 5.6.2 (and likely previous versions), ICNTL(23)=0
        # corresponds to "use default increase factor over prediction".
        actual = linear_solver.get_icntl(23)
        percent_increase = linear_solver.get_icntl(14)
        increase_factor = 1.0 + percent_increase / 100.0

        if actual == 0:
            actual = increase_factor * predicted

        # As of Mumps 5.6.2, predicted == 9, which is lower than the
        # default actual of 10.8
        # self.assertTrue(predicted == 12 or predicted == 11)
        self.assertTrue(actual > predicted)
        # NOTE: This test will break if Mumps (or your Mumps version)
        # gets more conservative at estimating memory requirement,
        # or if the numeric factorization gets more efficient.


if __name__ == '__main__':
    #
    unittest.main()
