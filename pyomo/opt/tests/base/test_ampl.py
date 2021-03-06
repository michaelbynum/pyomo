#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for pyomo.opt.problem.ampl
#

from itertools import zip_longest
import json
import os
import sys
from os.path import abspath, dirname, join
currdir = dirname(abspath(__file__))+os.sep

from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.errors import ApplicationError

import pyomo.opt

old_tempdir = TempfileManager.tempdir

def filter(text):
    return 'Problem:' in text or text.startswith('NAME')

def filter_nl(text):
    return '# problem'

solver = None
class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ
        solvers = pyomo.opt.check_available_solvers('glpk')

    def setUp(self):
        TempfileManager.tempdir = currdir

    def tearDown(self):
        TempfileManager.clear_tempfiles()
        TempfileManager.tempdir = old_tempdir

    def test3_write_nl(self):
        """ Convert from AMPL to NL """
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3.mod'))
        """ Convert from MOD+DAT to NL """
        try:
            self.model.write(join(currdir, 'test3.nl'))
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail("Unexpected ApplicationError - ampl is enabled "
                          "but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail("Unexpected ConverterError - ampl is enabled "
                          "but not available: '%s'" % str(err))
            return
        _out, _log = join(currdir, 'test3.nl'), join(currdir, 'test3.baseline.nl')
        self.assertTrue(cmp(_out, _log),
                        msg="Files %s and %s differ" % (_out, _log))

    def test3_write_lp(self):
        """ Convert from AMPL to LP """
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3.mod'))
        try:
            self.model.write(join(currdir, 'test3.lp'))
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("glpsol"):
                self.fail("Unexpected ApplicationError - glpsol is enabled "
                          "but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("glpsol"):
                self.fail("Unexpected ConverterError - glpsol is enabled "
                          "but not available: '%s'" % str(err))
            return
        with open(join(currdir, 'test3.lp'), 'r') as run, \
            open(join(currdir, 'test3.baseline.lp'), 'r') as baseline:
                for line1, line2 in zip_longest(run, baseline):
                    if 'Problem' not in line1:
                        self.assertEqual(line1, line2)

    def test3_write_mps(self):
        """ Convert from AMPL to MPS """
        if not pyomo.common.Executable("ampl"):
            self.skipTest("The ampl executable is not available")
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3.mod'))
        try:
            self.model.write(join(currdir, 'test3.mps'))
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail("Unexpected ApplicationError - ampl is enabled "
                          "but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail("Unexpected ConverterError - ampl is enabled "
                          "but not available: '%s'" % str(err))
            return
        _out, _log = join(currdir, 'test3.mps'), join(currdir, 'test3.baseline.mps')
        self.assertTrue(cmp(_out, _log),
                        msg="Files %s and %s differ" % (_out, _log))

    def test3a_write_nl(self):
        """ Convert from AMPL to NL """
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3a.mod'), join(currdir, 'test3a.dat'))
        try:
            self.model.write(join(currdir, 'test3a.nl'))
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail("Unexpected ApplicationError - ampl is enabled "
                          "but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail("Unexpected ConverterError - ampl is enabled "
                          "but not available: '%s'" % str(err))
            return
        _out, _log = join(currdir, 'test3a.nl'), join(currdir, 'test3.baseline.nl')
        self.assertTrue(cmp(_out, _log),
                        msg="Files %s and %s differ" % (_out, _log))

    def test3a_write_lp(self):
        """ Convert from AMPL to LP """
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3a.mod'), join(currdir, 'test3a.dat'))
        try:
            self.model.write(join(currdir, 'test3a.lp'))
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("glpsol"):
                self.fail("Unexpected ApplicationError - glpsol is enabled "
                          "but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("glpsol"):
                self.fail("Unexpected ConverterError - glpsol is enabled "
                          "but not available: '%s'" % str(err))
            return
        with open(join(currdir, 'test3a.lp'), 'r') as run, \
            open(join(currdir, 'test3.baseline.lp'), 'r') as baseline:
                for line1, line2 in zip_longest(run, baseline):
                    if 'Problem' not in line1:
                        self.assertEqual(line1, line2)

    def test3a_write_mps(self):
        """ Convert from AMPL to MPS """
        if not pyomo.common.Executable("ampl"):
            self.skipTest("The ampl executable is not available")
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3a.mod'), join(currdir, 'test3a.dat'))
        try:
            self.model.write(join(currdir, 'test3a.mps'))
        except ApplicationError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail("Unexpected ApplicationError - ampl is enabled "
                          "but not available: '%s'" % str(err))
            return
        except pyomo.opt.ConverterError:
            err = sys.exc_info()[1]
            if pyomo.common.Executable("ampl"):
                self.fail("Unexpected ConverterError - ampl is enabled "
                          "but not available: '%s'" % str(err))
            return
        _out, _log = join(currdir, 'test3a.mps'), join(currdir, 'test3.baseline.mps')
        self.assertTrue(cmp(_out, _log),
                        msg="Files %s and %s differ" % (_out, _log))

    def test3_solve(self):
        if not 'glpk' in solvers:
            self.skipTest("glpk solver is not available")
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3.mod'))
        opt = pyomo.opt.SolverFactory('glpk')
        results = opt.solve(self.model, keepfiles=False)
        results.write(filename=join(currdir, 'test3.out'), format='json')
        with open(join(currdir,"test3.out"), 'r') as out, \
            open(join(currdir,"test3.baseline.out"), 'r') as txt:
            self.assertStructuredAlmostEqual(json.load(txt), json.load(out),
                                             abstol=1e-6,
                                             allow_second_superset=True)

    def test3a_solve(self):
        if not 'glpk' in solvers:
            self.skipTest("glpk solver is not available")
        self.model = pyomo.opt.AmplModel(join(currdir, 'test3a.mod'), join(currdir, 'test3a.dat'))
        opt = pyomo.opt.SolverFactory('glpk')
        results = opt.solve(self.model, keepfiles=False)
        results.write(filename=join(currdir, 'test3a.out'), format='json')
        with open(join(currdir,"test3a.out"), 'r') as out, \
            open(join(currdir,"test3.baseline.out"), 'r') as txt:
            self.assertStructuredAlmostEqual(json.load(txt), json.load(out),
                                             abstol=1e-6,
                                             allow_second_superset=True)


if __name__ == "__main__":
    unittest.main()

