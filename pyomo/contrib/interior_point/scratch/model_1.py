from pyomo.environ import *
#from pyomo.contrib.pynumero.interfaces import PyomoNLP

m = ConcreteModel()

m.v = Var(range(1,11), initialize=1, domain=NonNegativeReals)

@m.Constraint(range(1,6))
def eq_con(m, i):
    i1 = i
    i2 = i + 3
    return 1 == 2*m.v[i1] + 3*m.v[i2]

@m.Constraint(range(6, 11))
def ineq_con(m, i):
    i1 = i
    i2 = i-4
    a1 = i1*10000 % 7
    a2 = i2*10000 % 7
    num = (i1 + i2)*100000 % 7
    return a1*m.v[i1] + a2*m.v[i2] >= num

obj_expr = sum((m.v[i] - i/100)**2 for i in range(1,11))
m.obj = Objective(expr=obj_expr)

solver = SolverFactory('ipopt')
solver.solve(m, tee=True)
