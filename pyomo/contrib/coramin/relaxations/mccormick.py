import pyomo.environ as pe
from pyomo.core.expr.numeric_expr import (
    NumericExpression, 
    MinExpression, 
    MaxExpression,
    NegationExpression,
    NPV_NegationExpression,
    PowExpression,
    NPV_PowExpression,
    ProductExpression,
    NPV_ProductExpression,
    MonomialTermExpression,
    DivisionExpression,
    NPV_DivisionExpression,
    SumExpression,
    LinearExpression,
    NPV_SumExpression,
    UnaryFunctionExpression,
    NPV_UnaryFunctionExpression,
)
from pyomo.core.base.var import VarData, ScalarVar
from pyomo.core.base.param import ParamData, ScalarParam
from pyomo.common.numeric_types import native_numeric_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, native_numeric_types
from pyomo.contrib.fbbt import interval
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
import math
from scipy.optimize import bisect
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
import warnings


def log(x):
    if type(x) in native_numeric_types and x == 0:
        return -math.inf
    return pe.log(x)


def golden_section_search(func, lb, ub, minimize=True):
    assert lb < math.inf
    assert ub > -math.inf

    x1 = lb
    x4 = ub
    f1 = func(x1)
    f4 = func(x4)

    golden_ratio = (1 + 5**0.5) / 2
    delta = x4 - x1
    tol = 1e-12
    if f1 <= f4:
        if minimize:
            best = x1
        else:
            best = x4
    else:
        if minimize:
            best = x4
        else:
            best = x1
    inf_step = 1
    while delta > tol * min(abs(x1), abs(x4)) or math.isinf(delta):
        if x1 == -math.inf and x4 <= -1e100:
            # assume the answer is -inf
            assert best == x1
            return x1
        if x4 == math.inf and x1 >= 1e100:
            # assume the answer is inf
            assert best == x4
            return x4

        if x1 == -math.inf and x4 == math.inf:
            x2 = -1
            x3 = 1
        elif x1 == -math.inf:
            x3 = x4 - inf_step
            x2 = x3 - inf_step
            inf_step *= 2
        elif x4 == math.inf:
            x2 = x1 + inf_step
            x3 = x2 + inf_step
            inf_step *= 2
        else:
            a = delta - (golden_ratio * delta) / (golden_ratio + 1)
            x2 = x1 + a
            b = x4 - x2
            c = a**2 / b
            x3 = x2 + c
            assert (x4 - x3) / (x3 - x2) == golden_ratio
            assert (x2 - x1) / (x3 - x2) == golden_ratio

        f2 = func(x2)
        f3 = func(x3)

        pts = [(f1, 0), (f2, 1), (f3, 2), (f4, 3)]
        if minimize:
            pts.sort()
        else:
            pts.sort(reverse=True)

        ndx = pts[0][1]
        best = [x1, x2, x3, x4][ndx]
        if ndx == 0 or ndx == 1:
            x4 = x3
            f4 = f3
        else:
            x1 = x2
            f1 = f2
    return best


class MidExpression(NumericExpression):
    __slots__ = ()
    PRECEDENCE = None

    def nargs(self):
        assert len(self._args_) == 3
        return 3
    
    def _apply_operation(self, result):
        s = list(result)
        s.sort()
        assert len(s) == 3
        return s[1]
    
    def getname(self, *args, **kwds):
        return 'mid'
    
    def _to_string(self, values, verbose, smap):
        return f"{self.getname()}({', '.join(values)})"


def handle_sum_expression(node, data, feasibility_tol):
    under = 0
    over = 0
    lb = 0
    ub = 0
    for arg_under, arg_over, arg_lb, arg_ub in data:
        if arg_under is None or arg_over is None:
            return [None, None, None, None]
        under += arg_under
        over += arg_over
        lb, ub = interval.add(lb, ub, arg_lb, arg_ub)
    under = MidExpression((lb, ub, under))
    over = MidExpression((lb, ub, over))
    return [under, over, lb, ub]


def handle_product_expression(node, data, feasibility_tol):
    assert len(data) == 2
    arg1_data, arg2_data = data
    under1, over1, lb1, ub1 = arg1_data
    under2, over2, lb2, ub2 = arg2_data

    lb, ub = interval.mul(lb1, ub1, lb2, ub2)

    alpha1 = MinExpression((lb2*under1, lb2*over1))
    alpha2 = MinExpression((lb1*under2, lb1*over2))

    beta1 = MinExpression((ub2*under1, ub2*over1))
    beta2 = MinExpression((ub1*under2, ub1*over2))

    gamma1 = MaxExpression((lb2*under1, lb2*over1))
    gamma2 = MaxExpression((ub1*under2, ub1*over2))

    delta1 = MaxExpression((ub2*under1, ub2*over1))
    delta2 = MaxExpression((lb1*under2, lb1*over2))

    under = MaxExpression((alpha1 + alpha2 - lb2*lb1, beta1 + beta2 - ub2*ub1))
    over = MinExpression((gamma1 + gamma2 - lb2*ub1, delta1 + delta2 - ub2*lb1))

    under = MidExpression((lb, ub, under))
    over = MidExpression((lb, ub, over))

    return [under, over, lb, ub]


def handle_log_expression(node, data, feasibility_tol):
    if data[0][2] < 0:
        # argument must be nonnegative
        data[0][2] = 0
    return handle_concave_univariate(node=node, func=log, data=data, feasibility_tol=feasibility_tol)


def handle_exp_expression(node, data, feasibility_tol):
    def func(x):
        return pe.exp(x)
    
    return handle_convex_univariate(node=node, func=func, data=data, feasibility_tol=feasibility_tol)


def handle_negation_expression(node, data, feasibility_tol):
    assert len(data) == 1
    arg_under, arg_over, arg_lb, arg_ub = data[0]

    if arg_under is None or arg_over is None:
        return [None, None, None, None]

    lb, ub = interval.sub(0, 0, arg_lb, arg_ub)
    minimizer = arg_ub
    maximizer = arg_lb
    hmin = MidExpression((arg_under, arg_over, minimizer))
    hmax = MidExpression((arg_under, arg_over, maximizer))
    under = -hmin
    over = -hmax
    under = MidExpression((lb, ub, under))
    over = MidExpression((lb, ub, over))
    return [under, over, lb, ub]


def handle_convex_univariate(node, func, data, feasibility_tol):
    assert len(data) == 1
    arg_under, arg_over, arg_lb, arg_ub = data[0]

    if arg_under is None or arg_over is None:
        return [None, None, None, None]

    m = pe.ConcreteModel()
    m.x = pe.Var(lb=arg_lb, ub=arg_ub)
    expr = func(m.x)

    lb, ub = compute_bounds_on_expr(expr)
    if lb is None:
        lb = -math.inf
    if ub is None:
        ub = math.inf

    # see if the function is monotonically increasing or decreasing
    der = reverse_sd(expr)[m.x]
    der_lb, der_ub = compute_bounds_on_expr(der)
    if der_lb is None:
        der_lb = -math.inf
    if der_ub is None:
        der_ub = math.inf

    if der_lb >= 0:
        # monotonically increasing
        minimizer = arg_lb
        maximizer = arg_ub
    elif der_ub <= 0:
        # monotonically decreasing
        minimizer = arg_ub
        maximizer = arg_lb
    else:
        minimizer = golden_section_search(func=func, lb=arg_lb, ub=arg_ub, minimize=True)
        maximizer = golden_section_search(func=func, lb=arg_lb, ub=arg_ub, minimize=False)

    if func(minimizer) > lb:
        lb = func(minimizer)
    if func(maximizer) < ub:
        ub = func(maximizer)

    hmin = MidExpression((arg_under, arg_over, minimizer))
    under = func(hmin)
    under = MidExpression((lb, ub, under))

    hmax = MidExpression((arg_under, arg_over, maximizer))
    if math.isfinite(lb) and math.isfinite(ub) and math.isfinite(arg_lb) and math.isfinite(arg_ub):
        xl, xu = arg_lb, arg_ub
        yl = func(xl)
        yu = func(xu)
        m = (yu - yl) / (xu - xl)
        b = yu - m * xu
        over = m * hmax + b
        over = MidExpression((lb, ub, over))
    else:
        over = ub
    return [under, over, lb, ub]    


def handle_concave_univariate(node, func, data, feasibility_tol):
    assert len(data) == 1
    arg_under, arg_over, arg_lb, arg_ub = data[0]

    if arg_under is None or arg_over is None:
        return [None, None, None, None]

    m = pe.ConcreteModel()
    m.x = pe.Var(lb=arg_lb, ub=arg_ub)
    expr = func(m.x)

    lb, ub = compute_bounds_on_expr(expr)
    if lb is None:
        lb = -math.inf
    if ub is None:
        ub = math.inf

    # see if the function is monotonically increasing or decreasing
    der = reverse_sd(expr)[m.x]
    der_lb, der_ub = compute_bounds_on_expr(der)
    if der_lb is None:
        der_lb = -math.inf
    if der_ub is None:
        der_ub = math.inf

    if der_lb >= 0:
        # monotonically increasing
        minimizer = arg_lb
        maximizer = arg_ub
    elif der_ub <= 0:
        # monotonically decreasing
        minimizer = arg_ub
        maximizer = arg_lb
    else:
        minimizer = golden_section_search(func=func, lb=arg_lb, ub=arg_ub, minimize=True)
        maximizer = golden_section_search(func=func, lb=arg_lb, ub=arg_ub, minimize=False)

    if func(minimizer) > lb:
        lb = func(minimizer)
    if func(maximizer) < ub:
        ub = func(maximizer)

    hmax = MidExpression((arg_under, arg_over, maximizer))
    over = func(hmax)
    over = MidExpression((lb, ub, over))

    hmin = MidExpression((arg_under, arg_over, minimizer))
    if math.isfinite(lb) and math.isfinite(ub) and math.isfinite(arg_lb) and math.isfinite(arg_ub):
        xl, xu = arg_lb, arg_ub
        yl = func(xl)
        yu = func(xu)
        m = (yu - yl) / (xu - xl)
        b = yu - m * xu
        under = m * hmin + b
        under = MidExpression((lb, ub, under))
    else:
        under = lb
    return [under, over, lb, ub]


def handle_pow_positive_base(node, data, feasibility_tol):
    # if the base is positive, we can use a log tranformation
    # exp(arg2 * log(arg1))
    arg1_data, arg2_data = data
    under1, over1, lb1, ub1 = arg1_data
    under2, over2, lb2, ub2 = arg2_data
    arg1, arg2 = node.args

    assert lb1 >= 0

    if under1 is None or over1 is None or under2 is None or over2 is None:
        return [None, None, None, None]
    
    log_arg = pe.log(arg1)
    log_under, log_over, log_lb, log_ub = handle_log_expression(
        node=log_arg,
        data=[(under1, over1, lb1, ub1)], 
        feasibility_tol=feasibility_tol,
    )

    prod_arg = arg2 * log_arg
    prod_under, prod_over, prod_lb, prod_ub = handle_product_expression(
        node=prod_arg,
        data=[(under2, over2, lb2, ub2), (log_under, log_over, log_lb, log_ub)],
        feasibility_tol=feasibility_tol,
    )

    exp_arg = pe.exp(prod_arg)
    exp_under, exp_over, exp_lb, exp_ub = handle_exp_expression(
        node=exp_arg,
        data=[(prod_under, prod_over, prod_lb, prod_ub)],
        feasibility_tol=feasibility_tol,
    )

    return [exp_under, exp_over, exp_lb, exp_ub]


def get_pow_positive_odd_exponent_underestimator(exponent, lb, ub, feasbility_tol):
    # find a line that intersects x**exponent 
    # at x=lb and is tangent to x**exponent 
    # at some point greater than lb
    assert exponent == round(exponent)
    assert exponent % 2 == 1
    assert exponent > 1
    assert lb < 0 < ub

    # lb**exponent = m * lb + b
    # exponent * x ** (exponent - 1) = m
    # x**exponent = m * x + b

    def f(x):
        return x**exponent * (exponent - 1) + lb ** exponent - exponent * x ** (exponent - 1) * lb
    
    xl = 0
    fl = f(xl)
    start_sign = math.copysign(1, fl)
    xu = 1
    fu = f(xu)
    sign = math.copysign(1, fu)
    _iter = 0
    while sign == start_sign:
        xl = xu
        xu *= 2
        fu = f(xu)
        sign = math.copysign(1, fu)
        _iter += 1
        if _iter >= 100:
            raise RuntimeError('Could not find a sign change to start bisection')
    x = bisect(f, xl, xu)
    if x > ub:
        x = ub
    xl = lb
    xu = x
    yl = xl ** exponent
    yu = xu ** exponent
    m = (yu - yl) / (xu - xl)
    b = yl - m * xl
    b -= feasbility_tol  # relax the underestimator slightly because bisection is approximate
    return x, m, b


def get_pow_positive_odd_exponent_overestimator(exponent, lb, ub, feasbility_tol):
    # find a line that intersects x**exponent 
    # at x=ub and is tangent to x**exponent 
    # at some point less than ub
    assert exponent == round(exponent)
    assert exponent % 2 == 1
    assert exponent > 1
    assert lb < 0 < ub

    # ub**exponent = m * ub + b
    # exponent * x ** (exponent - 1) = m
    # x**exponent = m * x + b

    def f(x):
        return x**exponent * (exponent - 1) + ub ** exponent - exponent * x ** (exponent - 1) * ub
    
    xu = 0
    fu = f(xu)
    start_sign = math.copysign(1, fu)
    xl = -1
    fl = f(xl)
    sign = math.copysign(1, fl)
    _iter = 0
    while sign == start_sign:
        xu = xl
        xl *= 2
        fl = f(xl)
        sign = math.copysign(1, fl)
        _iter += 1
        if _iter >= 100:
            raise RuntimeError('Could not find a sign change to start bisection')
    x = bisect(f, xl, xu)
    if x < lb:
        x = lb
    xl = x
    xu = ub
    yl = xl ** exponent
    yu = xu ** exponent
    m = (yu - yl) / (xu - xl)
    b = yl - m * xl
    b += feasbility_tol  # relax the overestimator slightly because bisection is approximate
    return x, m, b


def handle_pow_positive_odd_exponent(node, data, feasibility_tol):
    arg1_data, arg2_data = data
    under1, over1, lb1, ub1 = arg1_data
    under2, over2, lb2, ub2 = arg2_data

    assert lb2 == ub2
    assert lb2 == round(lb2)
    assert lb2 % 2 == 1
    assert lb1 < 0 < ub1

    if under1 is None or over1 is None:
        return [None, None, None, None]
    
    exponent = lb2
    lb, ub = interval.power(lb1, ub1, exponent, exponent, feasibility_tol=feasibility_tol)
    minimizer = lb1
    maximizer = ub1
    hmin = MidExpression((under1, over1, minimizer))
    hmax = MidExpression((under1, over1, maximizer))

    # underestimator
    tangent_point, m, b = get_pow_positive_odd_exponent_underestimator(exponent=exponent, lb=lb1, ub=ub1, feasbility_tol=feasibility_tol)
    under = m * hmin + b
    if tangent_point < ub1:
        under1 = under
        _x = 0.5 * (tangent_point + ub1)
        b = _x ** exponent
        m = exponent * _x ** (exponent - 1)
        under2 = m * (hmin - _x) + b
        _x = ub1
        b = _x ** exponent
        m = exponent * _x ** (exponent - 1)
        under3 = m * (hmin - _x) + b
        under = MaxExpression((under1, under2, under3))
    under = MidExpression((lb, ub, under))

    # overestimator
    tangent_point, m, b = get_pow_positive_odd_exponent_overestimator(exponent=exponent, lb=lb1, ub=ub1, feasbility_tol=feasibility_tol)
    over = m * hmax + b
    if tangent_point > lb1:
        over1 = over
        _x = 0.5 * (tangent_point + lb1)
        b = _x ** exponent
        m = exponent * _x ** (exponent - 1)
        over2 = m * (hmax - _x) + b
        _x = lb1
        b = _x ** exponent
        m = exponent * _x ** (exponent - 1)
        over3 = m * (hmax - _x) + b
        over = MaxExpression((over1, over2, over3))
    over = MidExpression((lb, ub, over))

    return [under, over, lb, ub]


def handle_pow_expression(node, data, feasibility_tol):
    arg1_data, arg2_data = data
    under1, over1, lb1, ub1 = arg1_data
    under2, over2, lb2, ub2 = arg2_data

    if under1 is None or over1 is None or under2 is None or over2 is None:
        return [None, None, None, None]

    if lb2 == ub2:
        exponent = lb2

        def func(x):
            return x ** exponent

        if exponent > 1:
            if lb1 >= 0:
                return handle_convex_univariate(node=node, func=func, data=[arg1_data], feasibility_tol=feasibility_tol)
            elif exponent == round(exponent) and exponent % 2 == 0:
                # exponent is fixed and even
                return handle_convex_univariate(node=node, func=func, data=[arg1_data], feasibility_tol=feasibility_tol)
            elif exponent == round(exponent) and exponent % 2 == 1:
                # exponent is fixed and odd
                if ub1 <= 0:
                    return handle_concave_univariate(node=node, func=func, data=[arg1_data], feasibility_tol=feasibility_tol)
                else:
                    assert lb1 < 0 < ub1
                    return handle_pow_positive_odd_exponent(node=node, data, feasibility_tol)
            else:
                # exponent is fixed, fractional, and greater than 1
                assert exponent != round(exponent)
                # base has to be positive
                data[0][2] = max(0, data[0][2])
                assert data[0][3] >= data[0][2]
                return handle_convex_univariate(node=node, func=func, data=[arg1_data], feasibility_tol=feasibility_tol)
        elif exponent == 1:
            return [under1, over1, lb1, ub1]
        elif exponent > 0:
            # exponent is fixed and between 0 and 1
            # base has to be positive
            data[0][2] = max(0, data[0][2])
            assert data[0][3] >= data[0][2]
            return handle_concave_univariate(node=node, func=func, data=[arg1_data], feasibility_tol=feasibility_tol)
        elif exponent == 0:
            return [1, 1, 1, 1]
        elif exponent > -1:
            # exponent is fixed and between -1 and 0
            # base has to be positive
            data[0][2] = max(0, data[0][2])
            assert data[0][3] >= data[0][2]
            def func(x):
                if type(x) in native_numeric_types and x == 0:
                    return math.inf
                return x ** exponent
            return handle_convex_univariate(node=node, func=func, data=[arg1_data], feasibility_tol=feasibility_tol)
        elif exponent == round(exponent) and exponent % 2 == 1:
            # odd, negative exponent
            if lb1 >= 0:
                def func(x):
                    if type(x) in native_numeric_types and x == 0:
                        return math.inf
                    return x ** exponent
                return handle_convex_univariate(node=node, func=func, data=[arg1_data], feasibility_tol=feasibility_tol)
            elif ub1 <= 0:
                def func(x):
                    if type(x) in native_numeric_types and x == 0:
                        return -math.inf
                    return x ** exponent
                return handle_concave_univariate(node=node, func=func, data=[arg1_data], feasibility_tol=feasibility_tol)
            else:
                return [-math.inf, math.inf, -math.inf, math.inf]
        elif exponent == round(exponent) and exponent % 2 == 0:
            # even, negative exponent
            def func(x):
                if type(x) in native_numeric_types and x == 0:
                    return math.inf
                return x ** exponent
            if lb1 >= 0:
                return handle_convex_univariate(node=node, func=func, data=[arg1_data], feasibility_tol=feasibility_tol)
            elif ub1 <= 0:
                return handle_convex_univariate(node=node, func=func, data=[arg1_data], feasibility_tol=feasibility_tol)
            else:
                # get a lower bound
                m = pe.ConcreteModel()
                m.x = pe.Var(lb=lb1, ub=ub1)
                expr = m.x ** exponent
                lb, ub = compute_bounds_on_expr(expr)
                return [lb, ub, lb, ub]
        else:
            # negative fractional exponent
            assert exponent != round(exponent)
            assert exponent < 0
            def func(x):
                if type(x) in native_numeric_types and x == 0:
                    return math.inf
                return x ** exponent
            return handle_convex_univariate(node=node, func=func, data=[arg1_data], feasibility_tol=feasibility_tol)
    else:
        # variable exponent
        # assume the base has to be positive
        if lb1 < 0:
            msg = f"found a variable exponent with a base that is potentially negative: {str(node)}; assuming the base is positive"
            data[0][2] = max(0, data[0][2])
            assert data[0][3] >= data[0][2]
        return handle_pow_positive_base(node=node, data=data, feasibility_tol=feasibility_tol)


def handle_float(node, data, feasibility_tol):
    val = float(node)
    return [val, val, val, val]


def handle_param(node, data, feasibility_tol):
    return [node.value, node.value, node.value, node.value]


def handle_var(node, data, feasibility_tol):
    if node.is_fixed():
        return [node.value, node.value, node.value, node.value]
    else:
        under = node
        over = node
        lb = node.lb
        if lb is None:
            lb = -math.inf
        ub = node.ub
        if ub is None:
            ub = math.inf
        return [node, node, lb, ub]


unary_handlers = {
    'exp': handle_exp_expression,
    'log': handle_log_expression,
}


def handle_univariate_expression(node, data, feasibility_tol):
    if node.getname() in unary_handlers:
        return unary_handlers[node.getname()](node, data, feasibility_tol)
    else:
        raise NotImplementedError(f'cannot yet generate relaxations for {node.getname()} expressions')


handlers = {
    NegationExpression: handle_negation_expression,
    NPV_NegationExpression: handle_negation_expression,
    PowExpression: handle_pow_expression,
    NPV_PowExpression: handle_pow_expression,
    ProductExpression: handle_product_expression,
    NPV_ProductExpression: handle_product_expression,
    MonomialTermExpression: handle_product_expression,
    SumExpression: handle_sum_expression,
    LinearExpression: handle_sum_expression,
    NPV_SumExpression: handle_sum_expression,
    UnaryFunctionExpression: handle_univariate_expression,
    NPV_UnaryFunctionExpression: handle_univariate_expression,
    VarData: handle_var,
    ScalarVar: handle_var,
    ParamData: handle_param,
    ScalarParam: handle_param,
}


for t in native_numeric_types:
    handlers[t] = handle_float


class RelaxationVisitor(StreamBasedExpressionVisitor):
    def __init__(self, feasibility_tol=1e-8, **kwds):
        self.feasibility_tol = feasibility_tol
        super().__init__(**kwds)

    def exitNode(self, node, data):
        node_type = type(node)
        if node_type not in handlers:
            if node_type in native_numeric_types:
                handlers[node_type] = handle_float
            else:
                raise NotImplementedError(f'cannot yet generate relaxations for {node_type} expressions')
        return handlers[node_type](node, data, self.feasibility_tol)
    

relaxation_visitor = RelaxationVisitor()


def generate_relaxation(expr: NumericExpression, feasibility_tol=1e-8):
    visitor = relaxation_visitor
    orig_feasibility_tol = visitor.feasibility_tol
    visitor.feasibility_tol = feasibility_tol
    under, over, lb, ub = visitor.walk_expression(expr)
    visitor.feasibility_tol = orig_feasibility_tol
    return under, over, lb, ub