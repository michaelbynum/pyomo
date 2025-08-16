from enum import Enum


class EigenValueBounder(Enum):
    GERSHGORIN = 'GERSHGORIN'
    GERSHGORIN_WITH_SIMPLIFICATION = 'GERSHGORIN_WITH_SIMPLIFICATION'
    GLOBAL = 'GLOBAL'


class Effort(Enum):
    NONE = 'NONE'
    VERY_LOW = 'VERY_LOW'
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'
    VERY_HIGH = 'VERY_HIGH'


class RelaxationSide(Enum):
    UNDER = 'UNDER'
    OVER = ''
    BOTH = 3


class FunctionShape(Enum):
    LINEAR = 1
    CONVEX = 2
    CONCAVE = 3
    UNKNOWN = 4
