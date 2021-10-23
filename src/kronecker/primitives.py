import operator as op
from enum import Enum

class ComparisonOperator(Enum):
    EQ = op.eq
    NE = op.ne
    GT = op.gt
    GE = op.ge
    LT = op.lt
    LE = op.le

class BinaryOperator(Enum):
    ADD = op.add
    SUB = op.sub
    MUL = op.mul
    TRUEDIV = op.truediv
    FLOORDIV = op.floordiv
    POW = op.pow