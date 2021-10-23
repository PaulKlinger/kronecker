from __future__ import annotations
import abc
from numbers import Real
from typing import Sequence, Tuple, Dict, Any, Union, Optional, List
import operator as op

from kronecker.primitives import ComparisonOperator, BinaryOperator


class Term(abc.ABC):
    def __init__(self, indices: Sequence[Index]):
        self.indices: Tuple[Index, ...] = tuple(indices)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(i.n for i in self.indices)

    def __comparison_op(self, other: Any, operator: ComparisonOperator) -> Equation:
        if isinstance(other, Term):
            return Equation(self, other, operator)
        elif isinstance(other, Real):
            return Equation(self, RealTerm(other, self.indices), operator)

        return NotImplemented

    # mypy complains that return type doesn't match that of the superclass (object),
    # which is bool.
    def __eq__(self, other: Any) -> Equation:  # type: ignore
        return self.__comparison_op(other, ComparisonOperator.EQ)

    def __ne__(self, other: Any) -> Equation: # type: ignore
        return self.__comparison_op(other, ComparisonOperator.NE)

    def __gt__(self, other: Any) -> Equation:
        return self.__comparison_op(other, ComparisonOperator.GT)

    def __ge__(self, other: Any) -> Equation:
        return self.__comparison_op(other, ComparisonOperator.GE)

    def __lt__(self, other: Any) -> Equation:
        return self.__comparison_op(other, ComparisonOperator.LT)

    def __le__(self, other: Any) -> Equation:
        return self.__comparison_op(other, ComparisonOperator.LE)

    def __binary_op(self, other: Any, operator: BinaryOperator) -> CompositeTerm:
        if isinstance(other, Real):
            return CompositeTerm(
                self.indices, self, RealTerm(other, self.indices), operator
            )
        elif isinstance(other, Term):
            return CompositeTerm(self.indices, self, other, operator)

        return NotImplemented

    def __add__(self, other: Any) -> CompositeTerm:
        return self.__binary_op(other, BinaryOperator.ADD)

    def __radd__(self, other: Any) -> CompositeTerm:
        return self + other

    def __sub__(self, other: Any) -> CompositeTerm:
        return self.__binary_op(other, BinaryOperator.SUB)

    def __rsub__(self, other: Any) -> CompositeTerm:
        return other + (-self)

    def __mul__(self, other: Any) -> CompositeTerm:
        return self.__binary_op(other, BinaryOperator.MUL)

    def __rmul__(self, other: Any) -> CompositeTerm:
        return self * other

    def __pow__(self, other: Any) -> CompositeTerm:
        return self.__binary_op(other, BinaryOperator.POW)

    def __floordiv__(self, other: Any) -> CompositeTerm:
        return self.__binary_op(other, BinaryOperator.FLOORDIV)

    def __rfloordiv__(self, other: Any) -> CompositeTerm:
        return self // other

    def __truediv__(self, other: Any) -> CompositeTerm:
        return self.__binary_op(other, BinaryOperator.TRUEDIV)

    def __rtruediv__(self, other: Any) -> CompositeTerm:
        return self / other

    def __neg__(self) -> CompositeTerm:
        return -1 * self


class RealTerm(Term):
    def __init__(self, value: Real, indices: Sequence[Index]):
        super().__init__(indices)
        self.value = value


class Index(Term):
    def __init__(self, n: int):
        # indices are updated later, once they are all instantiated
        self.indices: Tuple[Index, ...] = (self,)
        self.n = n

    def __hash__(self) -> int:
        return id(self)


class CompositeTerm(Term):
    def __init__(
        self,
        indices: Sequence[Index],
        left: Term,
        right: Term,
        operator: BinaryOperator,
    ):
        super().__init__(indices)
        self.left = left
        self.right = right
        self.operator = operator


class Equation:
    def __init__(self, left: Term, right: Term, operator: ComparisonOperator):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch: {left.shape}, {right.shape}")
        elif left.indices != right.indices:
            raise ValueError(
                f"Identity mismatch, all indices must be created in the same kronecker.indices call!"
            )

        self.indices = left.indices
        self.left = left
        self.right = right
        self.operator = operator
        self.shape = left.shape
