from __future__ import annotations
import abc
from numbers import Real
from typing import Sequence, Tuple, Dict, Any, Union, Optional, List

import numpy as np


class Term(abc.ABC):
    def __init__(self, indices: Sequence[Index]):
        self.indices: Tuple[Index, ...] = tuple(indices)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(i.n for i in self.indices)

    def __comparison_op(self, other: Any, operator: np.ufunc) -> Equation:
        if isinstance(other, Term):
            return Equation(self, other, operator)
        elif isinstance(other, Real):
            return Equation(self, RealTerm(other, self.indices), operator)

        return NotImplemented

    # mypy complains that return type doesn't match that of the superclass (object),
    # which is bool.
    def __eq__(self, other: Any) -> Equation:  # type: ignore
        return self.__comparison_op(other, np.equal)

    def __neq__(self, other: Any) -> Equation:
        return self.__comparison_op(other, np.not_equal)

    def __gt__(self, other: Any) -> Equation:
        return self.__comparison_op(other, np.greater)

    def __ge__(self, other: Any) -> Equation:
        return self.__comparison_op(other, np.greater_equal)

    def __lt__(self, other: Any) -> Equation:
        return self.__comparison_op(other, np.less)

    def __le__(self, other: Any) -> Equation:
        return self.__comparison_op(other, np.less_equal)

    def __binary_op(self, other: Any, operator: np.ufunc) -> CompositeTerm:
        if isinstance(other, Real):
            return CompositeTerm(
                self.indices, self, RealTerm(other, self.indices), operator
            )
        elif isinstance(other, Term):
            return CompositeTerm(self.indices, self, other, operator)

        return NotImplemented

    def __add__(self, other: Any) -> CompositeTerm:
        return self.__binary_op(other, np.add)

    def __radd__(self, other: Any) -> CompositeTerm:
        return self + other

    def __sub__(self, other: Any) -> CompositeTerm:
        return self.__binary_op(other, np.subtract)

    def __rsub__(self, other: Any) -> CompositeTerm:
        return self - other

    def __mul__(self, other: Any) -> CompositeTerm:
        return self.__binary_op(other, np.multiply)

    def __rmul__(self, other: Any) -> CompositeTerm:
        return self * other

    def __pow__(self, other: Any) -> CompositeTerm:
        return self.__binary_op(other, np.power)

    def __floordiv__(self, other: Any) -> CompositeTerm:
        return self.__binary_op(other, np.floor_divide)

    def __rfloordiv__(self, other: Any) -> CompositeTerm:
        self // other

    def __truediv__(self, other: Any) -> None:
        raise NotImplementedError(
            "True division is not available, use // for integer division."
        )

    def __rtruediv__(self, other: Any) -> None:
        self / other


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
        left: Union[Real, Term],
        right: Union[Real, Term],
        operator: np.ufunc,
    ):
        super().__init__(indices)
        self.left = left
        self.right = right
        self.operator = operator


class Equation:
    def __init__(self, left: Term, right: Term, operator: np.ufunc):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch: {left.shape}, {right.shape}")
        elif left.indices != right.indices:
            raise ValueError(
                f"Identity mismatch, all indices must be created in the same kronecker.dims call!"
            )

        self.indices = left.indices
        self.left = left
        self.right = right
        self.operator = operator
        self.shape = left.shape
