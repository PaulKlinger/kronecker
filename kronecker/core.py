from __future__ import annotations
import abc
from numbers import Number
from typing import Sequence, Tuple, Dict, Union

import numpy as np


class Term(abc.ABC):
    @property
    def shape(self):
        return tuple(i.n for i in self.indices)
    
    def __comparison_op(self, other, operator: np.ufunc) -> Equation:
        if isinstance(other, Term):
            return Equation(self, other, operator)
        elif isinstance(other, Number):
            return Equation(self, NumberTerm(other, self.indices), operator)
        
        return NotImplemented

    def __eq__(self, other) -> Equation:
        return self.__comparison_op(other, np.equal)
    
    def __neq__(self, other) -> Equation:
        return self.__comparison_op(other, np.not_equal)
    
    def __gt__(self, other) -> Equation:
        return self.__comparison_op(other, np.greater)
    
    def __ge__(self, other) -> Equation:
        return self.__comparison_op(other, np.greater_equal)
    
    def __lt__(self, other) -> Equation:
        return self.__comparison_op(other, np.less)
    
    def __le__(self, other) -> Equation:
        return self.__comparison_op(other, np.less_equal)

    def __binary_op(self, other, operator: np.ufunc) -> CompositeTerm:
        if isinstance(other, Number):
            return CompositeTerm(self.indices, self, NumberTerm(other, self.indices), operator)
        elif isinstance(other, Term):
            return CompositeTerm(self.indices, self, other, operator)

        return NotImplemented

    def __add__(self, other) -> CompositeTerm:
        return self.__binary_op(other, np.add)

    def __sub__(self, other) -> CompositeTerm:
        return self.__binary_op(other, np.subtract)

    def __mul__(self, other) -> CompositeTerm:
        return self.__binary_op(other, np.multiply)
    
    def __pow__(self, other) -> CompositeTerm:
        return self.__binary_op(other, np.power)

    def __floordiv__(self, other) -> CompositeTerm:
        return self.__binary_op(other, np.floor_divide)

    def __truediv__(self, other) -> None:
        raise NotImplementedError("True division is not available, use // for integer division.")


class NumberTerm(Term):
    def __init__(self, value: Number, indices: Sequence[Index]):
        self.value = value
        self.indices = indices


class Index(Term):
    def __init__(self, n: int):
        self.indices = None
        self.n = n

    def __hash__(self):
        return id(self)


class CompositeTerm(Term):
    def __init__(self, indices: Sequence[Index], left: Union[Number, Term], right: Union[Number, Term], operator=np.ufunc):
        self.indices = indices
        self.left = left
        self.right = right
        self.operator = operator


class Equation:
    def __init__(self, left: Term, right: Term, operator: np.ufunc):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch: {left.shape}, {right.shape}")
        elif left.indices != right.indices:
            raise ValueError(f"Identity mismatch, all indices must be created in the same kronecker.dims call!")
            
        self.indices = left.indices
        self.left = left
        self.right = right
        self.operator = operator
