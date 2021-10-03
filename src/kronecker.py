from __future__ import annotations
import abc
from numbers import Number
from typing import Sequence, Tuple, Callable, Dict

import numpy as np


class Term(abc.ABC):
    @abc.abstractmethod
    def __call__(self, values: IndexFun) -> np.ndarray:
        pass
    
    def __comparison_op(self, other, operator: np.ufunc) -> Equation:
        if isinstance(other, Term):
            return Equation(self, other, operator)
        elif isinstance(other, Number):
            return Equation(self, CompositeTerm(self.indices, lambda v: other), operator)
        
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
            return CompositeTerm(self.indices, lambda v: operator(self(v), other))
        elif isinstance(other, Term):
            return CompositeTerm(self.indices, lambda v: operator(self(v), other(v)))

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
        

class Index(Term):
    def __init__(self, n: int):
        self.indices = None
        self.n = n

    def __hash__(self):
        return id(self)

    def __call__(self, values: IndexFun) -> np.ndarray:
        return values[self]


IndexFun = Callable[[Dict[Index, np.ndarray]], np.ndarray]


class CompositeTerm(Term):
    def __init__(self, indices: Sequence[Index], f: IndexFun):
        self.indices = indices
        self.f = f

    def __call__(self, values: IndexFun) -> np.ndarray:
        return self.f(values)
        

def create_index_array(shape: Tuple[int,...], index_dim: int) -> np.ndarray:
    initial_shape = np.ones(len(shape), dtype=np.int32)
    initial_shape[index_dim] = shape[index_dim]
    idxs = np.arange(shape[index_dim]).reshape(initial_shape)
    return np.tile(idxs, tuple(1 if i == index_dim else s for i, s in enumerate(shape)))


class Equation:
    def __init__(self, left: Term, right: Term, operator: np.ufunc):
        if left.indices != right.indices:
            raise ValueError(f"Shape mismatch: {left.shape}, {right.shape}")
            
        self.indices = left.indices
        self.left = left
        self.right = right
        self.operator = operator

    def _create_index_arrays(self) -> Dict[Index, np.ndarray]:
        shape = tuple(idx.n for idx in self.indices)
        return {idx: create_index_array(shape, i) for i, idx in enumerate(self.indices)}

    def toarray(self):
        index_values = self._create_index_arrays()
        return self.operator(self.left(index_values), self.right(index_values))


def dims(shape: Sequence[int]) -> Tuple[Index]:
    idxs = tuple(Index(n) for n in shape)
    for idx in idxs:
        idx.indices = idxs
    return idxs