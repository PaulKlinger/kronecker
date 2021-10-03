from __future__ import annotations
import enum
from numbers import Number
from typing import Sequence, Tuple, Callable, Dict

import numpy as np


class Operation:
    def __init__(self, indices: Sequence[Index], f: IndexFun):
        self.indices = indices
        self.f = f

    def __call__(self, values: IndexFun) -> np.ndarray:
        return self.f(values)
    
    def __add__(self, other) -> Operation:
        if isinstance(other, Index):
            return Operation(self.indices, lambda v: self(v) + v[other])
        elif isinstance(other, Number):
            return Operation(self.indices, lambda v: self(v) + other)
        elif isinstance(other, Operation):
            return Operation(self.indices, lambda v: self(v) + other(v))

        return NotImplemented

    def __comparison_op(self, other, operator: np.ufunc) -> Equation:
        if isinstance(other, Operation):
            return Equation(self, other, operator)
        elif isinstance(other, Index):
            return Equation(self, Operation(self.indices, lambda v: v[other]), operator)
        elif isinstance(other, Number):
            return Equation(self, Operation(self.indices, lambda v: other), operator)
        
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
        

def create_index_array(shape: Tuple[int,...], index_dim: int) -> np.ndarray:
    initial_shape = np.ones(len(shape), dtype=np.int32)
    initial_shape[index_dim] = shape[index_dim]
    idxs = np.arange(shape[index_dim]).reshape(initial_shape)
    return np.tile(idxs, tuple(1 if i == index_dim else s for i, s in enumerate(shape)))


class Equation:
    def __init__(self, left: Operation, right: Operation, operator: np.ufunc):
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


class Index:
    def __init__(self, n: int):
        self.indices = None
        self.n = n

    def __hash__(self):
        return id(self)
    
    def __add__(self, other) -> Operation:
        if isinstance(other, Index):
            return Operation(self.indices, lambda v: v[self] + v[other])
        elif isinstance(other, Number):
            return Operation(self.indices, lambda v: v[self] + other)

        return NotImplemented

    def __comparison_op(self, other, operator) -> Operation:
        if isinstance(other, Number):
            return Equation(
                Operation(self.indices, lambda v: v[self]),
                Operation(self.indices, lambda v: other),
                operator)
        elif isinstance(other, Index):
            return Equation(
                Operation(self.indices, lambda v: v[self]),
                Operation(self.indices, lambda v: v[other]),
                operator)
        
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


IndexFun = Callable[[Dict[Index, np.ndarray]], np.ndarray]


def dims(shape: Sequence[int]) -> Tuple[Index]:
    idxs = tuple(Index(n) for n in shape)
    for idx in idxs:
        idx.indices = idxs
    return idxs