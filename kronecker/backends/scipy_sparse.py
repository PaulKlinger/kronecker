from typing import cast, Union, Dict, Optional, Literal, Tuple, List, Callable
from numbers import Real
from math import ceil
from itertools import chain
from collections import defaultdict
import operator as op

import scipy.sparse as sparse

from kronecker.backends.base import Backend
from kronecker.core import Equation, Term, RealTerm, CompositeTerm, Index
from kronecker.primitives import BinaryOperator, ComparisonOperator


LinearIndexExpression = defaultdict[Optional[Index], float]

non_linear_error = NotImplementedError("The scipy.sparse backend currently only supports linear expressions in the indices!")

INVERSE_OPERATOR = {
    ComparisonOperator.EQ: ComparisonOperator.EQ,
    ComparisonOperator.NE: ComparisonOperator.NE,
    ComparisonOperator.GT: ComparisonOperator.LT,
    ComparisonOperator.GE: ComparisonOperator.LE,
    ComparisonOperator.LT: ComparisonOperator.GT,
    ComparisonOperator.LE: ComparisonOperator.GE
}

def realise_term(
    term: Union[Term, Real]
    ) -> LinearIndexExpression:
    if isinstance(term, Real):
        return cast(LinearIndexExpression, defaultdict(int, {None: term}))
    elif isinstance(term, RealTerm):
        return cast(LinearIndexExpression, defaultdict(int, {None: term.value}))
    elif isinstance(term, Index):
        return cast(LinearIndexExpression, defaultdict(int, {term: 1, None: 0}))
    elif isinstance(term, CompositeTerm):
        if term.operator is BinaryOperator.POW:
            raise non_linear_error

        left = realise_term(term.left)
        right = realise_term(term.right)
        combined_keys = set(left) | set(right)
        if term.operator in (BinaryOperator.ADD, BinaryOperator.SUB):
            return cast(LinearIndexExpression, defaultdict(int, {k: term.operator.value(left[k], right[k]) for k in combined_keys}))
        elif term.operator in (BinaryOperator.MUL, BinaryOperator.TRUEDIV):
            factor: float
            if list(left) == [None]:
                factor = left[None]
                base = right
            elif list(right) == [None]:
                factor = right[None]
                base = left
            else:
                raise non_linear_error

            return defaultdict(int, {k: term.operator.value(base[k], factor) for k in base.keys()})
        else:
            raise NotImplementedError(f"Operator {term.operator} is not supported by scipy.sparse backend!")

    raise ValueError(f"Numpy backend can't realise term {term}")


def get_build_fun(operator: ComparisonOperator, cols: int, a: float, b: float) -> Callable[..., Tuple[List[int], List[Literal[True]]]]:
    if operator is ComparisonOperator.EQ:
        return lambda row, cols=cols, a=a, b=b: ([x], [True]) if 0 <= (x := a * row + b) < cols and (isinstance(x, int) or x.is_integer()) else ([], [])
    elif operator is ComparisonOperator.NE:
        return lambda row, cols=cols, a=a, b=b: (
            (list(chain(range(int(x)), range(int(x) + 1, cols))),
                [True] * (cols - 1))
            if isinstance(x := a * row + b, int) or x.is_integer()
            else (list(range(cols)), [True] * cols))
    elif operator is ComparisonOperator.GT:
        return lambda row, cols=cols, a=a, b=b: (list(range(x := max(0, int(a * row + b + 1)), cols)),
                            [True] * max(0, cols - x))
    elif operator is ComparisonOperator.GE:
        return lambda row, cols=cols, a=a, b=b: (list(range(x := max(0, int(ceil(a * row + b))), cols)),
                            [True] * max(0, cols - x))
    elif operator is ComparisonOperator.LT:
        return lambda row, cols=cols, a=a, b=b: (list(range(x := min(int(ceil(a * row + b)), cols))),
                            [True] * max(0, x))
    elif operator is ComparisonOperator.LE:
        return lambda row, cols=cols, a=a, b=b: (list(range(x := min(int(a * row + b) + 1, cols))),
                            [True] * max(0, x))
    else:
        raise NotImplementedError(f"Operator {operator} is not supported!")


class ScipySparseBackend(Backend):
    @staticmethod
    def realise(eq: Equation) -> sparse.csr_matrix:
        if len(eq.shape) != 2:
            raise ValueError("Scipy.sparse only supports 2 dimensional matrices!")
        row_index, col_index = eq.indices
        rows, cols = eq.shape

        left = realise_term(eq.left)
        right = realise_term(eq.right)
        # col = a * row + b
        col_mult = left[col_index] - right[col_index]
        if col_mult > 0:
            operator = eq.operator
        else:
            operator = INVERSE_OPERATOR[eq.operator]
        a = (right[row_index] - left[row_index]) / col_mult
        b = (right[None] - left[None]) / col_mult

        build_fun = get_build_fun(operator, cols, a, b)
        lilmatrix = sparse.lil_matrix((rows, cols), dtype=bool)
        for i in range(rows):
            row_indices, row_data = build_fun(i)
            lilmatrix.rows[i] = row_indices
            lilmatrix.data[i] = row_data

        return lilmatrix.tocsr()
