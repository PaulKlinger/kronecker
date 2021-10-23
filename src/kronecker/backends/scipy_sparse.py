from typing import cast, Union, Dict, Optional, Literal, Tuple, List, Callable
from numbers import Real
from math import ceil
from itertools import chain
from collections import defaultdict
import ast
import warnings

import scipy.sparse as sparse

from kronecker.backends.base import Backend
from kronecker.core import Equation, Term, RealTerm, CompositeTerm, Index
from kronecker.primitives import BinaryOperator, ComparisonOperator


LinearIndexExpression = defaultdict[Optional[Index], float]
RowBuildFun = Callable[..., Tuple[List[int], List[Literal[True]]]]


class NonLinearError(NotImplementedError):
    pass


INVERSE_OPERATOR = {
    ComparisonOperator.EQ: ComparisonOperator.EQ,
    ComparisonOperator.NE: ComparisonOperator.NE,
    ComparisonOperator.GT: ComparisonOperator.LT,
    ComparisonOperator.GE: ComparisonOperator.LE,
    ComparisonOperator.LT: ComparisonOperator.GT,
    ComparisonOperator.LE: ComparisonOperator.GE
}

AST_COMPARISON_OP = {
    ComparisonOperator.EQ: ast.Eq,
    ComparisonOperator.NE: ast.NotEq,
    ComparisonOperator.GT: ast.Gt,
    ComparisonOperator.GE: ast.GtE,
    ComparisonOperator.LT: ast.Lt,
    ComparisonOperator.LE: ast.LtE
}

AST_BINARY_OP = {
    BinaryOperator.ADD: ast.Add,
    BinaryOperator.SUB: ast.Sub,
    BinaryOperator.MUL: ast.Mult,
    BinaryOperator.TRUEDIV: ast.Div,
    BinaryOperator.FLOORDIV: ast.FloorDiv,
    BinaryOperator.POW: ast.Pow
}


def get_linear_coefficients(
    term: Term
    ) -> LinearIndexExpression:
    """Get the linear coefficients of the given expression (if it is a linear expression in the indices).
    Raises non_linear_error if the expression is not linear.

    Parameters
    ----------
    term
        Term to get coefficients for.

    Returns
    -------
        Mapping from index to coefficient, None represents constant term.
    """

    if isinstance(term, RealTerm):
        return cast(LinearIndexExpression, defaultdict(int, {None: term.value}))
    elif isinstance(term, Index):
        return cast(LinearIndexExpression, defaultdict(int, {term: 1, None: 0}))
    elif isinstance(term, CompositeTerm):
        if term.operator is BinaryOperator.POW:
            raise NonLinearError()

        left = get_linear_coefficients(term.left)
        right = get_linear_coefficients(term.right)
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
                raise NonLinearError()

            return defaultdict(int, {k: term.operator.value(base[k], factor) for k in base.keys()})
        elif term.operator in (BinaryOperator.POW, BinaryOperator.FLOORDIV):
            raise NonLinearError()
        else:
            raise NotImplementedError(f"Operator {term.operator} is not supported by scipy.sparse backend!")

    raise ValueError(f"Numpy backend can't realise term {term}")


def get_linear_build_fun(
    operator: ComparisonOperator,
    a: float, b: float,
    cols: int
    ) -> RowBuildFun:
    """Get a function to create each row of the (rows, cols) matrix described by the equation
        col_index {operator} {a} * row_index + {b}
    when given row ∈ [0, rows)

    Parameters
    ----------
    operator
        comparison operator in the equation
    a
        coefficient of row index
    b
        constant term
    cols
        number of columns in the output matrix

    Returns
    -------
        Function mapping from row index to row of the matrix.
    """
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


def term_to_ast(term: Term, row_index: Index, col_index: Index) -> ast.expr:
    """Convert the given term to an ast expression

    Parameters
    ----------
    term
    row_index
        Index object that specifies the row location.
    col_index
        Index object that specifies the column location.

    Returns
    -------
        Ast version of term, with row_index replaced by "row" variable
        and col_index replaced by "col" variable.
    """
    if isinstance(term, RealTerm):
        return ast.Constant(term.value)
    elif isinstance(term, Index):
        if term is row_index:
            return ast.Name("row", ctx=ast.Load())
        elif term is col_index:
            return ast.Name("col", ctx=ast.Load())
        else:
            raise ValueError(f"Unidentified index {term}, expected row or column index!")
    elif isinstance(term, CompositeTerm):
        return ast.BinOp(
            term_to_ast(term.left, row_index, col_index),
            AST_BINARY_OP[term.operator](),
            term_to_ast(term.right, row_index, col_index)
            )
    else:
        raise NotImplementedError(f"Unsupported term: {term}")


def get_non_linear_build_fun(eq: Equation, row_index: Index, col_index: Index) -> RowBuildFun:
    """Get a function to create each row of the (rows, cols) matrix described by the equation eq
    when given row ∈ [0, rows).

    Parameters
    ----------
    eq
    row_index
        Index object that specifies the row location.
    col_index
        Index object that specifies the column location.

    Returns
    -------
        Function mapping from row index to tuple of (list of non-zero indices, list of values).
    """
    # we don't want to have a bunch of nested function calls for each entry,
    # so flatten it out by creating an AST for the whole expression
    # and then compiling it into a single lambda
    ast_left = term_to_ast(eq.left, row_index, col_index)
    ast_right = term_to_ast(eq.right, row_index, col_index)
    ast_bool_fun = ast.fix_missing_locations(ast.Expression(
        ast.Lambda(
            ast.arguments(posonlyargs=[], args=[ast.arg(arg="row"), ast.arg(arg="col")],
                          kwonlyargs=[], kw_defaults=[], defaults=[]),
            ast.Compare(ast_left, [AST_COMPARISON_OP[eq.operator]()], [ast_right])
        )
    ))
    ast_lambda = eval(compile(ast_bool_fun, filename="<built_expr>", mode="eval"))
    def build_fun(
        row: int,
        cols: int=eq.shape[1],
        ast_lambda: Callable[[int, int], bool]=ast_lambda
        ) -> Tuple[List[int], List[Literal[True]]]:
        
        non_zero_indices = [col for col in range(cols) if ast_lambda(row, col)]
        return non_zero_indices, [True] * len(non_zero_indices)

    return build_fun


def get_build_fun(eq: Equation) -> RowBuildFun:
    """Get a function to create each row of the (rows, cols) matrix described by the equation eq
    when given row ∈ [0, rows). Uses fast approach for linear equations and slower one
    for non-linear ones.

    Parameters
    ----------
    eq

    Returns
    -------
        Function mapping from row index to (list of non-zero indices, list of values).
    """
    row_index, col_index = eq.indices
    rows, cols = eq.shape

    try:
        left = get_linear_coefficients(eq.left)
        right = get_linear_coefficients(eq.right)
    except NonLinearError:
        warnings.warn(
            "Using slow path, this could take a long time for big matrices!"
            "Using Equation.to_numpy() and converting to sparse matrix will almost always be faster.",
            RuntimeWarning)
        build_fun = get_non_linear_build_fun(eq, row_index, col_index)
    else:
        # we have a linear equation, put it into the form
        # col = a * row + b
        col_mult = left[col_index] - right[col_index]
        if col_mult > 0:
            operator = eq.operator
        else:
            # flip the comparison operator if we divide by a negative value
            operator = INVERSE_OPERATOR[eq.operator]
        a = (right[row_index] - left[row_index]) / col_mult
        b = (right[None] - left[None]) / col_mult

        build_fun = get_linear_build_fun(operator=operator, a=a, b=b, cols=cols)
    return build_fun


class ScipySparseBackend(Backend):
    @staticmethod
    def realise(eq: Equation) -> sparse.csr_matrix:
        """Create the matrix represented by eq as a scipy sparse matrix.
        If eq is purely linear in the indices the equation is simplified
        and execution time is O(n_rows * n_non_zero_entries). Else every
        entry has to be checked separately, i.e. it's O(n_rows * n_cols). 

        Parameters
        ----------
        eq
            equation to realise

        Returns
        -------
            scipy sparse matrix in csr format
        """
        if len(eq.shape) != 2:
            raise ValueError("Scipy.sparse only supports 2 dimensional matrices!")

        rows, cols = eq.shape

        row_build_fun = get_build_fun(eq)

        lilmatrix = sparse.lil_matrix((rows, cols), dtype=bool)
        for i in range(rows):
            row_indices, row_data = row_build_fun(i)
            lilmatrix.rows[i] = row_indices
            lilmatrix.data[i] = row_data

        return lilmatrix.tocsr()
