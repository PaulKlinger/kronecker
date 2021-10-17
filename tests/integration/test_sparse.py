import numpy as np
import scipy.sparse as sparse
import pytest

import kronecker


def test_diag():
    expected = np.eye(4)
    i, j = kronecker.dims((4, 4))
    res = (i == j).tosparse()

    np.testing.assert_array_equal(res.todense(), expected)


def test_add_operation():
    expected = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    i, j = kronecker.dims((3, 3))
    res = (i + 1 == j).tosparse()

    np.testing.assert_array_equal(res.todense(), expected)


def test_ineq_float():
    expected = np.array([
       [False, False, False, False, False, False],
       [ True,  True, False, False, False, False],
       [ True,  True,  True, False, False, False],
       [ True,  True,  True,  True, False, False],
       [ True,  True,  True,  True,  True,  True],
       [ True,  True,  True,  True,  True,  True]
    ])
    i, j = kronecker.dims((6, 6))
    res = (i * 1.3 > j).tosparse()

    np.testing.assert_array_equal(res.todense(), expected)


@pytest.mark.parametrize("rows, cols, eq_str", [
    (100, 100, "i == j"),
    (100, 100, "i * 5 - 6 + j * 2 - 3 * i == 8 * j"),
    (100, 100, "j / 5 > i / 13"),
    (100, 100, "j / 5 >= i / 13"),
    (100, 100, "j / 5 < i / 13"),
    (100, 100, "j / 5 <= i / 13"),
    (100, 100, "i != j * 2")
])
def test_sparse_against_numpy(rows, cols, eq_str):
    i, j = kronecker.dims((rows, cols))
    eq = eval(eq_str)
    res_numpy = eq.toarray()
    res_sparse = eq.tosparse().todense()
    np.testing.assert_array_equal(res_sparse, res_numpy)