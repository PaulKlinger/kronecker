import numpy as np
import scipy.sparse as sparse
import pytest

import kronecker


def test_diag():
    expected = np.eye(4)
    i, j = kronecker.indices(4, 4)
    res = (i == j).to_sparse()

    np.testing.assert_array_equal(res.todense(), expected)


def test_add_operation():
    expected = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    i, j = kronecker.indices(3, 3)
    res = (i + 1 == j).to_sparse()

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
    i, j = kronecker.indices(6, 6)
    res = (i * 1.3 > j).to_sparse()

    np.testing.assert_array_equal(res.todense(), expected)


def test_sparsity():
    # would run out of memory if not sparse
    i, j = kronecker.indices(1000000, 1000000)
    x = (i * 5 == j).to_sparse()
    assert x.sum() == 200000


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
    i, j = kronecker.indices(rows, cols)
    eq = eval(eq_str)
    res_numpy = eq.to_numpy()
    res_sparse = eq.to_sparse().todense()
    np.testing.assert_array_equal(res_sparse, res_numpy)


@pytest.mark.parametrize("rows, cols, eq_str", [
    (100, 100, "i == j ** 2"),
    (100, 100, "i // 3 + j * i == 8 * j"),
    (100, 100, "j / 5 > i / (j + 1)")
])
def test_sparse_slow_path_against_numpy(rows, cols, eq_str):
    i, j = kronecker.indices(rows, cols)
    eq = eval(eq_str)
    res_numpy = eq.to_numpy()
    res_sparse = eq.to_sparse().todense()
    np.testing.assert_array_equal(res_sparse, res_numpy)