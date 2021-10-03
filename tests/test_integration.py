import numpy as np

import src.kronecker as kronecker


def test_diag():
    expected = np.eye(4)
    i, j = kronecker.dims((4, 4))
    res = (i == j).toarray()

    np.testing.assert_array_equal(res, expected)


def test_add_operation():
    expected = np.array([
        [[0], [1], [0]],
        [[0], [0], [1]],
        [[0], [0], [0]]
    ])
    i, j, k = kronecker.dims((3, 3, 1))
    res = (i + 1 == j).toarray()

    np.testing.assert_array_equal(res, expected)


def test_extract_row():
    expected = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ])
    i, j = kronecker.dims((3, 3))
    res = (i == 1).toarray()

    np.testing.assert_array_equal(res, expected)