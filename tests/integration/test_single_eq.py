import numpy as np
import pytest

import kronecker


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


def test_extract_block():
    expected = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [1, 1, 1]
    ])
    i, j = kronecker.dims((3, 3))
    res = (i >= 1).toarray()

    np.testing.assert_array_equal(res, expected)


def test_nested():
    expected = np.array([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0]
    ])

    i, j = kronecker.dims((5, 5))
    res = (i - 3 * (j - 2) >= 0).toarray()

    np.testing.assert_array_equal(res, expected)


def test_operations_1():
    expected = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    i, j = kronecker.dims((5, 5))
    res1 = (i * 4 // 5  - j >= 0).toarray()
    res2 = (0 <= i * 4 // 5  - j).toarray()

    np.testing.assert_array_equal(res1, expected)
    np.testing.assert_array_equal(res2, expected)


def test_pow():
    expected = np.array([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])

    i, j = kronecker.dims((5, 5))
    res = (i ** 2 > j).toarray()
    for r in res:
        print(r)

    np.testing.assert_array_equal(res, expected)


def test_div():
    i, j = kronecker.dims((5, 5))
    with pytest.raises(NotImplementedError):
        (i / 2 > j).toarray()