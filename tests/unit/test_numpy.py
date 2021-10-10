import numpy as np

from kronecker.backends.numpy import create_index_array


def test_create_index_array():
    expected = np.array([
        [[0, 0], [1, 1], [2, 2], [3, 3]]
    ] * 3)
    res = create_index_array((3, 4, 2), 1)

    np.testing.assert_array_equal(res, expected)


def test_create_index_array_single():
    expected = np.array([0, 1, 2, 3])
    res = create_index_array((4,), 0)
    np.testing.assert_array_equal(res, expected)