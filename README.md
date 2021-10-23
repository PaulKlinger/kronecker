# Kronecker

A small library that provides a convenient syntax for creating boolean tensors / sparse matrices.

## Examples

```Python
i, j = kronecker.indices(4, 4)
arr = (i >= j * 2 - 1).to_numpy()

np.array_equal(arr, np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0]
    ]).astype(bool))
> True
```

```Python
i, j, k = kronecker.indices(2, 2, 3)
arr = (i >= j + k - 1).to_numpy()

np.array_equal(arr, np.array([
        [[1, 1, 0], [1, 0, 0]],
        [[1, 1, 1], [1, 1, 0]]
    ]).astype(bool))
> True
```

```Python
# would run out of memory if created as a numpy array
i, j = kronecker.indices(1_000_000, 1_000_000)
x = (i * 5 == j).to_sparse()
assert x.sum() == 200000
```

## Limitations
* When creating sparse matrices linear expressions in the indices are simplified and evaluated once per row, giving a complexity of O(n_rows * n_True_per_row). For non-linear expressions (including ones that contain integer division `//`) a slower, O(n_rows * n_cols), path is used. This is mostly useless, creating a numpy array and converting is much faster (but uses more memory).
* Only the operators {`+`, `-`, `*`, `/`, `//`, `**`} are supported.
* Multiple comparisons (`i < j < 2 * i`) are not supported.
