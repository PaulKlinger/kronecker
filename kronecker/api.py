from typing import Tuple

from kronecker.core import Index

def indices(*shape: int) -> Tuple[Index,...]:
    """Create Index objects for tensor of the given shape.
    Use these to construct an equation and realise it, e.g.

    i, j, k = kronecker.indices(2, 2, 3)
    arr = (i >= j + k - 1).to_numpy()
    np.array_equal(arr == np.array([
         [[1, 1, 0], [1, 0, 0]],
         [[1, 1, 1], [1, 1, 0]]
        ]).astype(bool))


    Parameters
    ----------
    shape
        n_dim integers giving the tensor shape

    Returns
    -------
        n Index objects, one for each dimension 
    """
    idxs = tuple(Index(n) for n in shape)
    for idx in idxs:
        idx.indices = idxs
    return idxs