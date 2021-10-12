from typing import Sequence, Tuple

from kronecker.core import Index

def dims(shape: Sequence[int]) -> Tuple[Index,...]:
    idxs = tuple(Index(n) for n in shape)
    for idx in idxs:
        idx.indices = idxs
    return idxs