from typing import Tuple, Sequence, Dict, Union
from numbers import Real

from kronecker.core import Index, Equation, Term, CompositeTerm, RealTerm
from kronecker.backends.base import Backend

import numpy as np

        
def create_index_array(shape: Tuple[int,...], index_dim: int) -> np.ndarray:
    initial_shape = np.ones(len(shape), dtype=np.int32)
    initial_shape[index_dim] = shape[index_dim]
    idxs = np.arange(shape[index_dim]).reshape(initial_shape)
    return np.tile(idxs, tuple(1 if i == index_dim else s for i, s in enumerate(shape)))


def create_index_arrays(indices: Sequence[Index]) -> Dict[Index, np.ndarray]:
    shape = tuple(idx.n for idx in indices)
    return {idx: create_index_array(shape, i) for i, idx in enumerate(indices)}

    
def realise_term(
    term: Term,
    index_values: Dict[Index, np.ndarray]
    ) -> Union[np.ndarray, Real]:
    if isinstance(term, RealTerm):
        return term.value
    elif isinstance(term, Index):
        return index_values[term]
    elif isinstance(term, CompositeTerm):
        return term.operator.value(
            realise_term(term.left, index_values),
            realise_term(term.right, index_values))
    raise ValueError(f"Numpy backend can't realise term {term}")


class NumpyBackend(Backend):
    @staticmethod
    def realise(eq: Equation) -> np.ndarray:
        """Create the matrix represented by eq as a numpy matrix.

        Parameters
        ----------
        eq
            equation to realise

        Returns
        -------
            boolean numpy array
        """
        index_values = create_index_arrays(eq.indices)
        return eq.operator.value(
            realise_term(eq.left, index_values),
            realise_term(eq.right, index_values))
