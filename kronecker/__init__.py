from kronecker.api import *

import kronecker.core as core
import kronecker.backends as backends

# not sure how to handle the typechecking for this...
core.Equation.to_numpy = backends.NumpyBackend.realise # type: ignore
core.Equation.to_sparse = backends.ScipySparseBackend.realise # type: ignore
