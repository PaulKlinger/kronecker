from kronecker.api import *

import kronecker.core as core
import kronecker.backends as backends

# not sure how to handle the typechecking for this...
core.Equation.toarray = backends.NumpyBackend.realise # type: ignore
core.Equation.tosparse = backends.ScipySparseBackend.realise # type: ignore
