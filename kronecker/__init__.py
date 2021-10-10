from kronecker.api import *

import kronecker.core as core
import kronecker.backends as backends

core.Equation.toarray = backends.NumpyBackend.realise
