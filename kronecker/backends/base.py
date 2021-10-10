import abc
from typing import Any

from kronecker.core import Equation

class Backend(abc.ABC):
    @abc.abstractmethod
    def realise(eq: Equation) -> Any:
        pass