from abc import ABC, abstractmethod
from typing import Any, Union, Iterable
import numpy as np


class Distribution(ABC):
    """
    Abstract class of a Distribution (necessary for the sampling procedure).
    """
    pass


class Normal(Distribution):
    """
    Class for representing a Normal (Gaussian) Distribution ~N(mean, std).
    """

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __str__(self):
        return f"Normal(loc={self.loc}, scale={self.scale})"

    def __repr__(self):
        return self.__str__()

