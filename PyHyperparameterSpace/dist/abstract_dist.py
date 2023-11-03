from abc import ABC, abstractmethod
from typing import Any, Union, Iterable
import numpy as np


class Distribution(ABC):
    """
    Abstract class of a Distribution (necessary for the sampling procedure).
    """
    @abstractmethod
    def change_distribution(self, **kwargs):
        """
        Changes the distribution to the given parameters.

        Args:
            **kwargs (dict):
                Parameters that defines the distribution
        """
        pass
