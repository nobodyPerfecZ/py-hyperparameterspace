from typing import Union, Iterable, Any
import numpy as np

from PyHyperparameterSpace.dist.abstract_dist import Distribution
from PyHyperparameterSpace.hp.abstract_hp import Hyperparameter


class Constant(Hyperparameter):
    """
    Abstract class to represent a constant hyperparameter, where the given default value does not get changed by the
    sampling procedure.

        Attributes:
            name (str): name of the hyperparameter
            default (Any): default value of the hyperparameter
            shape (tuple[int]): shape of the hyperparameter
    """

    def __init__(
            self,
            name: str,
            default: Any,
            shape: Union[int, tuple[int, ...], None] = None,
    ):
        super().__init__(name=name, shape=shape, bounds=None, choices=None, default=default, distribution=None,
                         weights=None)

    def _check_bounds(self, bounds: Union[tuple[int, int], tuple[float, float], None]) \
            -> Union[tuple[int, int], tuple[float, float], None]:
        # Does not need to check bounds, because constant HPs does not use them
        return bounds

    def _is_legal_bounds(self, bounds: Union[tuple[int, int], tuple[float, float], None]) -> bool:
        raise Exception("Constant hyperparameter does not have bounds!")

    def _check_choices(self, choices: Union[list[Any], None]) -> Union[list[Any], None]:
        # Does not need to check bounds, because constant HPs does not use them
        return choices

    def _is_legal_choices(self, choices: Union[list[Any], None]) -> bool:
        raise Exception("Constant hyperparameter does not have choices!")

    def _check_default(self, default: Any) -> Any:
        if self._is_legal_default(default):
            return default
        else:
            raise Exception(f"Illegal default value {default}")

    def _is_legal_default(self, default: Any) -> bool:
        if default is None:
            return False
        return True

    def _check_shape(self, shape: Union[int, tuple[int, ...], None]) -> Union[int, tuple[int, ...], None]:
        if shape is None:
            # Case: Adjust the shape according to the given default value
            if isinstance(self._default, (int, float, bool, str)):
                return (1,)
            elif isinstance(self._default, np.ndarray):
                return self._default.shape
        if self._is_legal_shape(shape):
            return shape
        else:
            raise Exception(f"Illegal shape {shape}!")

    def _is_legal_shape(self, shape: Union[int, tuple[int, ...]]) -> bool:
        if shape == 1 or shape == (1,):
            # Check if shape has the right format for the default value
            if isinstance(self._default, (int, float, bool, str)):
                return True
        elif isinstance(shape, int):
            # Check if shape has the right format for the default value
            if isinstance(self._default, np.ndarray) and shape == len(self._default):
                return True
        elif isinstance(shape, tuple) and all(isinstance(s, int) for s in shape):
            # Check if shape is in the right format for the default value
            if isinstance(self._default, np.ndarray) and shape == self._default.shape:
                return True
        return False

    def _check_distribution(self, distribution: Union[Distribution, None]) -> Union[Distribution, None]:
        return distribution

    def _is_legal_distribution(self, distribution: Union[Distribution, None]) -> bool:
        raise Exception("Constant hyperparameter does not have distribution!")

    def _check_weights(self, weights: Union[list[int], list[float], None]) -> Union[list[int], list[float], None]:
        return weights

    def _is_legal_weights(self, weights: Union[list[int], list[float], None]) -> bool:
        raise Exception("Constant hyperparameter does not have weights!")

    def sample(self, random: np.random.RandomState, size: Union[int, None] = None) -> Any:
        sample_size = Constant._get_sample_size(size=size, shape=self._shape)
        if sample_size is None or sample_size == 1:
            return self._default
        else:
            return np.full(shape=sample_size, fill_value=self._default)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        text = f"Constant({self._name}, default={self._default}, shape={self._shape})"
        return text
