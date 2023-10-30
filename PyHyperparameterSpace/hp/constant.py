from typing import Union, Iterable, Any
import numpy as np

from PyHyperparameterSpace.dist.abstract_dist import Distribution
from PyHyperparameterSpace.hp.abstract_hp import Hyperparameter


class Constant(Hyperparameter):
    """
    Class to represent a constant hyperparameter, where the given default value does not get changed by the
    sampling procedure.

        Attributes:
            name (str):
                Name of the hyperparameter

            default (Any):
                Default value of the hyperparameter

            shape (tuple[int]):
                dimensions-space of the hyperparameter
    """

    def __init__(
            self,
            name: str,
            default: Any,
            shape: Union[int, tuple[int, ...], None] = None,
    ):
        super().__init__(name=name, shape=shape, default=default)

    def _check_default(self, default: Any) -> Any:
        if self._is_legal_default(default):
            return default
        else:
            raise Exception(f"Illegal default value {default}")

    def _is_legal_default(self, default: Any) -> bool:
        if default is None:
            return False
        return True

    def _check_shape(self, shape: Union[int, tuple[int, ...], None]) -> Union[int, tuple[int, ...]]:
        if shape is None:
            # Case: Adjust the shape according to the given default value
            if isinstance(self._default, (int, float, bool, str)):
                return (1,)
            elif isinstance(self._default, np.ndarray):
                return self._default.shape
        elif self._is_legal_shape(shape):
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

    def sample(self, random: np.random.RandomState, size: Union[int, None] = None) -> Any:
        sample_size = Constant._get_sample_size(size=size, shape=self._shape)
        if sample_size is None or sample_size == 1:
            return self._default
        else:
            return np.full(shape=sample_size, fill_value=self._default)

    def valid_configuration(self, value: Any) -> bool:
        if isinstance(value, (list, np.ndarray)):
            # Case: value is multi-dimensional
            return np.array_equal(self._default, value)
        else:
            # Case: value is single-dimensional
            return self._default == value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        text = f"Constant({self._name}, default={self._default}, shape={self._shape})"
        return text
