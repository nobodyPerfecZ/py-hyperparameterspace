from typing import Any, Union

import numpy as np

from PyHyperparameterSpace.hp.abstract_hp import Hyperparameter


class Constant(Hyperparameter):
    """
    Class to represent a constant hyperparameter, where the given default value does not get changed by the
    sampling procedure.

        Args:
            name (str):
                Name of the hyperparameter

            default (Any):
                Default value of the hyperparameter

            shape (Union[tuple[int, ...], None]):
                Dimensions-space of the hyperparameter
    """

    def __init__(
            self,
            name: str,
            default: Any,
            shape: Union[tuple[int, ...], None] = None,
    ):
        super().__init__(name=name, shape=shape, default=default)

    def _check_default(self, default: Any) -> Any:
        if self._is_legal_default(default):
            return default
        else:
            raise Exception(f"Illegal default {default}. The argument should be given!")

    def _is_legal_default(self, default: Any) -> bool:
        return default is not None

    def _check_shape(self, shape: Union[tuple[int, ...], None]) -> tuple[int, ...]:
        if shape is None and isinstance(self._default, (int, float, bool, str, np.int_, np.float_, np.str_, np.bool_)):
            # Case: shape is not given and default is single dimensional
            return 1,
        elif shape is None and isinstance(self._default, np.ndarray):
            # Case: shape is not given and default is multidimensional
            return self._default.shape
        elif shape is not None and self._is_legal_shape(shape):
            # Case: shape is given and has to be checked
            return shape
        else:
            # Case: shape is not valid
            raise Exception(f"Illegal shape {shape}. The argument should be given in the format (dim1, ...)!")

    def _is_legal_shape(self, shape: tuple[int, ...]) -> bool:
        if shape == (1,):
            # Case: shape refers to single dimensional
            return isinstance(self._default, (int, float, bool, str, np.int_, np.float_, np.bool_, np.str_))
        elif isinstance(shape, tuple) and all(isinstance(s, int) for s in shape):
            # Case: shape refers to multidimensional
            return isinstance(self._default, np.ndarray) and shape == self._default.shape
        return False

    def sample(self, random: np.random.RandomState, size: Union[int, None] = None) -> Any:
        sample_size = Constant._get_sample_size(size=size, shape=self._shape)
        if sample_size is None or sample_size == 1:
            return self._default
        else:
            return np.full(shape=sample_size, fill_value=self._default)

    def valid_configuration(self, value: Any) -> bool:
        if isinstance(value, np.ndarray):
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
        header = f"Constant({self._name}, "
        default = f"default={self._default}, "
        shape = f"shape={self._shape}"
        end = ")"
        return "".join([header, default, shape, end])
