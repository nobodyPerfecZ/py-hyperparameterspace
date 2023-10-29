from typing import Union, Iterable, Any
import numpy as np

from PyHyperparameterSpace.dist.abstract_dist import Distribution
from PyHyperparameterSpace.dist.categorical import Choice
from PyHyperparameterSpace.hp.abstract_hp import Hyperparameter


class Categorical(Hyperparameter):
    """
    Class to represent a categorical hyperparameter.

        Attributes:
            name (str): name of the hyperparameter
            choices (Union[list[Any], None]): all possible discrete values of hyperparameter
            default (Any): default value of the hyperparameter
            weights (Union[tuple[int], tuple[float], None]): probability distribution for each possible discrete value
    """

    def __init__(
            self,
            name: str,
            choices: list[Any],
            default: Union[Any, None] = None,
            shape: Union[int, tuple[int, ...], None] = None,
            weights: Union[list[int], list[float], None] = None,
    ):
        super().__init__(name=name, shape=shape, bounds=None, choices=choices, default=default, distribution=Choice(),
                         weights=weights)

    def _check_bounds(self, bounds: Union[tuple[int, int], tuple[float, float], None]) \
            -> Union[tuple[int, int], tuple[float, float], None]:
        return bounds

    def _is_legal_bounds(self, bounds: Union[tuple[int, int], tuple[float, float], None]) -> bool:
        raise Exception("Constant hyperparameter does not have bounds!")

    def _check_choices(self, choices: Union[list[Any], None]) -> Union[list[Any], None]:
        if self._is_legal_choices(choices):
            return choices
        else:
            raise Exception(f"Illegal choices {choices}!")

    def _is_legal_choices(self, choices: Union[list[Any], None]) -> bool:
        if isinstance(choices, list) and len(choices) > 1:
            return True
        else:
            return False

    def _check_default(self, default: Union[Any, None]) -> Any:
        if default is None:
            if self._weights:
                # Case: Take the option with the highest probability as default value
                return self._choices[self._weights.index(max(self._weights))]
            else:
                # Case: Take the first option as default value
                return self._choices[0]
        elif self._is_legal_default(default):
            return default
        else:
            raise Exception(f"Illegal default value {default}!")

    def _is_legal_default(self, default: Union[Any, None]) -> bool:
        if isinstance(default, np.ndarray):
            return any(np.array_equal(default, choice) for choice in self._choices)
        else:
            return default in self._choices

    def _check_shape(self, shape: Union[int, tuple[int, ...], None]) -> Union[int, tuple[int, ...], None]:
        if shape is None:
            # Case: Adjust the shape according to the given default value
            if self._default is None or isinstance(self._default, (int, float, bool, str)):
                return (1,)
            elif isinstance(self._default, np.ndarray):
                return self._default.shape
        elif self._is_legal_shape(shape):
            return shape
        else:
            raise Exception(f"Illegal shape {shape}")

    def _is_legal_shape(self, shape: Union[int, tuple[int, ...], None]) -> bool:
        if shape == 1 or shape == (1,):
            if isinstance(self._default, (int, float, bool, str)):
                return True
        elif isinstance(shape, (int, float, bool, str)):
            if isinstance(self._default, np.ndarray) and shape == len(self._default):
                return True
        elif isinstance(shape, tuple) and all(isinstance(s, int) for s in shape):
            if isinstance(self._default, np.ndarray) and shape == self._default.shape:
                return True
        return False

    def _check_distribution(self, distribution: Union[Distribution, None]) -> Union[Distribution, None]:
        if self._is_legal_distribution(distribution):
            return distribution
        else:
            raise Exception(f"Illegal distribution {distribution}")

    def _is_legal_distribution(self, distribution: Union[Distribution, None]) -> bool:
        if isinstance(distribution, Choice):
            return True
        else:
            return False

    def _check_weights(self, weights: Union[list[int], list[float], None]) -> Union[list[int], list[float]]:
        if weights is None:
            return Categorical._normalize([1 for _ in range(len(self._choices))])
        elif self._is_legal_weights(weights):
            return Categorical._normalize(weights)
        else:
            raise Exception(f"Illegal weights {weights}")

    def _is_legal_weights(self, weights: Union[list[int], list[float], None]) -> bool:
        if isinstance(weights, list) and len(weights) == len(self._choices) and all(0 <= w for w in weights):
            return True
        return False

    def sample(self, random: np.random.RandomState, size: Union[int, None] = None) -> Any:
        if isinstance(self._distribution, Choice):
            indices = random.choice(len(self._choices), size=size, replace=True, p=self._weights)
            if isinstance(indices, int):
                indices = [indices]
            return np.array([self._choices[idx] for idx in indices])
        else:
            raise Exception("#ERROR_BINARY: Unknown Probability Distribution!")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        text = f"Categorical({self._name}, choices={self._choices}, default={self._default}, weights={self._weights})"
        return text
