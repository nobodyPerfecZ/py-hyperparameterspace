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
            distribution (Union[Distribution], None]): distribution from where we sample new values for hyperparameter
            weights (Union[tuple[int], tuple[float], None]): probability distribution for each possible discrete value
    """

    def __init__(
            self,
            name: str,
            choices: list[Any],
            default: Union[Any, None] = None,
            shape: Union[int, tuple[int, ...], None] = None,
            distribution: Union[Distribution] = Choice(),
            weights: Union[list[int], list[float], None] = None,
    ):
        # First set the variables
        self._choices = choices
        self._distribution = distribution
        self._weights = weights

        super().__init__(name=name, shape=shape, default=default)

        # Then check the variables and set them again
        self._choices = self._check_choices(choices)
        self._distribution = self._check_distribution(distribution)
        self._weights = self._check_weights(weights)

    def get_choices(self) -> list[str]:
        """
        Returns:
            list[str]:
                List of choices
        """
        return self._choices

    def get_distribution(self) -> Distribution:
        """
        Returns:
            Distribution:
                Distribution from where we sample
        """
        return self._distribution

    def get_weights(self) -> list[float]:
        """
        Returns:
            list[float]:
                List of weights for each choice
        """
        return self._weights

    def _check_choices(self, choices: list[Any]) -> list[Any]:
        """
        Checks if the given choices are legal. A choice is called legal, if it fulfills the format [item1, item2, ...]

        Args:
            choices (list[Any]):
                Choices to check

        Returns:
            list[Any]:
                Legal choices
        """
        if self._is_legal_choices(choices):
            return choices
        else:
            raise Exception(f"Illegal choices {choices}!")

    def _is_legal_choices(self, choices: Union[list[Any], None]) -> bool:
        """
        Returns True if the given choices fulfills the format [item1, item2, ...].

        Args:
            choices (Union[list[Any], None]):
                Choices to check

        Returns:
            bool:
                True, if given choices is legal
        """
        if isinstance(choices, (list, np.ndarray)) and len(choices) > 1:
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
        if isinstance(default, (list, np.ndarray)):
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
        elif isinstance(shape, int):
            if isinstance(self._default, np.ndarray) and shape == len(self._default):
                return True
        elif isinstance(shape, tuple) and all(isinstance(s, int) for s in shape):
            if isinstance(self._default, np.ndarray) and shape == self._default.shape:
                return True
        return False

    def _check_distribution(self, distribution: Distribution) -> Distribution:
        """
        Checks if the distribution is legal. A distribution is called legal, if the class of the distribution can be
        used for the given hyperparameter class.

        Args:
            distribution (Distribution):
                Distribution to check

        Returns:
            Distribution:
                Legal distribution
        """
        if self._is_legal_distribution(distribution):
            return distribution
        else:
            raise Exception(f"Illegal distribution {distribution}")

    def _is_legal_distribution(self, distribution: Distribution) -> bool:
        """
        Returns True if the given distribution can be used for the given hyperparameter class.

        Args:
            distribution (Distribution):
                distribution to check

        Returns:
            bool:
                True if the given distribution can be used for the hyperparameter class
        """
        if isinstance(distribution, Choice):
            return True
        else:
            return False

    def _check_weights(self, weights: Union[list[int], list[float], None]) -> Union[list[int], list[float]]:
        """
        Checks if the given weights are legal. Weights are called legal, if (...)
            - fulfills the right format [w1, w2, ...]
            - length of weights and choices are equal
            - for all w_i >= 0

        and normalizes the weights to a probability distribution.

        Args:
            weights (Union[list[int], list[float], None]):
                Weights to check

        Returns:
            Union[list[int], list[float], None]:
                Normalized weights
        """
        if weights is None:
            return Categorical._normalize([1 for _ in range(len(self._choices))])
        elif self._is_legal_weights(weights):
            return Categorical._normalize(weights)
        else:
            raise Exception(f"Illegal weights {weights}")

    def _is_legal_weights(self, weights: Union[list[int], list[float], None]) -> bool:
        """
        Returns True if the given weights (...)
            - fulfills the right format [w1, w2, ...]
            - length of weights and choices are equal
            - for all w_i >= 0

        Args:
            weights (Union[list[int], list[float], None]):
                Weights to check

        Returns:
            bool:
                True if weights are legal
        """
        if isinstance(weights, (list, np.ndarray)) and len(weights) == len(self._choices) and all(0 <= w for w in weights):
            return True
        return False

    def sample(self, random: np.random.RandomState, size: Union[int, None] = None) -> Any:
        if isinstance(self._distribution, Choice):
            indices = random.choice(len(self._choices), size=size, replace=True, p=self._weights)
            if isinstance(indices, int):
                indices = [indices]
            return np.array([self._choices[idx] for idx in indices])
        else:
            raise Exception(f"Unknown Probability Distribution {self._distribution}!")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        text = f"Categorical({self._name}, choices={self._choices}, default={self._default}, weights={self._weights})"
        return text
