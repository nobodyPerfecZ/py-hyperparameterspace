from typing import Union, Iterable, Any
import numpy as np

from PyHyperparameterSpace.dist.abstract_dist import Distribution
from PyHyperparameterSpace.dist.categorical import Choice
from PyHyperparameterSpace.hp.abstract_hp import Hyperparameter


class Categorical(Hyperparameter):
    """
    Class to represent a categorical hyperparameter.

        Attributes:
            name (str):
                Name of the hyperparameter

            choices (Union[list[Any], np.ndarray]):
                All possible discrete values of hyperparameter

            default (Any):
                Default value of the hyperparameter

            distribution (Union[Distribution, None]):
                Distribution from where we sample new values for hyperparameter
    """

    def __init__(
            self,
            name: str,
            choices: list[Any],
            default: Union[Any, None] = None,
            shape: Union[int, tuple[int, ...], None] = None,
            distribution: Union[Distribution, None] = None,
    ):
        if isinstance(choices, list):
            choices = np.array(choices)

        # First set the variables
        self._choices = choices
        self._distribution = distribution

        super().__init__(name=name, shape=shape, default=default)

        # Then check the variables and set them again
        self._choices = self._check_choices(choices)
        self._distribution = self._check_distribution(distribution)

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
            if self._distribution is not None:
                # Case: Take the option with the highest probability as default value
                return self._choices[np.argmax(self._distribution.weights)]
            else:
                # Case: Take the first option as default value
                return self._choices[0]
        elif self._is_legal_default(default):
            return default
        else:
            raise Exception(f"Illegal default {default}!")

    def _is_legal_default(self, default: Union[Any, None]) -> bool:
        if isinstance(default, (list, np.ndarray)):
            return any(np.array_equal(default, choice) for choice in self._choices)
        else:
            return default in self._choices

    def _check_shape(self, shape: Union[int, tuple[int, ...], None]) -> Union[int, tuple[int, ...], None]:
        if shape is None:
            # Case: Adjust the shape according to the given default value
            if self._default is None or isinstance(self._default, (int, float, bool, str)):
                # Case: default value is not given or is single dimensional
                return (1,)
            elif isinstance(self._default, np.ndarray):
                # Case: default value is multidimensional
                return self._default.shape
        elif self._is_legal_shape(shape):
            return shape
        else:
            raise Exception(f"Illegal shape {shape}!")

    def _is_legal_shape(self, shape: Union[int, tuple[int, ...]]) -> bool:
        if shape == 1 or shape == (1,):
            # Case: shape refers to single dimensional
            if isinstance(self._default, (int, float, bool, str)):
                return True
        elif isinstance(shape, int):
            # Case: shape refers to array-like dimensional
            if isinstance(self._default, np.ndarray) and shape == len(self._default) and self._default.ndim == 1:
                # Case: default is array-like dimensional
                return True
        elif isinstance(shape, tuple) and all(isinstance(s, int) for s in shape):
            # Case: shape refers to multidimensional
            if isinstance(self._default, np.ndarray) and shape == self._default.shape:
                # Case: default value is multidimensional
                return True
        return False

    def _check_distribution(self, distribution: Union[Distribution, None]) -> Distribution:
        """
        Checks if the distribution is legal. A distribution is called legal, if the class of the distribution can be
        used for the given hyperparameter class.

        Args:
            distribution (Union[Distribution, None]):
                Distribution to check

        Returns:
            Distribution:
                Legal distribution
        """
        if distribution is None:
            # Case: Distribution is not given
            return Choice(weights=np.ones(len(self._choices)))
        elif self._is_legal_distribution(distribution):
            # Case: Distribution is given and legal
            return distribution
        else:
            raise Exception(f"Illegal distribution {distribution}!")

    def _is_legal_distribution(self, distribution: Distribution) -> bool:
        """
        Returns True if the given distribution can be used for the given hyperparameter class.

        Args:
            distribution (Distribution):
                Distribution to check

        Returns:
            bool:
                True if the given distribution can be used for the hyperparameter class
        """
        if isinstance(distribution, Choice):
            return np.isclose(sum(distribution.weights), 1) and distribution.weights.shape == (self._choices.shape[0],)
        return False

    def change_distribution(self, **kwargs):
        """
        Changes the distribution to the given parameters.

        Args:
            **kwargs (dict):
                Parameters that defines the distribution
        """
        self._distribution.change_distribution(**kwargs)
        self._check_distribution(self._distribution)

    def sample(self, random: np.random.RandomState, size: Union[int, None] = None) -> Any:
        if isinstance(self._distribution, Choice):
            # Case: Sample from given distribution (with weights)
            indices = random.choice(len(self._choices), size=size, replace=True, p=self._distribution.weights)
            if isinstance(indices, int):
                # Case: Only a single sample should be returned
                if len(self._shape) > 1:
                    # Case: values is multidimensional
                    return np.array(self._choices[indices])
                else:
                    # Case: values are single dimensional
                    return self._choices[indices]
            return np.array([self._choices[idx] for idx in indices])
        else:
            raise Exception(f"Unknown Probability Distribution {self._distribution}!")

    def valid_configuration(self, value: Any) -> bool:
        if isinstance(value, (list, np.ndarray)):
            # Case: value is multidimensional
            return any(np.array_equal(value, choice) for choice in self._choices)
        else:
            # Case: value is single dimensional
            return value in self._choices

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        text = f"Categorical({self._name}, choices={self._choices}, default={self._default}, distribution={self._distribution})"
        return text

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["choices"] = self._choices
        state["distribution"] = self._distribution
        return state

    def __setstate__(self, state) -> dict:
        super().__setstate__(state)
        self._choices = state["choices"]
        self._distribution = state["distribution"]
