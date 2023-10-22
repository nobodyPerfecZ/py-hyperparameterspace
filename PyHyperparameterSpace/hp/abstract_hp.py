from abc import ABC, abstractmethod
from typing import Union, Iterable, Any
import numpy as np

from PyHyperparameterSpace.dist.abstract_dist import Distribution
from PyHyperparameterSpace.dist.continuous import Normal


class Hyperparameter(ABC):
    """
    Abstract class to represent a hyperparameter.

        Attributes:
            name (str): name of the hyperparameter
            bounds (Union[tuple[int], tuple[float], None]): (lower, upper) bounds of hyperparameter
            choices (Union[list[Any], None]): all possible discrete values of hyperparameter
            default (Any): default value of the hyperparameter
            shape (Union[int, tuple[int], None]): shape of the hyperparameter
            distribution (Union[Distribution, None]): distribution from where we sample new values for hyperparameter
            weights (Union[list[int], list[float], None]): probability distribution for each possible discrete value
    """

    def __init__(
            self,
            name: str,
            bounds: Union[tuple[int, int], tuple[float, float], None] = None,
            choices: Union[list[Any], None] = None,
            default: Any = None,
            shape: Union[int, tuple[int, ...], None] = None,
            distribution: Union[Distribution, None] = None,
            weights: Union[list[int], list[float], None] = None,
    ):
        # First set the variables
        self._name = name
        self._bounds = bounds
        self._choices = choices
        self._default = default
        self._shape = shape
        self._distribution = distribution
        self._weights = weights

        # Then check the variables and set them again
        self._bounds = self._check_bounds(bounds)
        self._choices = self._check_choices(choices)
        self._default = self._check_default(default)
        self._shape = self._check_shape(shape)
        self._distribution = self._check_distribution(distribution)
        self._weights = self._check_weights(weights)

    @property
    def lb(self) -> Union[int, float, None]:
        """
        Returns:
            Union[int, float, None]: lower bound of the range
        """
        if self._bounds:
            return self._bounds[0]
        else:
            return None

    @property
    def ub(self) -> Union[int, float, None]:
        """
        Returns:
            Union[int, float, None]: upper bound of the range
        """
        if self._bounds:
            return self._bounds[1]
        else:
            return None

    def get_name(self) -> str:
        """
        Returns:
            str: name of the hyperparameter
        """
        return self._name

    def get_choices(self) -> list[str]:
        """
        Returns:
            list[str]: list of choices
        """
        return self._choices

    def get_default(self) -> Any:
        """
        Returns:
            Any: default value of the hyperparameter
        """
        return self._default

    def get_shape(self) -> Union[int, tuple[int, ...], None]:
        """
        Returns:
            Union[int, tuple[int, ...], None]: shape of the hyperparameter
        """
        return self._shape

    @abstractmethod
    def _check_bounds(self, bounds: Union[tuple[int, int], tuple[float, float], None]) \
            -> Union[tuple[int, int], tuple[float, float], None]:
        """
        Check if the given bound is legal. A bound is called legal if it fulfills the format (lower, upper).

        Args:
            bounds (Union[tuple[int, int], tuple[float, float], None]): bounds to check

        Returns:
            Union[tuple[int, int], tuple[float, float], None]: legal bounds
        """
        pass

    @abstractmethod
    def _is_legal_bounds(self, bounds: Union[tuple[int, int], tuple[float, float], None]) -> bool:
        """
        Returns True if the given bounds fulfills the right format of (lower, upper).

        Args:
            bounds (Union[tuple[int, int], tuple[float, float], None]): bounds to check

        Returns:
            bool: True if given bounds are legal
        """
        pass

    @abstractmethod
    def _check_choices(self, choices: Union[list[Any], None]) -> Union[list[Any], None]:
        """
        Checks if the given choices are legal. A choice is called legal, if it fulfills the format [item1, item2, ...]

        Args:
            choices (Union[list[Any], None]): choices to check

        Returns:
            Union[list[Any], None]: legal choices
        """
        pass

    @abstractmethod
    def _is_legal_choices(self, choices: Union[list[Any], None]) -> bool:
        """
        Returns True if the given choices fulfills the format [item1, item2, ...].

        Args:
            choices (Union[list[Any], None]): choices to check

        Returns:
            bool: True, if given choices is legal
        """
        pass

    @abstractmethod
    def _check_default(self, default: Any) -> Any:
        """
        Checks if the given default value is either (...)
            - between (lower, upper) bound
            - an option of choices

        If default value is not given, then the default value will be assigned (...)
            - as middle point between (lower, upper) bounds
            - if weights are given: option with the highest probability
            - if weights are not given: first option of choices

        Args:
            default (Any): default value to check

        Returns:
            Any: default value according to the description
        """
        pass

    @abstractmethod
    def _is_legal_default(self, default: Any) -> bool:
        """
        Returns True if default value is either (...)
            - between (lower, upper) bound
            - an option of choices
            - bounds and choices are not given (constant hyperparameter)

        Args:
            default (Any): default value to check

        Returns:
            bool: True if default value is legal, otherwise False
        """
        pass

    @abstractmethod
    def _check_shape(self, shape: Union[int, tuple[int, ...], None]) -> Union[int, tuple[int, ...], None]:
        """
        Checks if the given shape is legal. A shape is called legal if it fulfills the format (...)
            - (dim1, dim2, ...)
            - (dim1,)
            - dim1

        and has the same dimension as the given default value.

        Args:
            shape (Union[int, tuple[int, ...], None]): shape to check

        Returns:
            Union[int, tuple[int, ...], None]: legal shape
        """
        pass

    @abstractmethod
    def _is_legal_shape(self, shape: Union[int, tuple[int, ...], None]) -> bool:
        """
        Returns true if the given shape fulfills the format (...)
            - (dim1, dim2, ...)
            - (dim1,)
            - dim1

        and has the same dimension as the given default value.

        Args:
            shape (Union[int, tuple[int, ...], None]): shape to check

        Returns:
            bool: True if given shape is legal
        """
        pass

    @abstractmethod
    def _check_distribution(self, distribution: Union[Distribution, None]) -> Union[Distribution, None]:
        """
        Checks if the distribution is legal. A distribution is called legal, if the class of the distribition can be
        used for the given hyperparameter class.

        Args:
            distribution (Union[Distribution, None]): distribution to check

        Returns:
            Union[Distribution, None]: legal distribution
        """
        pass

    @abstractmethod
    def _is_legal_distribution(self, distribution: Union[Distribution, None]) -> bool:
        """
        Returns True if the given distribution can be used for the given hyperparameter class.

        Args:
            distribution (Union[Distribution, None]): distribution to check

        Returns:
            bool: True if the given distribution can be used for the hyperparameter class
        """
        pass

    @abstractmethod
    def _check_weights(self, weights: Union[list[int], list[float], None]) -> Union[list[int], list[float]]:
        """
        Checks if the given weights are legal. Weights are called legal, if (...)
            - fulfills the right format [w1, w2, ...]
            - length of weights and choices are equal
            - for all w_i >= 0

        and normalizes the weights to a probability distribution.

        Args:
            weights (Union[list[int], list[float], None]): weights to check

        Returns:
            Union[list[int], list[float], None]: normalized weights
        """
        pass

    @abstractmethod
    def _is_legal_weights(self, weights: Union[list[int], list[float], None]) -> bool:
        """
        Returns True if the given weights (...)
            - fulfills the right format [w1, w2, ...]
            - length of weights and choices are equal
            - for all w_i >= 0

        Args:
            weights (Union[list[int], list[float], None]): weights to check

        Returns:
            bool: True if weights are legal
        """
        pass

    @abstractmethod
    def sample(self, random: np.random.RandomState, size: Union[int, None] = None) -> Any:
        """
        Returns a sample of values from the given hyperparameter, according to the given distribution.

        Args:
            random (np.random.RandomState): random generator for the sampling procedure
            size (Union[int, None]): number of samples

        Returns:
            Any: sample of values from the given hyperparameter
        """
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @classmethod
    def _get_sample_size(
            cls,
            size: Union[int, None] = None,
            shape: Union[int, tuple[int, ...], None] = None,
    ) -> Union[int, tuple[int], None]:
        """
        Returns the resulting shape of the sample, according to size and shape.

        Args:
            size (Union[int, None]): number of samples
            shape (Union[int, Iterable, tuple[int], None]): shape of one hyperparameter

        Returns:
            Union[int, tuple[int, ...]]: shape of the samples
        """
        assert size is None or size > 0, "#ERROR_HYPERPARAMETER: size should be None or higher than 0!"
        if isinstance(shape, int):
            assert shape > 0, "#ERROR_HYPERPARAMETER: shape should be higher than 0!"
        elif isinstance(shape, tuple):
            assert all(s > 0 for s in shape), "#ERROR_HYPERPARAMETER: shape should be higher than 0!"

        if shape == 1 or shape == (1,):
            # Case: Shape of the hyperparameter is just a single value
            return size
        elif size is None or size == 1:
            return shape
        elif isinstance(shape, int):
            # Case: shape is a single value
            return size, shape
        else:
            # Case: shape is a tuple
            return size, *shape

    @classmethod
    def _normalize(cls, p: list[float]) -> list[float]:
        """
        Normalizes the given probability dist, so that sum(p)=1.

        Args:
            p (list[float]): (non-)normalized probability dist
        """
        assert all(0.0 <= prob for prob in p), \
            "The given non-normalized dist p cannot contain negative values!"

        sum_p = np.sum(p)
        if sum_p == 1:
            # Case: p is already normalized
            return p
        # Case: p should be normalized
        return [prob / sum_p for prob in p]
