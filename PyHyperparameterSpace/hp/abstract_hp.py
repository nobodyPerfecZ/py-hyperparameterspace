from abc import ABC, abstractmethod
from typing import Any, Iterable, Union

import numpy as np


class Hyperparameter(ABC):
    """
    Abstract class to represent a hyperparameter.

        Attributes:
            name (str):
                Name of the hyperparameter

            default (Any):
                Default value of the hyperparameter

            shape (Union[int, tuple[int, ...], None]):
                Shape of the hyperparameter
    """

    def __init__(
            self,
            name: str,
            default: Any = None,
            shape: Union[tuple[int, ...], None] = None,
    ):
        if isinstance(default, list):
            default = np.array(default)

        # First set the variables
        self._name = name
        self._default = default
        self._shape = shape

        # Then check the variables and set them again
        self._default = self._check_default(default)
        self._shape = self._check_shape(shape)

    def get_name(self) -> str:
        """
        Returns:
            str:
                Name of the hyperparameter
        """
        return self._name

    def get_default(self) -> Any:
        """
        Returns:
            Any:
                Default value of the hyperparameter
        """
        return self._default

    def get_shape(self) -> tuple[int, ...]:
        """
        Returns:
            tuple[int, ...]:
                Shape of the hyperparameter
        """
        return self._shape

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
            default (Any):
                Default value to check

        Returns:
            Any:
                Default value according to the description
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
            default (Any):
                Default value to check

        Returns:
            bool:
                True if default value is legal, otherwise False
        """
        pass

    @abstractmethod
    def _check_shape(self, shape: Union[tuple[int, ...], None]) -> tuple[int, ...]:
        """
        Checks if the given shape is legal. A shape is called legal if it fulfills the format (...)
            - (dim1, dim2, ...)
            - (dim1,)
            - dim1

        and has the same dimension as the given default value.

        Args:
            shape (Union[int, tuple[int, ...], None]):
                Shape to check

        Returns:
            tuple[int, ...]:
                Legal shape
        """
        pass

    @abstractmethod
    def _is_legal_shape(self, shape: tuple[int, ...]) -> bool:
        """
        Returns true if the given shape fulfills the format (...)
            - (dim1, dim2, ...)
            - (dim1,)
            - dim1

        and has the same dimension as the given default value.

        Args:
            shape (tuple[int, ...]):
                Shape to check

        Returns:
            bool:
                True if given shape is legal
        """
        pass

    @abstractmethod
    def sample(self, random: np.random.RandomState, size: Union[int, None] = None) -> Any:
        """
        Returns a sample of values from the given hyperparameter, according to the given distribution.

        Args:
            random (np.random.RandomState):
                Random generator for the sampling procedure

            size (Union[int, None]):
                Number of samples

        Returns:
            Any:
                Sample of values from the given hyperparameter
        """
        pass

    @abstractmethod
    def valid_configuration(self, value: Any) -> bool:
        """
        Checks if the given value is a valid configuration for the given hyperparameter.

        Args:
            value (Any):
                The configuration you want to check

        Returns:
            bool:
                True if the configuration is valid for the given hyperparameter.
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

    def __getstate__(self) -> dict:
        """ Magic function to save a custom class as yaml file. """
        return {
            "name": self._name,
            "default": self._default,
            "shape": self._shape,
        }

    def __setstate__(self, state):
        """ Magic function to load a custom class from yaml file. """
        self._name = state["name"]
        self._default = state["default"]
        self._shape = state["shape"]

    @classmethod
    def _get_sample_size(
            cls,
            size: Union[int, None] = None,
            shape: Union[tuple[int, ...], None] = None,
    ) -> Union[int, tuple[int], None]:
        """
        Returns the resulting shape of the sample, according to size and shape.

        Args:
            size (Union[int, None]):
                Number of samples

            shape (Union[int, Iterable, tuple[int], None]):
                Shape of one hyperparameter

        Returns:
            Union[int, tuple[int, ...]]:
                Shape of the samples
        """
        assert size is None or size > 0, f"Illegal size {size}. The argument should be None or higher than 0!"
        if isinstance(shape, int):
            assert shape > 0, f"Illegal shape {shape}. The argument should be higher than 0!"
        elif isinstance(shape, tuple):
            assert all(s > 0 for s in shape), f"Illegal shape {shape}. The argument should be higher than 0!"

        if shape == 1 or shape == (1,):
            # Case: Shape of the hyperparameter is just a single value
            return size
        elif size is None or size == 1:
            # Case: Only one sample should be sampled
            return shape
        elif isinstance(shape, int):
            # Case: shape is a single value
            return size, shape
        else:
            # Case: shape is a tuple
            return size, *shape
