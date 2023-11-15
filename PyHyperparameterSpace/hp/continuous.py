from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np

from PyHyperparameterSpace.dist.abstract_dist import Distribution
from PyHyperparameterSpace.dist.continuous import MatrixNormal, MultivariateNormal, Normal, Uniform
from PyHyperparameterSpace.hp.abstract_hp import Hyperparameter


class Continuous(Hyperparameter, ABC):
    """
    Abstract class to represent a continuous/discrete Hyperparameter.

        Attributes:
            name (str):
                Name of the hyperparameter

            bounds (Union[tuple[int, int], tuple[float, float]]):
                (lower, upper) bounds of hyperparameter

            default (Any):
                Default value of the hyperparameter

            shape (Union[int, tuple[int, ...], None]):
                Shape of the hyperparameter

            distribution (Union[Distribution, None]):
                Distribution from where we sample new values for hyperparameter
    """

    def __init__(
            self,
            name: str,
            bounds: Union[tuple[int, int], tuple[float, float]],
            default: Any = None,
            shape: Union[tuple[int, ...], None] = None,
            distribution: Union[Distribution, None] = None,
    ):
        # First set the variables
        self._bounds = bounds
        self._distribution = distribution

        super().__init__(name=name, shape=shape, default=default)

        # Then check the variables and set them again
        self._bounds = self._check_bounds(bounds)
        self._distribution = self._check_distribution(distribution)

    @property
    def lb(self) -> Union[int, float]:
        """
        Returns:
            Union[int, float]:
                Lower bound of the range
        """
        return self._bounds[0]

    @property
    def ub(self) -> Union[int, float]:
        """
        Returns:
            Union[int, float]:
                Upper bound of the range
        """
        return self._bounds[1]

    def get_bounds(self) -> Union[tuple[int, int], tuple[float, float]]:
        """
        Returns:
            Union[tuple[int, int], tuple[float, float]]:
                Bounds of the value ranges (lower bound, upper bound)
        """
        return self._bounds

    def get_distribution(self) -> Distribution:
        """
        Returns:
            Distribution:
                Distribution from where we sample
        """
        return self._distribution

    @abstractmethod
    def _check_bounds(self, bounds: Union[tuple[float, float], tuple[int, int]]) \
            -> Union[tuple[float, float], tuple[int, int]]:
        """
        Check if the given bound is legal. A bound is called legal if it fulfills the format (lower, upper).

        Args:
            bounds (Union[tuple[int, int], tuple[float, float]]):
                Bounds to check

        Returns:
            Union[tuple[int, int], tuple[float, float]]:
                Legal bounds
        """
        pass

    @abstractmethod
    def _is_legal_bounds(self, bounds: Union[tuple[float, float], tuple[int, int]]):
        """
        Returns True if the given bounds fulfills the right format of (lower, upper).

        Args:
            bounds (Union[tuple[int, int], tuple[float, float]]):
                Bounds to check

        Returns:
            bool:
                True if given bounds are legal
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def adjust_configuration(self, value: Any) -> Any:
        """
        Adjusts the given value of the hyperparameter by the given bounds.

        If value is under the lower bound than it will be assigned to the lower bound.
        If value is above the upper bound than it will be assigned next to the upper bound (but not equal to).

        Args:
            value (Any):
                Value to adjust according to the given bounds

        Returns:
            Any:
                Adjusted value according to the given bounds
        """
        pass

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["bounds"] = self._bounds
        state["distribution"] = self._distribution
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._bounds = state["bounds"]
        self._distribution = state["distribution"]


class Float(Continuous):
    """
    Class to represent a floating hyperparameter.

        Attributes:
            name (str):
                Name of the hyperparameter

            bounds (Union[tuple[int, int], tuple[float, float]):
                (lower, upper) bounds of hyperparameter

            default (Any):
                Default value of the hyperparameter

            shape (Union[int, tuple[int, ...], None]):
                Shape of the hyperparameter

            distribution (Union[Distribution, None]):
                Distribution from where we sample new values for hyperparameter
    """

    def __init__(
            self,
            name: str,
            bounds: Union[tuple[float, float], tuple[int, int]],
            default: Union[int, float, list, np.ndarray] = None,
            shape: Union[tuple[int, ...], None] = None,
            distribution: Distribution = None,
    ):
        if isinstance(default, list):
            default = np.array(default, dtype=float)

        super().__init__(name=name, shape=shape, bounds=bounds, default=default, distribution=distribution)

    def _check_bounds(self, bounds: Union[tuple[float, float], tuple[int, int]]) \
            -> Union[tuple[float, float], tuple[int, int]]:
        if self._is_legal_bounds(bounds):
            return bounds
        else:
            raise Exception(
                f"Illegal bounds {bounds}. The argument should have the format (lower, upper), where lower < upper!")

    def _is_legal_bounds(self, bounds: Union[tuple[float, float], tuple[int, int]]):
        if isinstance(bounds, tuple) and len(bounds) == 2 and \
                all(isinstance(b, (float, int, np.int_, np.float_)) for b in bounds) and bounds[0] < bounds[1]:
            return True
        else:
            return False

    def _check_default(self, default: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
        if default is None and self._shape == (1,):
            # Case: default is not given and shape refers to single dimensional values
            return (self.lb + self.ub) / 2
        elif default is None and self._shape is not None and all(isinstance(s, (int, np.int_)) for s in self._shape):
            # Case: default is not given and shape refers to multi dimensional values
            return np.full(shape=self._shape, fill_value=((self.lb + self.ub) / 2))
        elif self._is_legal_default(default):
            return default
        else:
            raise Exception(f"Illegal default {default}. The argument should be in between the bounds (lower, upper)!")

    def _is_legal_default(self, default: Any) -> bool:
        if not isinstance(default, (float, int, np.int_, np.float_)) and \
                not (isinstance(default, np.ndarray) and np.issubdtype(default.dtype, np.floating)) and \
                not (isinstance(default, np.ndarray) and np.issubdtype(default.dtype, np.integer)):
            # Case: default is not in the right format
            return False
        if isinstance(default, (float, int, np.int_, np.float_)):
            # Case: default is single dimensional
            return self.lb <= default < self.ub
        elif isinstance(default, np.ndarray):
            # Case: default is multidimensional
            return np.all((default >= self.lb) & (default < self.ub))
        return False

    def _check_shape(self, shape: Union[tuple[int, ...], None]) -> tuple[int, ...]:
        if shape is None and isinstance(self._default, (float, int, np.int_, np.float_)):
            # Case: shape is not given and default is single dimensional
            return 1,
        elif shape is None and isinstance(self._default, np.ndarray):
            # Case: shape is not given and default is multidimensional
            return self._default.shape
        elif self._is_legal_shape(shape):
            # Case: shape is given and legal
            return shape
        else:
            # Case: shape is illegal
            raise Exception(f"Illegal shape {shape}. The argument should have the right format (dim1, ...)!")

    def _is_legal_shape(self, shape: tuple[int, ...]) -> bool:
        if shape == (1,) and isinstance(self._default, (float, int, np.int_, np.float_)):
            # Case: shape and default refers to single dimensional
            return True
        elif isinstance(shape, tuple) and all(isinstance(s, int) for s in shape) and \
                isinstance(self._default, np.ndarray) and shape == self._default.shape:
            # Case: shape and default refers to multidimensional
            return True
        return False

    def _check_distribution(self, distribution: Union[Distribution, None]) -> Union[Distribution, None]:
        if distribution is None:
            return Uniform(lb=self.lb, ub=self.ub)
        elif self._is_legal_distribution(distribution):
            return distribution
        else:
            raise Exception(
                f"Illegal distribution {distribution}. The argument should be in class of MatrixNormal(...), MultivariateNormal(...), Normal(...) or Uniform(...)!")

    def _is_legal_distribution(self, distribution: Distribution) -> bool:
        if isinstance(distribution, MatrixNormal):
            # Case: Matrix normal distribution
            # Check if mean in between the bounds and shape should have a format of (n,p)
            return np.all((self.lb <= distribution.M) & (distribution.M < self.ub)) and \
                   self._shape == distribution.M.shape
        if isinstance(distribution, MultivariateNormal):
            # Case: Multivariate normal distribution
            # Check if mean is in between the bounds and shape should have a format of (n,)
            return np.all((self.lb <= distribution.mean) & (distribution.mean < self.ub)) and \
                   self._shape == distribution.mean.shape
        elif isinstance(distribution, Normal):
            # Case: Normal distribution
            # Check if mean (loc) is in between the bounds
            return self.lb <= distribution.mean < self.ub
        elif isinstance(distribution, Uniform):
            # Case: Uniform distribution
            return self.lb <= distribution.lb < self.ub and self.lb <= distribution.ub <= self.ub and \
                   distribution.lb < distribution.ub
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
        if isinstance(self._distribution, MatrixNormal):
            # Case: Sample from matrix normal distribution by sampling from multivariate normal distribution
            sample = random.multivariate_normal(
                mean=np.ravel(self._distribution.M),
                cov=np.kron(self._distribution.U, self._distribution.V),
                size=size
            )
            sample = self.adjust_configuration(sample)

            # Reshape the samples into matrices
            if size is None:
                sample = sample.reshape(self._distribution.M.shape[0], self._distribution.M.shape[1])
            else:
                sample = sample.reshape(size, self._distribution.M.shape[0], self._distribution.M.shape[1])
            return sample
        elif isinstance(self._distribution, MultivariateNormal):
            # Case: Sample from multivariate normal distribution
            sample = random.multivariate_normal(mean=self._distribution.mean, cov=self._distribution.cov, size=size)
            sample = self.adjust_configuration(sample)
            return sample
        elif isinstance(self._distribution, Normal):
            # Case: Sample from normal distribution
            sample_size = Float._get_sample_size(size=size, shape=self._shape)
            sample = random.normal(loc=self._distribution.mean, scale=self._distribution.std, size=sample_size)
            sample = self.adjust_configuration(sample)
            return sample
        elif isinstance(self._distribution, Uniform):
            # Case: Sample from uniform distribution
            sample_size = Float._get_sample_size(size=size, shape=self._shape)
            sample = random.uniform(low=self._distribution.lb, high=self._distribution.ub, size=sample_size)
            sample = self.adjust_configuration(sample)
            return sample
        else:
            raise Exception(f"Unknown Distribution {self._distribution}!")

    def valid_configuration(self, value: Any) -> bool:
        if isinstance(value, (list, np.ndarray)):
            # Case: Value is multidimensional
            value = np.array(value)
            return np.all((self.lb <= value) & (value < self.ub)) and self._shape == value.shape
        elif isinstance(value, (int, float)):
            # Case: value is single dimensional
            return self.lb <= value < self.ub
        return False

    def adjust_configuration(self, value: Any) -> Any:
        if isinstance(value, (list, np.ndarray)) and \
                (np.issubdtype(value.dtype, np.float_) or np.issubdtype(value.dtype, np.int_)):
            # Case: value is multidimensional
            value = np.array(value)

            # Do not exceed lower, upper bound
            value[value < self.lb] = self.lb
            value[value >= self.ub] = self.ub - 1e-10

            return value
        elif isinstance(value, (int, float, np.int_, np.float_)):
            # Case: value is single dimensional
            if value < self.lb:
                value = self.lb
            elif value >= self.ub:
                value = self.ub - 1e-10

            return value
        else:
            # Case: value is illegal
            raise Exception(
                f"Illegal value {value}. The argument should be inside the bounds ({self._lb}, {self._ub})!")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        text = f"Float({self._name}, bounds={self._bounds}, default={self._default}, shape={self._shape}, distribution={self._distribution})"
        return text


class Integer(Continuous):
    """
    Class to represent a discrete hyperparameter.

        Attributes:
            name (str):
                Name of the hyperparameter

            bounds (tuple[int, int]):
                (lower, upper) bounds of hyperparameter

            default (Any):
                Default value of the hyperparameter

            shape (Union[int, tuple[int, ...], None]):
                Shape of the hyperparameter

            distribution (Union[Distribution, None]):
                Distribution from where we sample new values for hyperparameter
    """

    def __init__(
            self,
            name: str,
            bounds: Union[tuple[int, int]],
            default: Union[int, list, np.ndarray, None] = None,
            shape: Union[tuple[int, ...], None] = None,
            distribution: Distribution = None,
    ):
        super().__init__(name=name, shape=shape, bounds=bounds, default=default, distribution=distribution)

    def _check_bounds(self, bounds: tuple[int, int]) -> tuple[int, int]:
        if self._is_legal_bounds(bounds):
            return bounds
        else:
            raise Exception(f"Illegal bounds {bounds}!")

    def _is_legal_bounds(self, bounds: tuple[int, int]) -> bool:
        if isinstance(bounds, tuple) and len(bounds) == 2 and \
                all(isinstance(b, (int, np.int_)) for b in bounds) and bounds[0] < bounds[1]:
            return True
        else:
            return False

    def _check_default(self, default: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        if default is None and self._shape == (1,):
            # Case: default is not given and shape refers to single dimensional values
            return int((self.lb + self.ub) / 2)
        elif default is None and self._shape is not None and all(isinstance(s, (int, np.int_)) for s in self._shape):
            # Case: default is not given and shape refers to multidimensional values
            return np.full(shape=self._shape, fill_value=int((self.lb + self.ub) / 2))
        elif self._is_legal_default(default):
            # Case: default value is legal
            return default
        else:
            # Case: default value is illegal
            raise Exception(f"Illegal default {default}. The argument should be in between the bounds (lower, upper)!")

    def _is_legal_default(self, default: Union[int, np.ndarray]) -> bool:
        if not isinstance(default, int) and \
                not (isinstance(default, np.ndarray) and np.issubdtype(default.dtype, np.integer)):
            # Case: default is not in the right format!
            return False
        if isinstance(default, (int, np.int_)):
            # Case: default is single dimensional
            return self.lb <= default <= self.ub
        else:
            # Case: default is multidimensional
            return np.all((default >= self.lb) & (default <= self.ub))

    def _check_shape(self, shape: Union[tuple[int, ...], None]) -> tuple[int, ...]:
        if shape is None and isinstance(self._default, (int, np.int_)):
            # Case: shape is not given and default is single dimensional
            return 1,
        elif shape is None and isinstance(self._default, np.ndarray):
            # Case: shape is not given and default is multidimensional
            return self._default.shape
        elif self._is_legal_shape(shape):
            # Case: shape is given
            return shape
        else:
            raise Exception(
                f"Illegal shape {shape}. The argument should be in the format (lower, upper), where lower < upper!")

    def _is_legal_shape(self, shape: tuple[int, ...]) -> bool:
        if shape == (1,) and isinstance(self._default, (int, np.int_)):
            # Case: shape and default refers to single dimensional
            return True
        elif isinstance(shape, tuple) and all(isinstance(s, (int, np.int_)) for s in shape) and \
                isinstance(self._default, np.ndarray) and shape == self._default.shape:
            return True
        return False

    def _check_distribution(self, distribution: Union[Distribution, None]) -> Distribution:
        if distribution is None:
            return Uniform(lb=self.lb, ub=self.ub)
        elif self._is_legal_distribution(distribution):
            return distribution
        else:
            raise Exception(f"Illegal distribution {distribution}. The argument should have the class Uniform(...)!")

    def _is_legal_distribution(self, distribution: Distribution) -> bool:
        if isinstance(distribution, Uniform):
            # Case: Uniform distribution
            return isinstance(distribution.lb, int) and isinstance(distribution.ub, int) and \
                   self.lb <= distribution.lb < self.ub and self.lb <= distribution.ub <= self.ub and \
                   distribution.lb < distribution.ub
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
        if isinstance(self._distribution, Uniform):
            sample_size = Integer._get_sample_size(size=size, shape=self._shape)
            sample = random.randint(low=self._distribution.lb, high=self._distribution.ub, size=sample_size)
            sample = self.adjust_configuration(sample)
            return sample
        else:
            raise Exception(f"Unknown Distribution {self._distribution}!")

    def valid_configuration(self, value: Any) -> bool:
        if isinstance(value, (list, np.ndarray)):
            # Case: Value is multi-dimensional
            value = np.array(value)
            return np.all((self.lb <= value) & (value < self.ub)) and self._shape == value.shape
        elif isinstance(value, (int, np.int_)):
            # Case: value is single-dimensional
            return self.lb <= value < self.ub
        else:
            return False

    def adjust_configuration(self, value: Any) -> Any:
        if isinstance(value, (list, np.ndarray)) and np.issubdtype(value.dtype, np.int_):
            # Case: value is multidimensional
            value = np.array(value)

            # Do not exceed lower, upper bound
            value[value < self.lb] = self.lb
            value[value >= self.ub] = self.ub - 1

            return value
        elif isinstance(value, (int, np.int_)):
            # Case: value is single dimensional
            if value < self.lb:
                value = self.lb
            elif value >= self.ub:
                value = self.ub - 1
            return value
        else:
            # Case: value is illegal
            raise Exception(
                f"Illegal value {value}. The argument should be inside the bounds ({self._lb}, {self._ub})!")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        text = f"Integer({self._name}, bounds={self._bounds}, default={self._default}, shape={self._shape})"
        return text
