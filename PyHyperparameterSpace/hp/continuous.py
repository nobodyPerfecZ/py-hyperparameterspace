from typing import Union, Iterable, Any
from abc import ABC, abstractmethod
import numpy as np

from PyHyperparameterSpace.dist.continuous import MultivariateNormal, Normal, Uniform
from PyHyperparameterSpace.hp.abstract_hp import Hyperparameter
from PyHyperparameterSpace.dist.abstract_dist import Distribution


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

            distribution (Distribution):
                Distribution from where we sample new values for hyperparameter
    """

    def __init__(
            self,
            name: str,
            bounds: Union[tuple[int, int], tuple[float, float]],
            default: Any = None,
            shape: Union[int, tuple[int, ...], None] = None,
            distribution: Union[Distribution, None] = Uniform(),
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

            distribution (Distribution):
                Distribution from where we sample new values for hyperparameter
    """

    def __init__(
            self,
            name: str,
            bounds: Union[tuple[float, float], tuple[int, int]],
            default: Union[int, float, list, np.ndarray] = None,
            shape: Union[int, tuple[int, ...], None] = None,
            distribution: Distribution = Uniform(),
    ):
        if isinstance(default, list):
            default = np.array(default, dtype=float)

        super().__init__(name=name, shape=shape, bounds=bounds, default=default, distribution=distribution)

    def _check_bounds(self, bounds: Union[tuple[float, float], tuple[int, int]]) \
            -> Union[tuple[float, float], tuple[int, int]]:
        if self._is_legal_bounds(bounds):
            return bounds
        else:
            raise Exception(f"Illegal bounds {bounds}!")

    def _is_legal_bounds(self, bounds: Union[tuple[float, float], tuple[int, int]]):
        if isinstance(bounds, tuple) and len(bounds) == 2 and \
                all(isinstance(b, (float, int)) for b in bounds) and bounds[0] < bounds[1]:
            return True
        else:
            return False

    def _check_default(self, default: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
        if default is None:
            # Case: default is not given
            if self._shape is None or self._shape == 1 or self._shape == (1,):
                # Case: default should be single dimensional
                return (self.lb + self.ub) / 2
            else:
                # Case: default should be multidimensional
                return np.full(shape=self._shape, fill_value=((self.lb + self.ub) / 2))
        elif self._is_legal_default(default):
            return default
        else:
            raise Exception(f"Illegal default value {default}!")

    def _is_legal_default(self, default: Any) -> bool:
        if not isinstance(default, (float, int)) and \
                not (isinstance(default, np.ndarray) and np.issubdtype(default.dtype, np.floating)) and \
                not (isinstance(default, np.ndarray) and np.issubdtype(default.dtype, np.integer)):
            # Case: default is not in the right format
            return False
        if isinstance(default, (float, int)):
            # Case: default is single dimensional
            return self.lb <= default < self.ub
        elif isinstance(default, np.ndarray):
            # Case: default is multidimensional
            return np.all((default >= self.lb) & (default < self.ub))
        return False

    def _check_shape(self, shape: Union[int, tuple[int, ...]]) -> Union[int, tuple[int, ...]]:
        if shape is None:
            # Case: Adjust the shape according to the given default value
            if self._default is None or isinstance(self._default, (float, int)):
                # Case: default is not given or single dimensional
                return (1,)
            elif isinstance(self._default, np.ndarray):
                # Case: default is multidimensional
                return self._default.shape
        elif self._is_legal_shape(shape):
            return shape
        else:
            raise Exception(f"Illegal shape {shape}!")

    def _is_legal_shape(self, shape: Union[int, tuple[int, ...]]) -> bool:
        if shape == 1 or shape == (1,):
            # Case: shape refers to single dimensional
            if isinstance(self._default, (float, int)):
                # Case: default is single dimensional
                return True
        elif isinstance(shape, int):
            # Case: shape refers to array-like dimensional
            if isinstance(self._default, np.ndarray) and shape == len(self._default) and self._default.ndim == 1:
                # Case: default is array-like dimensional
                return True
        elif isinstance(shape, tuple) and all(isinstance(s, int) for s in shape):
            # Case: shape refers to multidimensional
            if isinstance(self._default, np.ndarray) and shape == self._default.shape:
                # Case: default is multidimensional
                return True
        return False

    def _check_distribution(self, distribution: Union[Distribution, None]) -> Union[Distribution, None]:
        if self._is_legal_distribution(distribution):
            return distribution
        else:
            raise Exception(f"Illegal distribution {distribution}!")

    def _is_legal_distribution(self, distribution: Union[Distribution, None]) -> bool:
        if isinstance(distribution, MultivariateNormal):
            # Case: Multivariate normal distribution
            # Check if mean is in between the bounds and shape should have a format of (N,)
            return all(self.lb <= m <= self.ub for m in distribution.mean) and \
                   self._shape == (len(distribution.mean),) and \
                   len(distribution.mean) > 1 and \
                   len(distribution.cov) > 1
        elif isinstance(distribution, Normal):
            # Case: Normal distribution
            # Check if mean (loc) is in between the bounds
            return self.lb <= distribution.loc < self.ub
        elif isinstance(distribution, Uniform):
            # Case: Uniform distribution
            return True
        return False

    def sample(self, random: np.random.RandomState, size: Union[int, None] = None) -> Any:
        if isinstance(self._distribution, MultivariateNormal):
            # Case: Sample from multivariate normal distribution
            sample = random.multivariate_normal(mean=self._distribution.mean, cov=self._distribution.cov, size=size)

            # Do not exceed lower, upper bound
            if isinstance(sample, float):
                # Case: Sample is a single value
                if sample < self.lb:
                    sample = self.lb
                elif sample >= self.ub:
                    sample = self.ub - 1e-10
            else:
                # Case: Sample is a numpy array
                sample[sample < self.lb] = self.lb
                sample[sample >= self.ub] = self.ub - 1e-10
            return sample
        elif isinstance(self._distribution, Normal):
            # Case: Sample from normal distribution
            sample_size = Float._get_sample_size(size=size, shape=self._shape)
            sample = random.normal(loc=self._distribution.loc, scale=self._distribution.scale, size=sample_size)

            # Do not exceed lower, upper bound
            if isinstance(sample, float):
                # Case: Sample is a single value
                if sample < self.lb:
                    sample = self.lb
                elif sample >= self.ub:
                    sample = self.ub - 1e-10
            else:
                # Case: Sample is a numpy array
                sample[sample < self.lb] = self.lb
                sample[sample >= self.ub] = self.ub - 1e-10
            return sample
        elif isinstance(self._distribution, Uniform):
            # Case: Sample from uniform distribution
            sample_size = Float._get_sample_size(size=size, shape=self._shape)
            sample = random.uniform(low=self.lb, high=self.ub, size=sample_size)
            return sample
        else:
            raise Exception("Unknown Probability Distribution!")

    def valid_configuration(self, value: Any) -> bool:
        if isinstance(value, (list, np.ndarray)):
            # Case: Value is multidimensional
            value = np.array(value)
            return np.all((self.lb <= value) & (value < self.ub)) and self._shape == value.shape
        elif isinstance(value, (int, float)):
            # Case: value is single dimensional
            return self.lb <= value < self.ub
        return False

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

            distribution (Distribution):
                Distribution from where we sample new values for hyperparameter
    """

    def __init__(
            self,
            name: str,
            bounds: Union[tuple[int, int]],
            default: Union[int, list, np.ndarray, None] = None,
            shape: Union[int, tuple[int, ...], None] = None,
            distribution: Distribution = Uniform(),
    ):
        super().__init__(name=name, shape=shape, bounds=bounds, default=default, distribution=distribution)

    def _check_bounds(self, bounds: Union[tuple[int, int], None]) -> Union[tuple[int], tuple[float], None]:
        if self._is_legal_bounds(bounds):
            return bounds
        else:
            raise Exception(f"Illegal bounds {bounds}!")

    def _is_legal_bounds(self, bounds: Union[tuple[int], tuple[float], None]) -> bool:
        if isinstance(bounds, tuple) and len(bounds) == 2 and \
                all(isinstance(b, int) for b in bounds) and bounds[0] < bounds[1]:
            return True
        else:
            return False

    def _check_default(self, default: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        if default is None:
            # Case: default is not given
            if self._shape is None or self._shape == 1 or self._shape == (1,):
                # Case: shape refers to single dimensional
                return int((self.lb + self.ub) / 2)
            else:
                # Case: Shape refers to multidimensional
                return np.full(shape=self._shape, fill_value=int((self.lb + self.ub) / 2))
        elif self._is_legal_default(default):
            # Case: default value is legal
            return default
        else:
            # Case: default value is illegal
            raise Exception(f"Illegal default value {default}!")

    def _is_legal_default(self, default: Union[int, np.ndarray]) -> bool:
        if not isinstance(default, int) and \
                not (isinstance(default, np.ndarray) and np.issubdtype(default.dtype, np.integer)):
            # Case: default is not in the right format!
            return False
        if isinstance(default, int):
            # Case: default is single dimensional
            return self.lb <= default <= self.ub
        else:
            # Case: default is multidimensional
            return np.all((default >= self.lb) & (default <= self.ub))

    def _check_shape(self, shape: Union[int, tuple[int, ...]]) -> Union[int, tuple[int, ...]]:
        if shape is None:
            # Case: Adjust the shape according to the given default value
            if self._default is None or isinstance(self._default, int):
                # Case: default is not given or single dimensional
                return (1,)
            elif isinstance(self._default, np.ndarray):
                # Case: default is multidimensional
                return self._default.shape
        elif self._is_legal_shape(shape):
            return shape
        else:
            raise Exception(f"Illegal shape {shape}!")

    def _is_legal_shape(self, shape: Union[int, tuple[int, ...]]) -> bool:
        if shape == 1 or shape == (1,):
            # Check if shape has the right format for the default value
            if isinstance(self._default, int):
                # Case: default is single dimensional
                return True
        elif isinstance(shape, int):
            # Check if shape has the right format for the default value
            if isinstance(self._default, np.ndarray) and shape == len(self._default) and self._default.ndim == 1:
                # Case: default is array-like dimensional
                return True
        elif isinstance(shape, tuple) and all(isinstance(s, int) for s in shape):
            # Check if shape is in the right format for the default value
            if isinstance(self._default, np.ndarray) and shape == self._default.shape:
                # Case: default is multidimensional
                return True
        return False

    def _check_distribution(self, distribution: Union[Distribution, None]) -> Union[Distribution, None]:
        if self._is_legal_distribution(distribution):
            return distribution
        else:
            raise Exception(f"Illegal distribution {distribution}")

    def _is_legal_distribution(self, distribution: Union[Distribution, None]) -> bool:
        if isinstance(distribution, Uniform):
            # Case: Uniform distribution
            return True
        return False

    def sample(self, random: np.random.RandomState, size: Union[int, None] = None) -> Any:
        if isinstance(self._distribution, Uniform):
            sample_size = Integer._get_sample_size(size=size, shape=self._shape)
            sample = random.randint(low=self.lb, high=self.ub, size=sample_size)
            return sample
        else:
            raise Exception("Unknown Probability Distribution!")

    def valid_configuration(self, value: Any) -> bool:
        if isinstance(value, (list, np.ndarray)):
            # Case: Value is multi-dimensional
            value = np.array(value)
            return np.all((self.lb <= value) & (value < self.ub)) and self._shape == value.shape
        elif isinstance(value, int):
            # Case: value is single-dimensional
            return self.lb <= value < self.ub
        else:
            return False

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        text = f"Integer({self._name}, bounds={self._bounds}, default={self._default}, shape={self._shape})"
        return text
