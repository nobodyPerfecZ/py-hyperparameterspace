from typing import Union, Iterable, Any
import numpy as np

from PyHyperparameterSpace.dist.continuous import MultivariateNormal, Normal, Uniform
from PyHyperparameterSpace.hp.abstract_hp import Hyperparameter
from PyHyperparameterSpace.dist.abstract_dist import Distribution


class Float(Hyperparameter):
    """
     Class to represent a floating hyperparameter.

        Attributes:
            name (str): name of the hyperparameter
            bounds (Union[tuple[float, float], tuple[int, int]]): (lower, upper) bounds of hyperparameter
            default (Any): default value of the hyperparameter
            shape (Union[int, tuple[int, ...], None]): shape of the hyperparameter
            distribution (Union[Distribution, None]): distribution from where we sample new values for hyperparameter
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

        super().__init__(name=name, shape=shape, bounds=bounds, choices=None, default=default,
                         distribution=distribution, weights=None)

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

    def _check_choices(self, choices: Union[list[Any], None]) -> Union[list[Any], None]:
        # Does not need to check weights, because float hyperparameters do not use them
        return choices

    def _is_legal_choices(self, choices: Union[list[Any], None]) -> bool:
        raise Exception("Float hyperparameter does not have choices!")

    def _check_default(self, default: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
        if default is None:
            if self._shape is None or self._shape == 1 or self._shape == (1,):
                # Case: default value is not given and shape signalize single value
                return (self.lb + self.ub) / 2
            else:
                # Case: make a default value matrix
                return np.full(shape=self._shape, fill_value=((self.lb + self.ub) / 2))
        elif self._is_legal_default(default):
            # Case: default value is legal
            return default
        else:
            # Case: default value is illegal
            raise Exception(f"Illegal default value {default}!")

    def _is_legal_default(self, default: Any) -> bool:
        if not isinstance(default, (float, int)) and \
                not(isinstance(default, np.ndarray) and np.issubdtype(default.dtype, np.floating)) and \
                not(isinstance(default, np.ndarray) and np.issubdtype(default.dtype, np.integer)):
            # Case: Default is not in the right format!
            return False
        if isinstance(default, (float, int)):
            # Case: Default is a float/int value
            return self.lb <= default < self.ub
        elif isinstance(default, np.ndarray):
            # Case: Default is a float/int matrix
            return np.all((default >= self.lb) & (default < self.ub))
        else:
            # Case: Default is a float/int tensor
            return torch.all((default >= self.lb) & (default < self.ub))

    def _check_shape(self, shape: Union[int, tuple[int, ...]]) -> Union[int, tuple[int, ...]]:
        if shape is None:
            # Case: Adjust the shape according to the given default value
            if self._default is None or isinstance(self._default, (float, int)):
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
            if isinstance(self._default, (float, int)):
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

    def _check_weights(self, weights: Union[list[int], list[float], None]) -> Union[list[int], list[float], None]:
        # Does not need to check weights, because float hyperparameters do not use them
        return weights

    def _is_legal_weights(self, weights: Union[list[int], list[float], None]) \
            -> Union[tuple[int], tuple[float], None]:
        raise Exception("Float hyperparameter does not have weights for choices!")

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
            raise Exception("#ERROR_FLOAT: Unknown Probability Distribution!")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        text = f"Float({self._name}, bounds={self._bounds}, default={self._default}, shape={self._shape}, distribution={self._distribution})"
        return text


class Integer(Hyperparameter):
    """
    Abstract class to represent a hyperparameter.

        Attributes:
            name (str): name of the hyperparameter
            bounds (Union[tuple[int], tuple[float], None]): (lower, upper) bounds of hyperparameter
            default (Any): default value of the hyperparameter
            shape (Union[int, tuple[int, ...], None]): shape of the hyperparameter
    """

    def __init__(
            self,
            name: str,
            bounds: Union[tuple[int, int], None],
            default: Union[int, list, np.ndarray, None] = None,
            shape: Union[int, tuple[int, ...], None] = None,
    ):
        if isinstance(default, list):
            default = np.array(default)

        super().__init__(name=name, shape=shape, bounds=bounds, choices=None, default=default, distribution=Uniform(),
                         weights=None)

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

    def _check_choices(self, choices: Union[list[Any], None]) -> Union[list[Any], None]:
        # Does not need to check weights, because float hyperparameters do not use them
        return choices

    def _is_legal_choices(self, choices: Union[list[Any], None]) -> bool:
        raise Exception("Integer hyperparameter does not have choices!")

    def _check_default(self, default: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        if default is None:
            if self._shape is None or self._shape == 1 or self._shape == (1,):
                # Case: shape signalize single value
                return int((self.lb + self.ub) / 2)
            else:
                # Case: make a default value matrix
                return np.full(shape=self._shape, fill_value=int((self.lb + self.ub) / 2))
        elif self._is_legal_default(default):
            # Case: default value is legal
            return default
        else:
            # Case: default value is illegal
            raise Exception(f"Illegal default value {default}!")

    def _is_legal_default(self, default: Union[int, np.ndarray]) -> bool:
        if not isinstance(default, int) and \
                not(isinstance(default, np.ndarray) and np.issubdtype(default.dtype, np.integer)):
            # Case: Default is not in the right format!
            return False
        if isinstance(default, int):
            # Case: Default is a float/int value
            return self.lb <= default <= self.ub
        else:
            # Case: Default is a float/int matrix
            return np.all((default >= self.lb) & (default <= self.ub))

    def _check_shape(self, shape: Union[int, tuple[int, ...]]) -> Union[int, tuple[int, ...]]:
        if shape is None:
            # Case: Adjust the shape according to the given default value
            if self._default is None or isinstance(self._default, int):
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
            if isinstance(self._default, int):
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
        if self._is_legal_distribution(distribution):
            return distribution
        else:
            raise Exception(f"Illegal distribution {distribution}")

    def _is_legal_distribution(self, distribution: Union[Distribution, None]) -> bool:
        if isinstance(distribution, Uniform):
            # Case: Uniform distribution
            return True
        return False

    def _check_weights(self, weights: Union[tuple[int], tuple[float], None]) -> Union[tuple[int], tuple[float], None]:
        # Does not need to check weights, because float hyperparameters do not use them
        return weights

    def _is_legal_weights(self, weights: Union[tuple[int], tuple[float], None]) -> bool:
        raise Exception("Integer hyperparameter does not have weights for choices!")

    def sample(self, random: np.random.RandomState, size: Union[int, None] = None) -> Any:
        if isinstance(self._distribution, Uniform):
            sample_size = Float._get_sample_size(size=size, shape=self._shape)
            sample = random.randint(low=self.lb, high=self.ub, size=sample_size)
            return sample
        else:
            raise Exception("#ERROR_INTEGER: Unknown Probability Distribution!")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        text = f"Integer({self._name}, bounds={self._bounds}, default={self._default}, shape={self._shape})"
        return text
