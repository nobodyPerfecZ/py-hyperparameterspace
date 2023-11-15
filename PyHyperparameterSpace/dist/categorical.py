from typing import Union
import numpy as np


from PyHyperparameterSpace.dist.abstract_dist import Distribution


class Choice(Distribution):
    """
    Class for representing a Categorical choice dist.

        Args:
            weights (Union[list[float], np.ndarray]):
                Probability distribution to select the item
    """

    def __init__(self, weights: Union[list[float], np.ndarray]):
        self.weights = None
        self.change_distribution(weights)

    def change_distribution(self, weights: Union[list[float], np.ndarray]):
        weights = np.array(weights)

        assert weights.ndim == 1, f"Illegal weights {weights}. Argument should be a matrix of size (n,)!"
        assert np.all(0.0 <= w for w in weights), \
            f"Illegal weights {weights}. Each element inside weights should be higher or equal to 0.0!"

        # Normalize the weights
        self.weights = self._normalize(weights)

    @classmethod
    def _normalize(cls, p: Union[list[float], np.ndarray]) -> Union[list[float], np.ndarray]:
        """
        Normalizes the given probability distribution, so that sum(p)=1.

        Args:
            p (Union[list[float], np.ndarray]):
                Non-normalized probability distribution

        Returns:
            Union[list[float], np.ndarray]:
                Normalized probability distribution
        """
        assert all(0.0 <= prob for prob in p), \
            "The given non-normalized dist p cannot contain negative values!"

        if isinstance(p, list):
            result_type = list
        else:
            result_type = np.array

        sum_p = np.sum(p)
        if sum_p == 1:
            # Case: p is already normalized
            return result_type(p)
        # Case: p should be normalized
        return result_type([prob / sum_p for prob in p])

    def __str__(self):
        return f"Choice(weights={self.weights})"

    def __repr__(self):
        return self.__str__()
