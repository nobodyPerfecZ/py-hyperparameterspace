from typing import Union

import numpy as np

from PyHyperparameterSpace.dist.abstract_dist import Distribution


class MatrixNormal(Distribution):
    """
    Class for representing a Matrix Normal (Gaussian) Distribution ~M N_n,p(M, U, V)

    Args:
        M (Union[list[list[float]], np.ndarray]):
            The mean matrix

        U (Union[list[list[float]], np.ndarray]):
            Covariance matrix for the rows

        V (Union[list[list[float]], np.ndarray]):
            Covariance matrix for the columns
    """

    def __init__(
            self,
            M: Union[list[list[float]], np.ndarray],
            U: Union[list[list[float]], np.ndarray],
            V: Union[list[list[float]], np.ndarray],
    ):
        self.M = None
        self.U = None
        self.V = None
        self.change_distribution(M, U, V)

    def change_distribution(
            self,
            M: Union[list[list[float]], np.ndarray],
            U: Union[list[list[float]], np.ndarray],
            V: Union[list[list[float]], np.ndarray],
    ):
        M = np.array(M)
        U = np.array(U)
        V = np.array(V)

        assert M.ndim == 2, f"Illegal M {M}. The argument should be a matrix of size (n, p)!"
        assert U.ndim == 2 and U.shape == (M.shape[0], M.shape[0]), \
            f"Illegal U {U}. The argument should be a matrix of size (n, n)!"
        assert V.ndim == 2 and V.shape == (M.shape[1], M.shape[1]), \
            f"Illegal V {V}. The argument should be a matrix of size (p, p)!"

        self.M = M
        self.U = U
        self.V = V

    def __str__(self):
        return f"MatrixNormal(mean={self.mean}, U={self.U}, V={self.V})"

    def __repr__(self):
        return self.__str__()


class MultivariateNormal(Distribution):
    """
    Class for representing a Multivariate Normal (Gaussian) Distribution ~N(mean, Covariance)

        Args:
            mean (Union[list[float], np.ndarray]):
                The mean vector

            cov (Union[list[list[float]], np.ndarray]):
                The covariance matrix
    """

    def __init__(self, mean: Union[list[float], np.ndarray], cov: Union[list[list[float]], np.ndarray]):
        self.mean = None
        self.cov = None

        # Set the parameters for the distribution
        self.change_distribution(mean, cov)

    def change_distribution(self, mean: Union[list[float], np.ndarray], cov: Union[list[list[float]], np.ndarray]):
        mean = np.array(mean)
        cov = np.array(cov)

        assert mean.ndim == 1, f"Illegal mean {mean}. Argument should be a vector of size (n,)!"
        assert cov.ndim == 2 and mean.shape == (cov.shape[0],) and mean.shape == (cov.shape[1],), \
            f"Illegal cov {cov}. Argument should be a matrix of size (n,n)!"

        self.mean = mean
        self.cov = cov

    def __str__(self):
        return f"MultivariateNormal(mean={self.mean}, cov={self.cov})"

    def __repr__(self):
        return self.__str__()


class Normal(Distribution):
    """
    Class for representing a Normal (Gaussian) Distribution ~N(mean, std).

        Args:
            mean (float):
                The mean value

            std (float):
                The standard deviation
    """

    def __init__(self, mean: float, std: float):
        self.mean = None
        self.std = None

        self.change_distribution(mean, std)

    def change_distribution(self, mean: float, std: float):
        assert std >= 0, f"Illegal std {std}. The argument should be >= 0!"

        self.mean = mean
        self.std = std

    def __str__(self):
        return f"Normal(mean={self.mean}, std={self.std})"

    def __repr__(self):
        return self.__str__()


class Uniform(Distribution):
    """
    Class for representing a continuous2 Uniform dist ~U(a,b).
    """

    def __init__(self):
        pass

    def change_distribution(self, **kwargs):
        raise Exception("Illegal call of change_distribution(). Uniform distribution cannot be changed!")

    def __str__(self):
        return "Uniform()"

    def __repr__(self):
        return self.__str__()
