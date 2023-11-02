from typing import Union
import numpy as np

from PyHyperparameterSpace.dist.abstract_dist import Distribution


class MatrixNormal(Distribution):
    """
    Class for representing a Matrix Normal (Gaussian) Distribution ~MN_n,p(M, U, V)
    """

    def __init__(
            self,
            M: Union[list[list[float]], np.ndarray],
            U: Union[list[list[float]], np.ndarray],
            V: Union[list[list[float]], np.ndarray]
    ):
        M = np.array(M)
        U = np.array(U)
        V = np.array(V)

        assert M.ndim == 2, f"Illegal M {M}. Argument should be a matrix of size (n, p)!"
        assert U.ndim == 2 and U.shape == (M.shape[0], M.shape[0]), f"Illegal U {U}. Argument should be a matrix of size (n, n)!"
        assert V.ndim == 2 and V.shape == (M.shape[1], M.shape[1]), f"Illegal V {V}. Argument should be a matrix of size (p, p)!"

        self.M = M
        self.U = U
        self.V = V


class MultivariateNormal(Distribution):
    """
    Class for representing a Multivariate Normal (Gaussian) Distribution ~N(mean, Covariance)
    """

    def __init__(self, mean: Union[list[float], np.ndarray], cov: Union[list[list[float]], np.ndarray]):
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
    """

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __str__(self):
        return f"Normal(loc={self.loc}, scale={self.scale})"

    def __repr__(self):
        return self.__str__()


class Uniform(Distribution):
    """
    Class for representing a continuous2 Uniform dist ~U(a,b).
    """

    def __init__(self):
        pass

    def __str__(self):
        return "Uniform()"

    def __repr__(self):
        return self.__str__()
