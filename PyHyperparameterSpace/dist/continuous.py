from PyHyperparameterSpace.dist.abstract_dist import Distribution


class MultivariateNormal(Distribution):
    """
    Class for representing a Multivariate Normal (Gaussian) Distribution ~N(mean, Covariance)
    """

    def __init__(self, mean: list[float], cov: list[list[float]]):
        assert len(mean) == len(cov), f"Illegal mean {mean}. Argument should be a vector of size (N,)."
        assert all(len(mean) == len(c) for c in cov), f"Illegal cov {cov}. Argument should be a matrix of size (N,N)"
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
