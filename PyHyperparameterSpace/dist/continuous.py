from PyHyperparameterSpace.dist.abstract_dist import Distribution


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
