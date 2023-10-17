from PyHyperparameterSpace.dist.abstract_dist import Distribution


class Choice(Distribution):
    """
    Class for representing a Categorical choice dist.
    """

    def __init__(self):
        pass

    def __str__(self):
        return "Choice()"

    def __repr__(self):
        return self.__str__()
