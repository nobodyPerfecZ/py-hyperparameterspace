from PyHyperparameterSpace.dist.abstract_dist import Distribution


class Choice(Distribution):
    """
    Class for representing a Categorical choice dist.
    TODO: Refactor
    """

    def __init__(self):
        pass

    def change_distribution(**kwargs):
        raise Exception("Illegal call of change_distribution(). Choice distribution cannot be changed!")

    def __str__(self):
        return "Choice()"

    def __repr__(self):
        return self.__str__()
