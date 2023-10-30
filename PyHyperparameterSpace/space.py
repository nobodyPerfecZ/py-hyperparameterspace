from typing import Mapping, Iterator, Any, Union, Iterable
import numpy as np

from PyHyperparameterSpace.hp.abstract_hp import Hyperparameter
from PyHyperparameterSpace.configuration import HyperparameterConfiguration


class HyperparameterConfigurationSpace(Mapping[str, Hyperparameter]):
    """
    Class to represent a hyperparameter configuration space, where the logic behind the sampling procedure is happening.

        Attributes:
            values (Mapping[str, Hyperparameter]): dictionary, where
                - key := name of hyperparameter
                - value := hyperparameter object
            seed (Union[int, None]): random state of the generator (useful for reproducibility)
    """

    def __init__(
            self,
            values: Mapping[str, Hyperparameter],
            seed: Union[int, None] = None,
    ):
        self._values = values
        self._random = np.random.RandomState(seed)

    def add_hyperparameter(self, hyperparameter: Hyperparameter):
        """
        Adds a hyperparameter to the configuration space.

        Args:
            hyperparameter (Hyperparameter): hyperparameter to add
        """
        hp_name = hyperparameter.get_name()
        if self._values.get(hp_name) is not None:
            # Case: hyperparameter already exist
            raise Exception(f"#ERROR_SPACE: hyperparameter with the name={hp_name} already exists!")
        self._values[hp_name] = hyperparameter

    def add_hyperparameters(self, hyperparameters: Iterable[Hyperparameter]):
        """
        Adds a list of hyperparameters to the configuration space.

        Args:
            hyperparameters (Iterable): list of hyperparameters
        """
        for hp in hyperparameters:
            self.add_hyperparameter(hp)

    def add_condition(self):
        # TODO: Implement here the conditions
        raise NotImplementedError

    def add_conditions(self):
        # TODO: Implement here the conditions
        raise NotImplementedError

    def sample_configuration(self, size: Union[int, None]) \
            -> Union[HyperparameterConfiguration, list[HyperparameterConfiguration]]:
        """
        Returns a sample of configurations (set of hyperparameters).

        Args:
            size (Union[int, Iterable, tuple[int], None]): number of configurations to be drawn

        Returns:
            Union[HyperparameterConfiguration, list[HyperparameterConfiguration]]: single or a list of configurations
        """
        assert size is None or size > 0, f"Illegal size {size}"

        if size is None:
            # Case: Sample one configuration
            size = 1

        configurations = []
        for i in range(size):
            configurations += [
                HyperparameterConfiguration(
                    cs=self,
                    values={key: value.sample(self._random) for key, value in self._values.items()}
                )
            ]
        if size == 1:
            # Case: Return the single sample without the list element
            configurations = configurations[0]
        return configurations

    def get_default_configuration(self) -> HyperparameterConfiguration:
        """
        Returns the configuration that contains all default values for each hyperparameter.

        Returns:
            HyperparameterConfiguration: configuration contains all default values for each hyperparameter
        """
        return HyperparameterConfiguration(
            cs=self,
            values={key: value.get_default() for key, value in self._values.items()}
        )

    def __contains__(self, key: Any) -> bool:
        return key in self._values

    def __getitem__(self, key: str) -> Hyperparameter:
        return self._values.__getitem__(key)

    def __len__(self) -> int:
        return self._values.__len__()

    def __iter__(self) -> Iterator[str]:
        return self._values.__iter__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return dict(self) == dict(other)
        return NotImplemented

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, Hyperparameter):
            self._values[key] = value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        values = dict(self)
        header = "HyperparameterConfiguration(values={"
        lines = [f"  '{key}': {repr(values[key])}," for key in sorted(values.keys())]
        end = "})"
        return "\n".join([header, *lines, end])

    def __getstate__(self) -> dict:
        return {
            "values": self._values,
            "random": self._random,
        }

    def __setstate__(self, state) -> dict:
        self._values = state["values"]
        self._random = state["random"]
