from typing import Mapping, Iterator, Any, Union
import numpy as np
import yaml
import json


class HyperparameterConfiguration(Mapping[str, Any]):
    """
    Class to represent a hyperparameter configuration space, where the logic behind the sampling procedure is
    happening.

        Attributes:
            values (Mapping[str, Union[str, float, int, bool, None]]): dictionary, where
                - key := name of hyperparameter
                - value := value of the hyperparameter
    """

    def __init__(
            self,
            values: Mapping[str, Union[str, float, int, bool, None]],
    ):
        self._values = values

    def save_yaml(self, path: str):
        """
        Saves the hyperparameter configuration as a .yaml file.

        Args:
            path (str):
                Where the .yaml file should be located
        """
        assert len(path) >= 5, f"Illegal path {path}. It should contain at least one digit before '.yaml'!"
        assert path[-5:] == ".yaml", f"Illegal path {path}. It should contain '.yaml' ending!"

        # Convert each numpy array to list
        self._convert_to_list()

        with open(path, "w") as yaml_file:
            yaml.dump(self._values, yaml_file, default_flow_style=False)

    def save_json(self, path: str):
        """
        Saves the hyperparameter configuration as a .json file.

        Args:
            path (str):
                Where the .json file should be located
        """
        assert len(path) >= 5, f"Illegal path {path}. It should contain at least one digit before '.json'!"
        assert path[-5:] == ".json", f"Illegal path {path}. It should contain '.json' ending!"

        # Convert each numpy array to list
        self._convert_to_list()

        with open(path, "w") as json_file:
            json.dump(self._values, json_file)

    @staticmethod
    def load_yaml(path: str) -> "HyperparameterConfiguration":
        """
        Loads the hyperparameter configuration from a .yaml file.

        Args:
            path (str):
                Where the .yaml file is located

        Returns:
            HyperparameterConfiguration:
                Hyperparameter configuration contains the data from the .yaml file
        """
        assert len(path) >= 5, f"Illegal path {path}. It should contain at least one digit before '.yaml'!"
        assert path[-5:] == ".yaml", f"Illegal path {path}. It should contain '.yaml' ending!"

        with open(path, "r") as yaml_file:
            loaded_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

        cfg = HyperparameterConfiguration(values=loaded_data)
        cfg._convert_to_numpy()
        return cfg

    @staticmethod
    def load_json(path: str) -> "HyperparameterConfiguration":
        """
        Loads the hyperparameter configuration from a .json file.

        Args:
            path (str):
                Where the .json file is located

        Returns:
            HyperparameterConfiguration:
                Hyperparameter configuration contains the data from the .json file
        """
        assert len(path) >= 5, f"Illegal path {path}. It should contain at least one digit before '.json'!"
        assert path[-5:] == ".json", f"Illegal path {path}. It should contain '.json' ending!"

        with open(path, "r") as json_file:
            loaded_data = json.load(json_file)

        cfg = HyperparameterConfiguration(values=loaded_data)
        cfg._convert_to_numpy()
        return cfg

    def _convert_to_list(self):
        """
        Converts each numpy array to a python-list.
        This step is necessary for saving the configuration as a file.
        """
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                self[key] = value.tolist()

    def _convert_to_numpy(self):
        """
        Converts each python-list to a numpy array
        This step is necessary if you load the configuration from a file.
        """
        for key, value in self.items():
            if isinstance(value, list):
                self[key] = np.array(value)

    def __contains__(self, key: Any) -> bool:
        return key in self._values

    def __getitem__(self, key: str) -> Any:
        return self._values.__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        self._values[key] = value

    def __len__(self) -> int:
        return self._values.__len__()

    def __iter__(self) -> Iterator[str]:
        return self._values.__iter__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self) -> str:
        values = dict(self)
        header = "HyperparameterConfiguration(values={"
        lines = [f"  '{key}': {repr(values[key])}," for key in sorted(values.keys())]
        end = "})"
        return "\n".join([header, *lines, end])
