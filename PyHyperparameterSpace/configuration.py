from typing import Any, Iterator, Mapping


class HyperparameterConfiguration(Mapping[str, Any]):
    """
    Class to represent a hyperparameter configuration space, where the logic behind the sampling procedure is
    happening.

        Attributes:
            values (Mapping[str, Any]): dictionary, where
                - key := name of hyperparameter
                - value := value of the hyperparameter
    """

    def __init__(
            self,
            cs: "HyperparameterConfigurationSpace",
            values: Mapping[str, Any],
    ):
        self._cs = cs
        self._values = self._check_configuration(values)

    def _check_configuration(self, values: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Checks if the given configuration space and given configuration are legal. The configuration space should
        match with the possible values from each hyperparameter.

        Args:
            values (Mapping[str, Any]):
                The sampled values for each hyperparameter

        Returns:
            Mapping[str, Any]:
                Legal values
        """
        if self._is_valid_configuration(values):
            return values
        else:
            raise Exception(f"Illegal values {values}!")

    def _is_valid_configuration(
            self,
            values: Mapping[str, Any]
    ) -> bool:
        """
        Returns True if the values matches with the configuration space.

        Args:
            values (Mapping[str, Any]):
                The sampled values for each hyperparameter

        Returns:
            bool:
                True, if values matches with the configuration space
        """
        for key in self._cs:
            if key not in values:
                # Case: Hyperparameter not found
                return False
            else:
                # Case: Hyperparameter found
                return self._cs[key].valid_configuration(values[key])
        return True

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

    def __getstate__(self) -> dict:
        return {
            "cs": self._cs,
            "values": self._values
        }

    def __setstate__(self, state) -> dict:
        self._cs = state["cs"]
        self._values = state["values"]
