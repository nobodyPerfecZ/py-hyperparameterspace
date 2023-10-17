from typing import Mapping, Iterator, Any, Union


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

    def __contains__(self, key: Any) -> bool:
        return key in self._values

    def __getitem__(self, key: str) -> Any:
        return self._values.__getitem__(key)

    def __len__(self) -> int:
        return self._values.__len__()

    def __iter__(self) -> Iterator[str]:
        return self._values.__iter__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return dict(self) == dict(other)
        return NotImplemented

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self) -> str:
        values = dict(self)
        header = "HyperparameterConfiguration(values={"
        lines = [f"  '{key}': {repr(values[key])}," for key in sorted(values.keys())]
        end = "})"
        return "\n".join([header, *lines, end])
