from typing import Literal

from autoop.core.ml.dataset import Dataset


class Feature(Dataset):
    def __init__(self,
                 name: str,
                 feature_type: Literal["numerical", "categorical"]):
        self._name = name
        self._type = feature_type

    def __str__(self):
        return f"Feature(name='{self._name}', type='{self._type}')"

    def __eq__(self, other):
        return (
            isinstance(other, Feature)
            and self._name == other._name
            and self._type == other._type
        )

    @property
    def type(self) -> Literal["numerical", "categorical"]:
        return self._type
