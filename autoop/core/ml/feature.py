from typing import Literal

from autoop.core.ml.dataset import Dataset


class Feature(Dataset):
    def __init__(self,
                 name: str,
                 feature_type: Literal["numerical", "categorical"]):
        self._name = name
        self._feature_type = feature_type

    def __str__(self):
        return f"Feature(name='{self._name}', type='{self._feature_type}')"

    def __eq__(self, other):
        return (
            isinstance(other, Feature)
            and self._name == other._name
            and self._feature_type == other._feature_type
        )

    @property
    def feature_type(self) -> Literal["numerical", "categorical"]:
        return self._feature_type

    @property
    def name(self) -> str:
        return self._name
