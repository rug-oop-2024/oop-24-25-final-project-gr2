from typing import Literal

from autoop.core.ml.dataset import Dataset


class Feature(Dataset):
    def __init__(self,
                 name: str,
                 feature_type: Literal["numerical", "categorical"]):
        self._name = name
        self._feature_type = feature_type

    def __str__(self):
        return f"Feature(name='{self.name}', type='{self.type}')"

    def __eq__(self, other):
        return (
            isinstance(other, Feature)
            and self.name == other.name
            and self.type == other.type
        )
