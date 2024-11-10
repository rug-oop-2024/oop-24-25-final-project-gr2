from typing import Literal
from autoop.core.ml.dataset import Dataset


class Feature(Dataset):
    """Feature class. Represents a feature in a dataset."""
    def __init__(self,
                 name: str,
                 feature_type: Literal["numerical", "categorical"]) -> None:
        """
        Initialize a Feature object.
        """
        self._name = name
        self._feature_type = feature_type

    @property
    def name(self) -> str:
        """ Gets the name of the feature."""
        return self._name

    @property
    def feature_type(self) -> Literal["numerical", "categorical"]:
        """ Gets the type of the feature."""
        return self._feature_type

    def __str__(self) -> str:
        """String representation of the Feature object."""
        return f"Feature(name='{self._name}', type='{self._feature_type}')"

    def __eq__(self, other) -> bool:
        """Equality comparison of two Feature objects."""
        return (
            isinstance(other, Feature)
            and self._name == other._name
            and self._feature_type == other._feature_type
        )
