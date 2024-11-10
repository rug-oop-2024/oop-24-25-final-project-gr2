from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from typing import Literal
from copy import deepcopy
import pickle


class Model(ABC):
    """Abstract base class for machine learning models."""

    def __init__(self, type: Literal["classification", "regression"]) -> None:
        """Initializes a Model instance."""
        self._type = type
        self._parameters = None

    @property
    def type(self) -> Literal["classification", "regression"]:
        """Returns the type of the model."""
        return self._type

    @property
    def parameters(self) -> dict:
        """Returns a deep copy of the model parameters."""
        return deepcopy(self._parameters)

    def __str__(self) -> str:
        """Returns a string representation of the model."""
        return f"Model(type='{self._type}', parameters='{self._parameters}')"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Abstract method to fit the model to the given data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Abstract method to predict the target variable for given data."""
        pass

    @abstractmethod
    def to_artifact(self, name: str) -> Artifact:
        """Abstract method to serialize the model
        and return an Artifact object."""
        serialized_model = pickle.dumps(self._model)
        return Artifact(name=name, data=serialized_model)
