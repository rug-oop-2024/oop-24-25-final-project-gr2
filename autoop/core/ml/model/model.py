from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from typing import Literal
from copy import deepcopy


class Model(ABC):
    def __init__(self, type: Literal["classification", "regression"]) -> None:
        self._type = type
        self._parameters = None

    @property
    def type(self) -> Literal["classification", "regression"]:
        return self._type

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    def __str__(self):
        return f"Model(type='{self._type}', parameters='{self._parameters}')"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def to_artifact(self, name: str) -> Artifact:
        pass
