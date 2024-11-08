from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
from copy import deepcopy


class GradientBoostingClassificationModel(Model):
    def __init__(self, parameters: dict = None) -> None:
        super().__init__(type="classification")
        self._parameters = parameters if parameters else {}
        self._model = GradientBoostingClassifier(**self._parameters)

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def to_artifact(self, name: str) -> Artifact:
        serialized_model = pickle.dumps(self._model)
        return Artifact(name=name, data=serialized_model)


class LogisticRegressionModel(Model):
    def __init__(self, parameters: dict = None) -> None:
        super().__init__(type="classification")
        self._parameters = parameters if parameters else {}
        self._model = LogisticRegression(**self._parameters)

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def to_artifact(self, name: str) -> Artifact:
        serialized_model = pickle.dumps(self._model)
        return Artifact(name=name, data=serialized_model)


class RandomForestClassificationModel(Model):
    def __init__(self, parameters: dict = None) -> None:
        super().__init__(type="classification")
        self._parameters = parameters if parameters else {}
        self._model = RandomForestClassifier(**self._parameters)

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def to_artifact(self, name: str) -> Artifact:
        serialized_model = pickle.dumps(self._model)
        return Artifact(name=name, data=serialized_model)
