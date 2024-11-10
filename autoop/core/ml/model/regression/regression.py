from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
import pickle
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class LassoRegressionModel(Model):
    def __init__(self, parameters: dict = None) -> None:
        super().__init__(type="regression")
        self._parameters = parameters if parameters else {}
        self._model = Lasso(**self._parameters)

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


class GBRModel(Model):
    def __init__(self, parameters: dict = None) -> None:
        super().__init__(type="regression")
        self._parameters = parameters if parameters else {}
        self._model = GradientBoostingRegressor(**self._parameters)

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


class RandomForestRegressionModel(Model):
    def __init__(self, parameters: dict = None) -> None:
        super().__init__(type="regression")
        self._parameters = parameters if parameters else {}
        self._model = RandomForestRegressor(**self._parameters)

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


class MultipleLinearRegression(Model):
    def __init__(self, parameters: dict = None) -> None:
        super().__init__(type="regression")
        self._parameters = parameters if parameters else {}
        self._model = LinearRegression(**self._parameters)

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
