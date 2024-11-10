from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from copy import deepcopy


class GradientBoostingClassificationModel(Model):
    """Gradient Boosting Classification Model"""

    def __init__(self, parameters: dict = None) -> None:
        """
        Initializes a GradientBoostingClassificationModel instance.

        parameters : dict
            A dictionary of parameters for
            the GradientBoostingClassifier model.
            If None, default parameters are used.
        """
        super().__init__(type="classification")
        self._parameters = parameters if parameters else {}
        self._model = GradientBoostingClassifier(**self._parameters)

    @property
    def parameters(self) -> dict:
        """Returns a deep copy of the model parameters."""
        return deepcopy(self._parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the given data."""
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the given data."""
        return self._model.predict(X)

    def to_artifact(self, name: str) -> Artifact:
        """Returns an Artifact object containing the serialized model."""
        return super().to_artifact(name)


class LogisticRegressionModel(Model):
    """Logistic Regression Model"""

    def __init__(self, parameters: dict = None) -> None:
        """
        Initializes a LogisticRegressionModel instance.

        parameters : dict
            A dictionary of parameters for the LogisticRegression model.
            If None, default parameters are used.
        """
        super().__init__(type="classification")
        self._parameters = parameters if parameters else {}
        self._model = LogisticRegression(**self._parameters)

    @property
    def parameters(self) -> dict:
        """Returns a deep copy of the model parameters."""
        return deepcopy(self._parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the given data."""
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the given data."""
        return self._model.predict(X)

    def to_artifact(self, name: str) -> Artifact:
        """Returns an Artifact object containing the serialized model."""
        return super().to_artifact(name)


class RandomForestClassificationModel(Model):
    """Random Forest Classification Model"""

    def __init__(self, parameters: dict = None) -> None:
        """
        Initializes a RandomForestClassificationModel instance.
        parameters : dict
            A dictionary of parameters for the RandomForestClassifier model.
            If None, default parameters are used.
        """
        super().__init__(type="classification")
        self._parameters = parameters if parameters else {}
        self._model = RandomForestClassifier(**self._parameters)

    @property
    def parameters(self) -> dict:
        """Returns a deep copy of the model parameters."""
        return deepcopy(self._parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the given data."""
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the given data."""
        return self._model.predict(X)

    def to_artifact(self, name: str) -> Artifact:
        """Returns an Artifact object containing the serialized model."""
        return super().to_artifact(name)
