from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class LassoRegressionModel(Model):
    """ A class for Lasso Regression model. """
    def __init__(self, parameters: dict = None) -> None:
        """
        Initializes a LassoRegressionModel instance.

        parameters : dict
            A dictionary of parameters for the Lasso model.
            If None, default parameters are used.
        """
        super().__init__(type="regression")
        self._parameters = parameters if parameters else {}
        self._model = Lasso(**self._parameters)

    @property
    def parameters(self) -> dict:
        """Returns a deep copy of the model parameters."""
        return deepcopy(self._parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the data."""
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the input data."""
        return self._model.predict(X)

    def to_artifact(self, name: str) -> Artifact:
        """Serializes the model and returns an Artifact object."""
        return super().to_artifact(name)


class GBRModel(Model):
    """ A class for Gradient Boosting Regression model. """
    def __init__(self, parameters: dict = None) -> None:
        """
        Initializes a GBRModel instance.

        parameters : dict
            A dictionary of parameters for the GradientBoostingRegressor model.
            If None, default parameters are used.
        """
        super().__init__(type="regression")
        self._parameters = parameters if parameters else {}
        self._model = GradientBoostingRegressor(**self._parameters)

    @property
    def parameters(self) -> dict:
        """Returns a deep copy of the model parameters."""
        return deepcopy(self._parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Fits the model to the data."""
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the input data."""
        return self._model.predict(X)

    def to_artifact(self, name: str) -> Artifact:
        """Serializes the model and returns an Artifact object."""
        return super().to_artifact(name)


class RandomForestRegressionModel(Model):
    """ A class for Random Forest Regression model. """
    def __init__(self, parameters: dict = None) -> None:
        """
        Initializes a RandomForestRegressionModel instance.

        parameters : dict
            A dictionary of parameters for the RandomForestRegressor model.
            If None, default parameters are used.
        """
        super().__init__(type="regression")
        self._parameters = parameters if parameters else {}
        self._model = RandomForestRegressor(**self._parameters)

    @property
    def parameters(self) -> dict:
        """Returns a deep copy of the model parameters."""
        return deepcopy(self._parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Fits the model to the data."""
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the input data."""
        return self._model.predict(X)

    def to_artifact(self, name: str) -> Artifact:
        """Serializes the model and returns an Artifact object."""
        return super().to_artifact(name)


class MultipleLinearRegression(Model):
    """ A class for Multiple Linear Regression model. """
    def __init__(self, parameters: dict = None) -> None:

        """
        Initializes a MultipleLinearRegression instance.

        parameters : dict
            A dictionary of parameters for the LinearRegression model.
            If None, default parameters are used.
        """
        super().__init__(type="regression")
        self._parameters = parameters if parameters else {}
        self._model = LinearRegression(**self._parameters)

    @property
    def parameters(self) -> dict:
        """Returns a deep copy of the model parameters."""
        return deepcopy(self._parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the data."""
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the input data."""
        return self._model.predict(X)

    def to_artifact(self, name: str) -> Artifact:
        """Serializes the model and returns an Artifact object."""
        return super().to_artifact(name)
