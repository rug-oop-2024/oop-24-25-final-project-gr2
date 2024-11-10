""" Public package interface for the model module. """
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.regression import (
    LassoRegressionModel,
    GBRModel,
    RandomForestRegressionModel,
    MultipleLinearRegression,
)
from autoop.core.ml.model.classification.classification import (
    GradientBoostingClassificationModel,
    LogisticRegressionModel,
    RandomForestClassificationModel,
)

# Lists of available regression and classification models as strings
REGRESSION_MODELS = [
    "LassoRegressionModel",
    "GBRModel",
    "RandomForestRegressionModel",
    "MultipleLinearRegression",
]

CLASSIFICATION_MODELS = [
    "GradientBoostingClassificationModel",
    "LogisticRegressionModel",
    "RandomForestClassificationModel",
]

# Mapping from model name strings to their corresponding classes
_MODEL_CLASS_MAPPING = {
    **{
        model_name: cls
        for model_name, cls in zip(
            REGRESSION_MODELS,
            [
                LassoRegressionModel,
                GBRModel,
                RandomForestRegressionModel,
                MultipleLinearRegression,
            ],
        )
    },
    **{
        model_name: cls
        for model_name, cls in zip(
            CLASSIFICATION_MODELS,
            [
                GradientBoostingClassificationModel,
                LogisticRegressionModel,
                RandomForestClassificationModel,
            ],
        )
    },
}


def get_model(model_name: str) -> Model:
    """
    Factory function to get a model instance by its name.

    Args:
        model_name (str): The name of the model to instantiate.

    Returns:
        Model: An instance of the requested model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    try:
        model_class = _MODEL_CLASS_MAPPING[model_name]
        return model_class()
    except KeyError:
        available_models = REGRESSION_MODELS + CLASSIFICATION_MODELS
        raise ValueError(
            f"Model '{model_name}' is not recognized. "
            f"Available models: {', '.join(available_models)}"
        )
