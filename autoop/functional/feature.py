from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    data = dataset.read()
    features = []
    numerical = data.select_dtypes(
        include=["int64", "float64"]
        ).columns.tolist()
    categorical = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for col in numerical:
        features.append(Feature(name=col, feature_type="numerical"))
    for col in categorical:
        features.append(Feature(name=col, feature_type="categorical"))

    return features
