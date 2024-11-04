
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
    
    for column in data.columns:
        if data[column].dtype == "category":
            feature_type = "categorical"
        elif data[column].dtype == "int64" or data[column].dtype == "float64":
            feature_type = "numerical"
        else:
            raise ValueError("Feature type not supported.")
        features.append(Feature(name=column, type=feature_type))
    
    return features
    # raise NotImplementedError("This should be implemented by you.")
