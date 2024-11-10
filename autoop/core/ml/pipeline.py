from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features


import numpy as np
from copy import deepcopy


class Pipeline:
    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split=0.8,
    ):
        """ Initialize the Pipeline object. """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (target_feature.feature_type == "categorical" and
                model.type != "classification"):
            raise ValueError(
                "Model type must be classification \
                for categorical target feature"
            )
        if (target_feature.feature_type == "continuous" and
                model.type != "regression"):
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self):
        """String representation of the Pipeline object."""
        return f"""
        Pipeline(
            model={self._model.type},
            input_features={list(map(str, self._input_features))},
            target_feature={str(self._target_feature)},
            split={self._split},
            metrics={list(map(str, self._metrics))},
        )
        """

    @property
    def model(self):
        """Get the model used in the pipeline"""
        return self._model

    @property
    def metrics(self) -> List[Metric]:
        """Get the metrics used to evaluate the model"""
        return deepcopy(self._metrics)

    @property
    def input_features(self) -> List[Feature]:
        """Get the input features"""
        return deepcopy(self._input_features)

    @property
    def target_feature(self) -> Feature:
        """Get the target feature"""
        return deepcopy(self._target_feature)

    @property
    def artifacts(self) -> List[Artifact]:
        """Used to get the artifacts generated
        during the pipeline execution to be saved"""
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact):
        """Register an artifact with the pipeline"""
        self._artifacts[name] = artifact

    def _preprocess_features(self):
        """Preprocess the input features and target feature"""
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact)
                               in input_results]

    def _split_data(self):
        """Split the data into training and testing sets"""
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):]
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[: int(split *
                                                  len(self._output_vector))]
        self._test_y = self._output_vector[int(split *
                                               len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """ Concatenate the vectors into a single array"""
        return np.concatenate(vectors, axis=1)

    def _train(self):
        """ Train the model on the training data"""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self):
        """
        Evaluate the model on the test data and compute the specified metrics.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def _evaluate_train(self):
        """
        Evaluate the model on the training data and
        compute the specified metrics.
        """
        X_train = self._compact_vectors(self._train_X)
        Y_train = self._train_y
        train_metrics_results = []
        train_predictions = self._model.predict(X_train)
        for metric in self._metrics:
            result = metric(train_predictions, Y_train)
            train_metrics_results.append((metric, result))
        self._train_metrics_results = train_metrics_results
        self._train_predictions = train_predictions

    def execute(self):
        """
        Executes the pipeline by performing the following steps:
        1. Preprocesses the input features.
        2. Splits the data into training and testing sets.
        3. Adjusts the target vectors for classification tasks.
        4. Trains the model on the training data.
        5. Evaluates the model on the test data.
        6. Evaluates the model on the training data.

        Returns:
            dict: A dictionary containing:
                - 'train_metrics': List of metrics results on the training set.
                - 'test_metrics': List of metrics results on the test set.
                - 'train_predictions': Predictions made on the training set.
                - 'test_predictions': Predictions made on the test set.
        """
        self._preprocess_features()
        self._split_data()
        if len(self._train_y.shape) > 1 and self._train_y.shape[1] > 1:
            self._train_y = np.argmax(self._train_y, axis=1)
            self._test_y = np.argmax(self._test_y, axis=1)
        else:
            self._train_y = self._train_y.ravel()
            self._test_y = self._test_y.ravel()

        self._train()
        self._evaluate()
        self._evaluate_train()
        return {
            "train_metrics": self._train_metrics_results,
            "test_metrics": self._metrics_results,
            "train_predictions": self._train_predictions,
            "test_predictions": self._predictions
        }
