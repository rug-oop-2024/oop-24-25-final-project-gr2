from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "mean_squared_log_error",
    "r_squared",
    "balanced_accuracy",
    "recall",
    "mcc",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str):
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "mean_squared_log_error":
        return MeanSquaredLogError()
    elif name == "r_squared":
        return RSquared()
    elif name == "balanced_accuracy":
        return BalancedAccuracy()
    elif name == "recall":
        return Recall()
    elif name == "mcc":
        return MCC()
    else:
        raise ValueError(f"Unknown metric name: {name}")


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        pass


# Regression Metrics


class MeanSquaredError(Metric):
    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Calculates Mean Squared Error
        between predictions and ground truth."""
        return np.mean((ground_truth - predictions) ** 2)


class MeanAbsoluteError(Metric):
    def __call__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates Mean Absolute Error
        between predictions and ground truth.
        """
        return np.mean(np.abs(predictions - ground_truth))


class MeanSquaredLogError(Metric):
    def __call__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates Mean Squared Logarithmic Error
        between predictions and ground truth.
        """
        if np.any(predictions < 0) or np.any(ground_truth < 0):
            raise ValueError("This metric does not work on negative values.")

        log_predictions = np.log(1 + predictions)
        log_ground_truth = np.log(1 + ground_truth)

        return np.mean((log_predictions - log_ground_truth) ** 2)


class RSquared(Metric):
    def __call__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates R-squared between predictions and ground truth.
        """
        total_variance = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        residual_variance = np.sum((ground_truth - predictions) ** 2)
        return 1 - (residual_variance / total_variance)


# Classification Metrics


class Accuracy(Metric):
    """Calculates the accuracy of the model."""

    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        return np.mean(ground_truth == predictions)


class ClassifcationMetricUsingCM(Metric):
    """Base class for classification metrics that use confusion matrix."""

    def __init__(self) -> None:
        """Initialize the confusion matrix."""
        self._confusion_matrix: np.ndarray = None

    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Computes and saves the the confusion matrix as an arg.
        Then, returns the metric."""
        self._confusion_matrix = self._compute_confusion_matrix(
            predictions, ground_truth
        )
        return self.compute_metric()

    def _compute_confusion_matrix(
        ground_truth: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute the confusion matrix for multi-class classification.
        """
        classes = np.unique(np.concatenate((predictions, ground_truth)))
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}

        conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)

        for pred, truth in zip(predictions, ground_truth):
            row_idx = class_to_index[truth]
            col_idx = class_to_index[pred]
            conf_matrix[row_idx][col_idx] += 1
        return conf_matrix

    @abstractmethod
    def compute_metric(self) -> float:
        """Abstract method to compute the metric using the confusion matrix."""
        pass


class MCC(ClassifcationMetricUsingCM):
    """Class that uses the confusion matrix to calculate
    the Matthews Correlation Coefficient."""

    def compute_metric(self) -> float:
        """Computes the Matthews Correlation Coefficient."""
        conf_matrix = self._confusion_matrix
        correct_pred = conf_matrix.diagonal().sum()
        class_count = conf_matrix.sum(axis=1)
        pred_class_count = conf_matrix.sum(axis=0)
        total = conf_matrix.sum()

        numerator = (correct_pred * total) - (class_count * pred_class_count)
        denominator = np.sqrt(
            (total**2 - np.sum(pred_class_count**2))
            * (total**2 - np.sum(class_count**2))
        )

        if denominator == 0:
            return 0.0
        else:
            return numerator / denominator


class BalancedAccuracy(ClassifcationMetricUsingCM):
    """Class that uses the confusion matrix
    to calculate the balanced accuracy metric."""

    def compute_metric(self) -> float:
        """Computes the balanced accuracy metric."""

        conf_matrix = self._confusion_matrix
        num_classes = conf_matrix.shape[0]

        recall_per_class = []
        specificity_per_class = []

        for i in range(num_classes):
            true_pos = conf_matrix[i, i]
            false_neg = np.sum(conf_matrix[i, :]) - true_pos

            true_neg = np.sum(conf_matrix) - (
                true_pos + false_neg + np.sum(conf_matrix[:, i]) - true_pos
            )
            false_pos = np.sum(conf_matrix[:, i]) - true_pos

            if (true_pos + false_neg) > 0:
                recall = true_pos / (true_pos + false_neg)
            else:
                recall = 0.0

            if (true_neg + false_pos) > 0:
                specificity = true_neg / (true_neg + false_pos)
            else:
                specificity = 0.0

            recall_per_class.append(recall)
            specificity_per_class.append(specificity)

        balanced_accuracy = (
            np.mean(recall_per_class) + np.mean(specificity_per_class)
        ) / 2
        return balanced_accuracy


class Recall(ClassifcationMetricUsingCM):
    """Class that uses the confusion matrix to calculate the recall metric."""

    def compute_metric(self) -> float:
        conf_matrix = self._confusion_matrix

        true_positives = np.diag(conf_matrix)
        actual_positives = np.sum(conf_matrix, axis=1)

        recall_per_class = np.divide(true_positives, actual_positives)

        return np.mean(recall_per_class)
