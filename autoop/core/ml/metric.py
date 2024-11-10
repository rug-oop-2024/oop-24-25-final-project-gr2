from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "r_squared",
    "balanced_accuracy",
    "recall",
    "hamming_loss",
    "max_error"
]


def get_metric(name: str) -> "Metric":
    """Returns the metric object based on the name."""
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r_squared":
        return RSquared()
    elif name == "balanced_accuracy":
        return BalancedAccuracy()
    elif name == "recall":
        return Recall()
    elif name == "hamming_loss":
        return HammingLossMetric()
    elif name == "max_error":
        return MaxError()
    else:
        raise ValueError(f"Unknown metric name: {name}")


class Metric(ABC):
    """Base class for all metrics."""

    def __init__(self) -> None:
        """ Initializes the name of the metric."""
        self._name = ""

    @property
    def name(self) -> str:
        """ Returns the name of the metric."""
        return self._name

    @abstractmethod
    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """ Abstract method to calculate the metric."""
        pass


# Regression Metrics


class MeanSquaredError(Metric):
    """ Metric for the Mean Squared Error."""
    def __init__(self) -> None:
        """ Initializes the name of the metric."""
        self._name = "mean_squared_error"

    @property
    def name(self) -> str:
        """ Returns the name of the metric."""
        return self._name

    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Calculates Mean Squared Error
        between predictions and ground truth."""
        return np.mean((ground_truth - predictions) ** 2)


class MeanAbsoluteError(Metric):
    """"Metric for the Mean Absolute Error."""
    def __init__(self) -> None:
        """ Initializes the name of the metric."""
        self._name = "mean_absolute_error"

    @property
    def name(self) -> str:
        """ Returns the name of the metric."""
        return self._name

    def __call__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates Mean Absolute Error
        between predictions and ground truth.
        """
        return np.mean(np.abs(predictions - ground_truth))


class MaxError(Metric):
    """Metric for the Maximum Error."""
    def __init__(self) -> None:
        """Initializes the name of the metric."""
        self._name = "max_error"

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return self._name

    def __call__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates the Maximum Error between predictions and ground truth.

        Args:
            predictions (np.ndarray): Model predictions.
            ground_truth (np.ndarray): True target values.

        Returns:
            float: The computed Maximum Error.
        """

        absolute_errors = np.abs(ground_truth - predictions)
        return np.max(absolute_errors)


class RSquared(Metric):
    """Metric for the R-squared."""
    def __init__(self) -> None:
        """ Initializes the name of the metric."""
        self._name = "r_squared"

    @property
    def name(self) -> str:
        """ Returns the name of the metric."""
        return self._name

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

    def __init__(self) -> None:
        """ Initializes the name of the metric."""
        self._name = "accuracy"

    @property
    def name(self) -> str:
        """ Returns the name of the metric."""
        return self._name

    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        return np.mean(ground_truth == predictions)


class ClassifcationMetricUsingCM(Metric):
    """Base class for classification metrics that use confusion matrix."""

    def __init__(self) -> None:
        """Initialize the confusion matrix."""
        self._confusion_matrix: np.ndarray = None
        self._ground_truth: np.ndarray = None
        self._predictions: np.ndarray = None

    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Computes and saves the the confusion matrix as an arg.
        Then, returns the metric."""
        self._confusion_matrix = self._compute_confusion_matrix(
            predictions, ground_truth
        )

        self._ground_truth = ground_truth
        self._predictions = predictions
        self._confusion_matrix = self._compute_confusion_matrix(
            ground_truth, predictions
        )
        return self.compute_metric()

    def _compute_confusion_matrix(self, ground_truth: np.ndarray,
                                  predictions: np.ndarray) -> np.ndarray:
        """
        Compute the confusion matrix for multi-class classification.
        """
        joint_classes = np.unique(np.concatenate((ground_truth, predictions)))
        classes = joint_classes.flatten()
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


class HammingLossMetric(ClassifcationMetricUsingCM):
    """Class that uses the confusion matrix
    to calculate the Hamming Loss metric."""

    def __init__(self) -> None:
        """ Initializes the name of the metric."""
        self._name = "hamming_loss"

    @property
    def name(self) -> str:
        """ Returns the name of the metric."""
        return self._name

    def compute_metric(self) -> float:
        """Computes the Hamming Loss metric."""
        conf_matrix = self._confusion_matrix
        total_samples = conf_matrix.sum()
        incorrect_predictions = total_samples - np.diag(conf_matrix).sum()
        if total_samples > 0:
            hamming_loss = incorrect_predictions / total_samples
        else:
            hamming_loss = 0.0
        return hamming_loss


class BalancedAccuracy(ClassifcationMetricUsingCM):
    """Class that uses the confusion matrix
    to calculate the balanced accuracy metric."""

    def __init__(self) -> None:
        """ Initializes the name of the metric."""
        self._name = "balanced_accuracy"

    @property
    def name(self) -> str:
        """ Returns the name of the metric."""
        return self._name

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

    def __init__(self) -> None:
        """ Initializes the name of the metric."""
        self._name = "recall"

    @property
    def name(self) -> str:
        """ Returns the name of the metric."""
        return self._name

    def compute_metric(self) -> float:
        """Computes the recall metric."""

        conf_matrix = self._confusion_matrix

        true_positives = np.diag(conf_matrix)
        actual_positives = np.sum(conf_matrix, axis=1)

        recall_per_class = np.divide(true_positives, actual_positives)

        return np.mean(recall_per_class)
