import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.logits = None
        self.labels = None
        self.probs = None

    def forward(self, logits, labels):
        """
        logits: (batch_size, num_classes)
        labels: (batch_size,)  -> integer class labels
        """

        self.logits = logits
        self.labels = labels

        # Numerical stability trick
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.probs = probs

        batch_size = logits.shape[0]

        # Negative log likelihood
        correct_log_probs = -np.log(probs[np.arange(batch_size), labels])
        loss = np.mean(correct_log_probs)

        return loss

    def backward(self):
        """
        Returns dLoss/dLogits
        """
        batch_size = self.logits.shape[0]
        dZ = self.probs.copy()

        dZ[np.arange(batch_size), self.labels] -= 1
        dZ /= batch_size

        return dZ


class MeanSquaredError:
    def __init__(self):
        self.predictions = None
        self.labels = None

    def forward(self, predictions, labels):
        """
        predictions: logits (batch_size, num_classes)
        labels: integer labels (batch_size,)
        """

        batch_size = predictions.shape[0]
        num_classes = predictions.shape[1]

        # Convert labels to one-hot
        one_hot = np.zeros_like(predictions)
        one_hot[np.arange(batch_size), labels] = 1

        self.predictions = predictions
        self.labels = one_hot

        loss = np.mean((predictions - one_hot) ** 2)

        return loss

    def backward(self):
        batch_size = self.predictions.shape[0]
        dZ = 2 * (self.predictions - self.labels) / batch_size
        return dZ
