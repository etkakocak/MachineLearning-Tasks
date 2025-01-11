from sklearn.metrics import confusion_matrix

class ROCAnalysis:
    """
    Class to calculate various metrics for Receiver Operating Characteristic (ROC) analysis.

    Attributes:
        y_pred (list): Predicted labels.
        y_true (list): True labels.
        tp (int): Number of true positives.
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
    """

    def __init__(self, y_predicted, y_true):
        """
        Initialize ROCAnalysis object.

        Parameters:
            y_predicted (list): Predicted labels (0 or 1).
            y_true (list): True labels (0 or 1).
        """
        self.y_pred = y_predicted
        self.y_true = y_true
        self.tp, self.fp, self.fn, self.tn = confusion_matrix(y_true, y_predicted).ravel()

    def tp_rate(self):
        """
        Calculate True Positive Rate (Sensitivity, Recall).

        Returns:
            float: True Positive Rate.
        """
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0

    def fp_rate(self):
        """
        Calculate False Positive Rate.

        Returns:
            float: False Positive Rate.
        """
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0

    def precision(self):
        """
        Calculate Precision.

        Returns:
            float: Precision.
        """
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0

    def f_score(self, beta=1):
        """
        Calculate the F-score.

        Parameters:
            beta (float, optional): Weighting factor for precision in the harmonic mean. Defaults to 1.

        Returns:
            float: F-score.
        """
        precision = self.precision()
        recall = self.tp_rate()
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall) if (precision + recall) else 0
